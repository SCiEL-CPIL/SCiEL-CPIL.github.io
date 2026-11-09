"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-restart CMA-ES with deadline guard for f22."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 300s kill
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = float('inf')
        calls_made = 0
        budget = self.budget
        dim = self.dim

        def elapsed():
            return time.time() - start_time

        def remaining():
            return budget - calls_made

        def time_ok(margin=5.0):
            return elapsed() < deadline - margin

        # Phase 1: Structured initial sampling to find promising basins
        # Use a mix of grid points and random points
        n_random = min(40, budget // 10)
        
        # Grid-based sampling (5x5x5x5x5 is too many, use 3^5=243 but we have limited budget)
        # Use a coarser grid: 3 points per dimension = 243 points, but we only have ~40 calls
        # So use a smarter approach: Latin hypercube or stratified sampling
        
        # Stratified sampling: divide each dimension into 3 intervals, pick one from each combination
        # But 3^5 = 243 > 40, so use a subset
        n_strata = 3
        points_per_dim = np.linspace(-4, 4, n_strata)
        
        # Generate all combinations but sample a subset
        import itertools
        all_combos = list(itertools.product(points_per_dim, repeat=dim))
        # Shuffle and take first n_random
        rng.shuffle(all_combos)
        
        for i in range(n_random):
            if calls_made >= budget or not time_ok():
                break
            if i < len(all_combos):
                x = np.array(all_combos[i], dtype=float)
            else:
                x = rng.uniform(-4, 4, dim)
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Also add some random points
        for i in range(n_random // 2):
            if calls_made >= budget or not time_ok():
                break
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
            calls_made += 1

        # Phase 2: CMA-ES from the best starting point
        def run_cmaes(x0, sigma0, max_calls, deadline_check):
            """Run CMA-ES with call limit and deadline check."""
            nonlocal best_f, best_x, calls_made
            
            x = x0.copy()
            sigma = sigma0
            # CMA-ES parameters
            c1 = 1 / (dim + 3)
            cmu = 2 / ((dim + 2) ** 2 + 5)
            cc = (4 + dim / 4) / (dim + 4 + 2 / dim)
            cs = (dim + 1.7) / (dim + 9)
            damps = 1 + 2 * max(0, np.sqrt((dim - 1) / (dim + 3)) - 1) + cs
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
            
            # Population size
            lambda_ = 4 + int(np.floor(3 * np.log(dim)))
            mu = lambda_ // 2
            weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1 / np.sum(weights ** 2)
            
            # Covariance matrix
            C = np.eye(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            
            local_best_f = float('inf')
            local_best_x = x0.copy()
            
            for gen in range(200):
                if calls_made >= max_calls or not deadline_check():
                    break
                
                # Generate lambda_ offspring
                offspring = []
                # Use Cholesky decomposition for efficiency
                try:
                    L = np.linalg.cholesky(C + 1e-12 * np.eye(dim))
                except np.linalg.LinAlgError:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.maximum(eigvals, 1e-12)
                    L = eigvecs @ np.diag(np.sqrt(eigvals))
                
                for i in range(lambda_):
                    if calls_made >= max_calls or not deadline_check():
                        break
                    z = rng.standard_normal(dim)
                    y = L @ z
                    xi = x + sigma * y
                    xi = np.clip(xi, -5, 5)
                    fi = func(xi)
                    calls_made += 1
                    offspring.append((fi, xi))
                
                if len(offspring) == 0:
                    break
                    
                # Sort by fitness
                offspring.sort(key=lambda t: t[0])
                
                # Update best
                for fi, xi in offspring:
                    if fi < local_best_f:
                        local_best_f = fi
                        local_best_x = xi.copy()
                    if fi < best_f:
                        best_f = fi
                        best_x = xi.copy()
                
                # Take top mu
                mu_actual = min(mu, len(offspring))
                selected = [offspring[i][1] for i in range(mu_actual)]
                
                # Compute mean of selected
                new_x = np.zeros(dim)
                for i in range(mu_actual):
                    new_x += weights[i] * selected[i]
                
                # Update step size (CSA)
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) / chi_n * (C @ (new_x - x))
                hsig = np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (gen + 1))) < (1 + 2 / (dim + 1)) * chi_n
                sigma *= np.exp(cs / damps * (np.linalg.norm(ps) / chi_n - 1))
                sigma = np.clip(sigma, 1e-10, 50)
                
                # Update covariance
                v = (new_x - x) / sigma
                C = (1 - c1 - cmu) * C + c1 * (v @ v.T + (1 - hsig) * cc * (2 - cc) * pc @ pc.T) + \
                    cmu * sum(weights[i] * np.outer(selected[i] - x, (selected[i] - x)) / sigma ** 2 for i in range(mu_actual))
                
                # Update pc
                pc = (1 - cc) * pc + cc * hsig * v
                
                x = new_x
                
                # Ensure C is symmetric positive definite
                C = (C + C.T) / 2
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T

                # Check for convergence
                if sigma < 1e-8:
                    break

            return local_best_f, local_best_x

        # Run CMA-ES from the best point
        # Use most of the remaining budget for CMA-ES
        cma_budget = min(remaining() - 10, 350)  # Leave 10 calls for potential restart
        run_cmaes(best_x, 1.5, cma_budget, lambda: time_ok(5.0))

        # Phase 3: If we have time and budget, do a few more restarts from different basins
        if remaining() > 20 and time_ok(10.0):
            # Try a few more starting points from the initial samples
            # Use the second and third best initial points
            # For simplicity, just do one more CMA-ES from a slightly perturbed best point
            perturbation = 1.0
            x_restart = best_x + rng.normal(0, perturbation, dim)
            x_restart = np.clip(x_restart, -5, 5)
            cma_budget2 = min(remaining() - 5, 150)
            if cma_budget2 > 10:
                run_cmaes(x_restart, 1.0, cma_budget2, lambda: time_ok(5.0))

        return best_x, best_f
# EVOLVE-BLOCK-END
