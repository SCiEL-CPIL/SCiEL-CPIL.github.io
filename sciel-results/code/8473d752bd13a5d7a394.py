"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """CMA-ES hybrid optimizer for shifted/rotated Rosenbrock-style f8."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        best_x = None
        best_f = np.inf
        call_count = 0
        dim = self.dim
        budget = self.budget

        def safe_call(x):
            nonlocal best_x, best_f, call_count
            if call_count >= budget:
                return np.inf
            call_count += 1
            x = np.asarray(x, dtype=np.float64)
            x_clipped = np.clip(x, -5.0, 5.0)
            try:
                fx = float(func(x_clipped))
            except Exception:
                fx = np.inf
            if np.isfinite(fx) and fx < best_f:
                best_f = fx
                best_x = x_clipped.copy()
            return fx

        # Phase 1: CMA-ES global search
        # CMA-ES is robust to high condition number and non-convexity
        # Unlike L-BFGS-B which can get stuck in narrow valleys, CMA-ES
        # adapts its covariance matrix to the problem landscape
        
        def cma_es_search(func_wrapper, x0, sigma, budget_remaining, deadline):
            """Simple CMA-ES implementation."""
            nonlocal call_count, best_x, best_f
            
            dim = self.dim
            # Population size
            lambda_ = 4 + int(3 * np.log(dim))
            
            # CMA-ES parameters
            cc = (4 + lambda_ / dim) / (dim + 4 + 2 * lambda_ / dim)
            cs = (lambda_ / 2 + 1) / (dim + lambda_ / 2 + 3)
            cmu = min(1 - cc, 2 * (sqrt2 - 1) / (dim + 2))
            damps = 1 + 2 * max(0, sqrt((lambda_ / 2 + 1) * (dim - 1) / (dim + 1) - 3)) + cs
            sigma_decay = (lambda_ / 2) **(-1 / (dim + 20))
            
            # Initial values
            mean = np.array(x0, dtype=np.float64)
            sigma = float(sigma)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            C = np.eye(dim)
            inv_C = np.eye(dim)
            step_size = np.sqrt((1 + 2 / (dim + 1)) * (1 - cc) / (cc * damps) + 1)
            
            # Precompute Cholesky factorization update
            chol_C = np.eye(dim)
            
            for gen in range(1000):
                if call_count >= budget or time.time() > deadline:
                    break
                
                # Generate population
                pop = []
                for i in range(lambda_):
                    z = np.random.randn(dim)
                    # Clip to box
                    xi = np.clip(mean + sigma * (chol_C @ z), -5.0, 5.0)
                    pop.append(xi)
                
                # Evaluate population
                fvals = []
                for xi in pop:
                    fx = func_wrapper(xi)
                    fvals.append(fx)
                
                fvals = np.array(fvals)
                sorted_idx = np.argsort(fvals)
                pop = [pop[i] for i in sorted_idx]
                fvals = fvals[sorted_idx]
                
                # Update mean (weighted recombination)
                weights = np.zeros(lambda_)
                m = min(lambda_ // 2, int(1 / cmu))
                for i in range(m):
                    weights[i] = np.log(m + 0.5) - np.log(i + 1)
                mu_eff = (weights.sum()) ** 2 / (weights ** 2).sum()
                weights /= weights.sum()
                
                new_mean = np.sum(weights[:, None] * np.array(pop), axis=0)
                
                # Update step-size control (CSA)
                ps = (1 - cs) * ps + cs / step_size * (chol_C.T @ ((new_mean - mean) / sigma))
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (gen + 1)))) < \
                       (1 + 2 / (dim + 1)) * np.sqrt(dim + 1 - 1)
                
                # Update evolution path
                pc = (1 - cc) * pc + hsig * cc / step_size * (new_mean - mean) / sigma
                
                # Update covariance matrix
                delta_hs = (1 - hsig) * cc * (2 - cc)
                rank_one = np.outer(pc, pc)
                rank_mu = np.sum(weights[:, None] * np.outer(np.array(pop) - new_mean, 
                                                              np.array(pop) - new_mean), axis=0)
                C = (1 + cs * damps * (cc - 1) + cs * delta_hs - (1 + 2 * cs * damps) * cc / (dim + 1)) * C \
                    + cs * damps * cc / (dim + 1) * (rank_one + rank_mu)
                
                # Ensure symmetry
                C = (C + C.T) / 2
                
                # Update sigma
                sigma *= np.exp(cs / damps * (np.linalg.norm(ps) / np.sqrt(dim) - 1))
                
                # Update Cholesky factorization
                try:
                    chol_C = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    # Regularize if Cholesky fails
                    C = C + 1e-10 * np.eye(dim)
                    try:
                        chol_C = np.linalg.cholesky(C)
                    except:
                        break
                
                mean = new_mean
                
                # Check convergence
                if np.all(np.abs(fvals[:10] - fvals[0]) < 1e-10):
                    break
                
                # Check if budget is exhausted
                if call_count >= budget:
                    break
            
            return mean, sigma

        # Phase 1: Run CMA-ES from a few diverse starting points
        rng = np.random.default_rng(42)
        
        # Strategic starting points for Rosenbrock
        start_points = [
            np.zeros(dim),
            np.ones(dim),
            np.full(dim, 0.5),
            np.full(dim, -1.0),
            np.full(dim, 2.0),
            np.full(dim, -2.0),
            np.full(dim, 3.0),
            np.full(dim, -3.0),
        ]
        
        # Add some random points
        for _ in range(4):
            start_points.append(rng.uniform(-4, 4, dim))
        
        # Run CMA-ES from the best few starting points
        # First, evaluate starting points to rank them
        scored_starts = []
        for sp in start_points:
            if call_count >= budget or time.time() > deadline - 100:
                break
            fx = safe_call(sp)
            if np.isfinite(fx):
                scored_starts.append((fx, sp))
        
        scored_starts.sort(key=lambda x: x[0])
        
        # Run CMA-ES from top 3 starting points
        max_cma_starts = min(3, len(scored_starts))
        for i in range(max_cma_starts):
            if call_count >= budget or time.time() > deadline - 50:
                break
            
            fx0, x0 = scored_starts[i]
            # Adaptive sigma based on problem scale
            sigma0 = 2.0
            cma_es_search(safe_call, x0, sigma0, budget - call_count, deadline)
        
        # Phase 2: Local polish with L-BFGS-B if budget remains
        if best_x is not None and call_count < budget and time.time() < deadline - 20:
            try:
                bounds = [(-5.0, 5.0)] * dim
                result = minimize(
                    lambda x: safe_call(x),
                    best_x,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={
                        'maxiter': 100,
                        'maxfun': max(1, budget - call_count - 5),
                        'ftol': 1e-12,
                        'gtol': 1e-8,
                    }
                )
            except Exception:
                pass
        
        # Phase 3: Final polish with Nelder-Mead if budget remains
        if best_x is not None and call_count < budget and time.time() < deadline - 10:
            try:
                remaining = max(1, budget - call_count - 5)
                result = minimize(
                    lambda x: safe_call(x),
                    best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 50,
                        'maxfev': remaining,
                        'xatol': 1e-10,
                        'fatol': 1e-12,
                        'adaptive': True,
                    }
                )
            except Exception:
                pass

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = float(func(best_x))
        
        return best_x, best_f
# EVOLVE-BLOCK-END
