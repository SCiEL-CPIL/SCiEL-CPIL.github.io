"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-restart CMA-ES with adaptive sigma for f22 (Gallagher 21 peaks)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s kill
        budget = self.budget
        dim = self.dim
        rng = np.random.default_rng(42)

        best_x = None
        best_f = float('inf')
        calls_made = 0

        def elapsed():
            return time.time() - start_time

        def time_ok():
            return elapsed() < deadline - 5

        def remaining():
            return budget - calls_made

        def safe_eval(x):
            nonlocal calls_made, best_f, best_x
            if calls_made >= budget or not time_ok():
                return float('inf')
            x = np.clip(x, -5.0, 5.0)
            fx = func(x)
            calls_made += 1
            if np.isfinite(fx) and fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx if np.isfinite(fx) else float('inf')

        # Phase 1: Random sampling to find diverse basins
        n_random = min(30, budget // 15)
        for i in range(n_random):
            if calls_made >= budget or not time_ok():
                break
            x = rng.uniform(-5, 5, dim)
            fx = safe_eval(x)
        
        if best_x is None:
            best_x = np.zeros(dim)
            safe_eval(best_x)

        # Collect top points for multi-restart
        # We need to track points from random phase, but we didn't store them all.
        # Let's just use the best one found so far as the primary seed,
        # and generate a few perturbed seeds for restarts.
        
        # Actually, a better strategy for f22 (multimodal) is to run CMA-ES
        # with a moderate population, and if it converges too fast (sigma < eps),
        # restart from the best point found so far with a larger sigma, or
        # restart from a random point if the current one is stuck.
        
        # Let's implement a single robust CMA-ES loop with adaptive restarts.
        
        # CMA-ES parameters
        lambda_ = 10
        mu = lambda_ // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1 / np.sum(weights ** 2)

        c1 = 1 / (dim + 3)
        cmu = 2 / ((dim + 2) ** 2 + 5)
        cc = (4 + dim / 4) / (dim + 4 + 2 / dim)
        cs = (dim + 1.7) / (dim + 9)
        damps = 1 + 2 * max(0, np.sqrt((dim - 1) / (dim + 3)) - 1) + cs
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # Initial state
        x = best_x.copy()
        sigma = 1.0
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        
        try:
            L = np.linalg.cholesky(C)
        except np.linalg.LinAlgError:
            L = np.eye(dim)

        # We will run CMA-ES in a loop. 
        # If sigma drops below 1e-8, we consider it converged.
        # Then we restart from the global best found so far (or a random point if stuck).
        
        max_restarts = 5
        restarts = 0
        
        while restarts < max_restarts and calls_made < budget and time_ok():
            # Run CMA-ES until convergence or budget/time limit
            gen = 0
            converged = False
            
            while not converged and calls_made < budget and time_ok():
                gen += 1
                
                # Generate offspring
                offspring = []
                for i in range(lambda_):
                    if calls_made >= budget or not time_ok():
                        break
                    z = rng.standard_normal(dim)
                    y = L @ z
                    xi = x + sigma * y
                    fi = safe_eval(xi)
                    offspring.append((fi, xi))
                
                if len(offspring) < mu:
                    converged = True
                    break
                
                # Sort by fitness
                offspring.sort(key=lambda t: t[0])
                
                # Update best
                for fi, xi in offspring:
                    if fi < best_f:
                        best_f = fi
                        best_x = xi.copy()
                        
                # Selection
                mu_actual = min(mu, len(offspring))
                selected = [offspring[i][1] for i in range(mu_actual)]
                
                # Compute weighted mean
                new_x = np.zeros(dim)
                for i in range(mu_actual):
                    new_x += weights[i] * selected[i]
                
                # Update step size (CSA)
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) / chi_n * (C @ (new_x - x))
                hsig = np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * gen)) < (1 + 2 / (dim + 1)) * chi_n
                sigma *= np.exp(cs / damps * (np.linalg.norm(ps) / chi_n - 1))
                sigma = np.clip(sigma, 1e-12, 50)
                
                # Update covariance
                v = (new_x - x) / sigma
                rank_one = c1 * (v @ v.T + (1 - hsig) * cc * (2 - cc) * pc @ pc.T)
                rank_mu = cmu * sum(weights[i] * np.outer(selected[i] - x, selected[i] - x) / sigma ** 2 for i in range(mu_actual))
                C = (1 - c1 - cmu) * C + rank_one + rank_mu
                
                # Update pc
                pc = (1 - cc) * pc + cc * hsig * v
                
                x = new_x
                
                # Ensure C is symmetric positive definite and update Cholesky
                C = (C + C.T) / 2
                try:
                    L = np.linalg.cholesky(C + 1e-12 * np.eye(dim))
                except np.linalg.LinAlgError:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.maximum(eigvals, 1e-12)
                    C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                    L = np.linalg.cholesky(C + 1e-12 * np.eye(dim))
                
                # Check convergence
                if sigma < 1e-8:
                    converged = True
            
            # If we converged, we might be stuck in a local minimum.
            # Restart from the global best found so far with a larger sigma.
            # Or, if we used up budget/time, break.
            if not (calls_made < budget and time_ok()):
                break
                
            restarts += 1
            # Restart from global best, with a fresh covariance and larger sigma
            x = best_x.copy()
            sigma = 2.0 / (restarts + 1)  # Decreasing sigma for subsequent restarts
            C = np.eye(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                L = np.eye(dim)

        # Final local refinement with Nelder-Mead if budget remains
        if calls_made < budget - 10 and time_ok():
            from scipy.optimize import minimize
            x0 = best_x.copy()
            try:
                res = minimize(
                    safe_eval,
                    x0,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 100,
                        'xatol': 1e-10,
                        'fatol': 1e-10,
                        'adaptive': True
                    }
                )
                if np.isfinite(res.fun) and res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
