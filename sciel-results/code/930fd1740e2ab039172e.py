"""Optimizer for BBOB-style f11 (Discus) using multi-start CMA-ES with adaptive restarts."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


class Optimizer:
    """Multi-start CMA-ES with L-BFGS-B polishing for BBOB f11 (Discus)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _cma_es(self, func, x0, sigma0, max_calls, call_tracker, bounds):
        """Run a simple CMA-ES implementation."""
        dim = len(x0)
        x = x0.copy()
        sigma = sigma0
        best_x = x0.copy()
        best_f = func(x0)
        call_tracker[0] += 1

        # Population size
        lambda_ = max(4, int(4 + 3 * np.log(dim)))
        
        # Covariance matrix
        C = np.eye(dim)
        # Step sizes for mean and covariance
        cmc = 1 + 2 * max(0, np.sqrt((lambda_ / 2 - 1) / (dim + 2)) - 1)
        cc = (4 + lambda_ / dim) / (dim + 4 + 2 * lambda_ / dim)
        cs = (lambda_ / 2 + 2) / (dim + lambda_ / 2 + 5)
        c1 = (2 / ((dim + 1.3) ** 2 + lambda_)) / (1 - cc)
        cmu = min(0.5 - c1, 2 * (2 - (dim + 2) / (dim + 1.3) ** 2) / ((dim + 2) ** 2 + lambda_))

        # Cumulation paths
        ps = np.zeros(dim)
        pc = np.zeros(dim)
        hs = np.sqrt((1 - (1 - cc) ** (1 + 1 / cmc)) / (1 - (1 - cc) ** 2))
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        while call_tracker[0] < max_calls:
            # Generate population
            Y = np.random.randn(lambda_, dim)
            pop = x + sigma * (Y @ np.linalg.cholesky(C).T)
            
            # Clip to bounds
            pop = np.clip(pop, bounds[0], bounds[1])
            
            # Evaluate
            fvals = np.array([func(xi) for xi in pop])
            call_tracker[0] += lambda_
            
            # Update best
            if np.min(fvals) < best_f:
                best_f = np.min(fvals)
                best_x = pop[np.argmin(fvals)].copy()

            # Sort by fitness
            order = np.argsort(fvals)
            mu = lambda_ // 2  # Number of parents
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mueff = 1 / np.sum(weights ** 2)
            
            # Update mean
            x_old = x.copy()
            x = x + sigma * np.sum(weights[:, None] * (pop[order[:mu]] - x), axis=0)
            x = np.clip(x, bounds[0], bounds[1])
            
            # Update step size
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) / chi_n * np.linalg.cholesky(C) @ (x_old - x) / sigma
            if np.linalg.norm(ps) / chi_n < (1.4 + 1 / cs) * np.sqrt(1 - (1 - cs) ** 2):
                sigma *= np.exp(cs / (dim + 2) * (np.linalg.norm(ps) / chi_n - 1))
            
            # Update covariance
            pc = (1 - cc) * pc + cc * np.sqrt(cc * (2 - cc) * mueff) / np.linalg.cholesky(C) @ (x_old - x) / sigma
            rank_mu = np.outer(weights @ (pop[order[:mu]] - x) / sigma, (weights @ (pop[order[:mu]] - x) / sigma).T)
            C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + cc * (2 - cc) * C) + cmu * rank_mu
            C = 0.5 * (C + C.T)  # Symmetrize
            
            # Ensure positive definiteness
            eigvals = np.linalg.eigvalsh(C)
            if np.min(eigvals) < 1e-12:
                C += (1e-12 - np.min(eigvals)) * np.eye(dim)

        return best_x, best_f

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit
        
        dim = self.dim
        budget = self.budget
        bounds = (np.full(dim, -5.0), np.full(dim, 5.0))
        
        # Track function calls
        call_tracker = [0]
        
        def wrapped_func(x):
            call_tracker[0] += 1
            return func(x)
        
        best_x = None
        best_f = np.inf
        
        # Phase 1: Initial global exploration with diverse starting points
        # Use a combination of random, stratified, and structured points
        rng = np.random.default_rng(42)
        
        # Generate diverse starting points
        n_initial = min(15, budget // 40)
        
        # Random points
        random_starts = rng.uniform(-4, 4, size=(n_initial // 2, dim))
        
        # Stratified points
        stratified_starts = np.zeros((n_initial - n_initial // 2, dim))
        for i in range(n_initial - n_initial // 2):
            for d in range(dim):
                stratified_starts[i, d] = -4 + 8 * ((i + 0.5) / (n_initial - n_initial // 2) + rng.random() / (n_initial - n_initial // 2))
        
        # Add center and some structured points
        center = np.zeros(dim)
        structured = np.array([
            np.full(dim, 3.0),
            np.full(dim, -3.0),
            np.array([3, -3, 3, -3, 3][:dim]),
            np.array([-3, 3, -3, 3, -3][:dim]),
        ])
        
        all_starts = np.vstack([random_starts, stratified_starts, [center], structured])
        all_starts = np.clip(all_starts, bounds[0], bounds[1])
        
        # Evaluate all starting points
        start_values = []
        for x0 in all_starts:
            if call_tracker[0] >= budget - 10:
                break
            if time.time() > deadline:
                break
            
            fx0 = wrapped_func(x0)
            start_values.append((fx0, x0.copy()))
            if fx0 < best_f:
                best_f = fx0
                best_x = x0.copy()
        
        # Sort by function value
        start_values.sort(key=lambda x: x[0])
        
        # Phase 2: CMA-ES from top starting points
        # Allocate budget based on remaining calls
        remaining = budget - call_tracker[0]
        
        # Try CMA-ES from the best few starts
        n_cma_starts = min(3, len(start_values))
        
        for i in range(n_cma_starts):
            if call_tracker[0] >= budget - 20:
                break
            if time.time() > deadline:
                break
            
            fx0, x0 = start_values[i]
            
            # Adaptive sigma based on function value
            sigma0 = 1.0 if i == 0 else 0.5
            
            try:
                # Run CMA-ES with a portion of the remaining budget
                cma_budget = max(30, remaining // (n_cma_starts - i))
                result_x, result_f = self._cma_es(
                    wrapped_func, x0, sigma0, cma_budget, call_tracker, bounds
                )
                
                if result_f < best_f:
                    best_f = result_f
                    best_x = result_x.copy()
                    
            except Exception:
                continue
        
        # Phase 3: L-BFGS-B polishing from the best point
        if best_x is not None and call_tracker[0] < budget - 10:
            if time.time() < deadline:
                try:
                    result = minimize(
                        wrapped_func,
                        best_x,
                        method='L-BFGS-B',
                        bounds=[(-5, 5)] * dim,
                        options={'maxiter': 100, 'maxls': 20}
                    )
                    
                    if result.fun < best_f:
                        best_f = result.fun
                        best_x = result.x.copy()
                        
                except Exception:
                    pass
        
        # Phase 4: Final polish if budget remains
        if best_x is not None and call_tracker[0] < budget - 5:
            if time.time() < deadline:
                try:
                    result = minimize(
                        wrapped_func,
                        best_x,
                        method='L-BFGS-B',
                        bounds=[(-5, 5)] * dim,
                        options={'maxiter': 50, 'maxls': 10}
                    )
                    
                    if result.fun < best_f:
                        best_f = result.fun
                        best_x = result.x.copy()
                        
                except Exception:
                    pass
        
        return best_x, best_f
# EVOLVE-BLOCK-END
