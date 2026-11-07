"""Rewritten optimizer for BBOB f4 (Büche-Rastrigin) with CMA-ES and multi-start."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint


class Optimizer:
    """CMA-ES based optimizer with multi-start for BBOB f4."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _cmaes(self, func, x0, sigma0, budget, rng, deadline, calls_counter):
        """Run CMA-ES from x0 with initial sigma0, using up to budget evaluations.
        Returns (best_x, best_f, calls_used)."""
        dim = self.dim
        n = dim
        
        # CMA-ES parameters
        # Default parameters from Hansen
        cc = (4 + n / (n + 4)) / ((n + 4) + 4 + n / (n + 4))
        cs = (n + 1.3) / (n + 3 + 2 * n / (n + 4))
        cmu = min(1 - 2 / (n + 2), 0.5 + 2 / ((n + 2) ** 2 - 1))
        dsigma = 1 + 2 * max(0, sqrt(n) - 1) / (n + 3)  # not used in standard
        # Actually let's use simpler CMA-ES implementation
        
        # Standard CMA-ES
        pop_size = 4 + int(3 * np.log(n))
        # For n=5, pop_size = 4 + int(3*1.609) = 4 + 4 = 8
        
        sigma = sigma0
        mean = np.array(x0, dtype=float)
        C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        cs = (n + 1.3) / (n + 3 + 2 * n / (n + 4))
        cc = (4 + n / (n + 4)) / ((n + 4) + 4 + n / (n + 4))
        cmu = min(1 - 2 / (n + 2), 0.5 + 2 / ((n + 2) ** 2 - 1))
        hsig = 1.0  # will be computed
        
        # Cholesky of C
        try:
            B = np.linalg.cholesky(C)
            D = np.diag(B)
            D_inv = 1.0 / D
            B_inv = np.linalg.inv(B)
        except:
            B = np.eye(n)
            D = np.ones(n)
            D_inv = np.ones(n)
            B_inv = np.eye(n)
        
        best_x = mean.copy()
        best_f = func(mean)
        calls_counter[0] += 1
        
        # Track best
        for gen in range(1, budget // pop_size + 1):
            if time.time() > deadline:
                break
            if calls_counter[0] >= self.budget:
                break
            
            # Generate population
            ys = []
            for i in range(pop_size):
                if calls_counter[0] >= self.budget:
                    break
                if time.time() > deadline:
                    break
                z = rng.standard_normal(n)
                y = mean + sigma * (B @ z)
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls_counter[0] += 1
                ys.append((fy, y))
            
            if len(ys) == 0:
                break
            
            # Sort by fitness
            ys.sort(key=lambda t: t[0])
            
            # Update best
            if ys[0][0] < best_f:
                best_f = ys[0][0]
                best_x = ys[0][1].copy()
            
            # Update mean
            mu = pop_size // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1.0 / np.sum(weights ** 2)
            
            new_mean = np.zeros(n)
            for i in range(mu):
                new_mean += weights[i] * ys[i][1]
            
            # Update covariance
            delta_mean = new_mean - mean
            C_update = C
            for i in range(mu):
                yw = (ys[i][1] - mean) / sigma
                C_update += cmu / mueff * np.outer(yw, yw)
            C = ((1 - cmu / mueff) * C + cmu / mueff * C_update)
            
            # Ensure C is symmetric positive definite
            C = (C + C.T) / 2
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-10, None)
                C = (eigvecs * eigvals) @ eigvecs.T
                C = (C + C.T) / 2
            except:
                pass
            
            # Update pc
            pc = (1 - cc) * pc + cc * delta_mean / sigma
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * C @ (delta_mean / sigma)
            
            # Update sigma
            sigma *= np.exp(cs / dsigma * (np.linalg.norm(ps) / np.sqrt(2 * n) - 1))
            sigma = np.clip(sigma, 1e-8, 10.0)
            
            mean = new_mean
            mean = np.clip(mean, -5, 5)
            
            # Update Cholesky
            try:
                B = np.linalg.cholesky(C)
                D = np.diag(B)
                D_inv = 1.0 / D
                B_inv = np.linalg.inv(B)
            except:
                B = np.eye(n)
                D = np.ones(n)
                D_inv = np.ones(n)
                B_inv = np.eye(n)
            
            # Stopping criteria
            if sigma < 1e-8:
                break
            if np.max(np.abs(delta_mean)) < 1e-10:
                break
        
        return best_x, best_f, calls_counter[0]

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        rng = np.random.default_rng(42)
        
        calls_counter = [0]
        best_x = None
        best_f = np.inf
        
        # Multi-start strategy:
        # 1. Evaluate several diverse starting points
        # 2. Run CMA-ES from the best few
        # 3. If budget remains, run from additional points
        
        n_starts = 10
        
        # Generate diverse starting points
        starts = []
        # Center
        starts.append(np.zeros(self.dim))
        # Random points
        for _ in range(n_starts - 1):
            starts.append(rng.uniform(-4.5, 4.5, self.dim))
        
        # Evaluate all starts
        candidates = []
        for x in starts:
            if calls_counter[0] >= self.budget:
                break
            if time.time() > deadline:
                break
            x = np.clip(x, -5, 5)
            fx = func(x)
            calls_counter[0] += 1
            candidates.append((fx, x.copy()))
        
        if not candidates:
            # Fallback
            x = rng.uniform(-5, 5, self.dim)
            fx = func(x)
            calls_counter[0] += 1
            return x, fx
        
        # Sort by function value
        candidates.sort(key=lambda t: t[0])
        
        # Run CMA-ES from top candidates
        n_cma_starts = min(3, len(candidates))
        
        for i in range(n_cma_starts):
            if calls_counter[0] >= self.budget:
                break
            if time.time() > deadline:
                break
            
            fx_init, x_init = candidates[i]
            
            # If this candidate is not better than current best, skip
            if fx_init >= best_f and best_x is not None:
                continue
            
            # Allocate budget for this CMA-ES run
            remaining = self.budget - calls_counter[0]
            # Use a fraction of remaining budget
            if i == 0:
                budget_for_cma = int(remaining * 0.5)
            elif i == 1:
                budget_for_cma = int(remaining * 0.3)
            else:
                budget_for_cma = int(remaining * 0.2)
            
            budget_for_cma = max(budget_for_cma, 10)
            
            x_loc, fx_loc, _ = self._cmaes(
                func, x_init.copy(), 1.0, budget_for_cma, rng, deadline, calls_counter
            )
            
            if fx_loc < best_f:
                best_f = fx_loc
                best_x = x_loc.copy()
        
        # If we still have budget, try another CMA-ES from the best point
        if best_x is not None and calls_counter[0] < self.budget:
            remaining = self.budget - calls_counter[0]
            if remaining >= 20:
                x_loc, fx_loc, _ = self._cmaes(
                    func, best_x.copy(), 0.5, remaining, rng, deadline, calls_counter
                )
                if fx_loc < best_f:
                    best_f = fx_loc
                    best_x = x_loc.copy()
        
        # Ensure best_x is set
        if best_x is None:
            best_x = candidates[0][1]
            best_f = candidates[0][0]
        
        return best_x, best_f
# EVOLVE-BLOCK-END
