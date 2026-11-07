"""Optimizer using Multi-start CMA-ES for BBOB f13 (Sharp Ridge).

Strategy:
1. Run CMA-ES with multiple restarts to handle the sharp ridge structure.
2. CMA-ES adapts its covariance matrix to align with the ridge direction.
3. Use a budget-splitting scheme: allocate evaluations across restarts.
4. Each restart uses a smaller population and adapts quickly.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """
    Multi-start CMA-ES optimizer for Sharp Ridge functions.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t0 = time.time()
        deadline = t0 + 290.0

        def elapsed_ok():
            return time.time() < deadline

        rng = np.random.default_rng(42)

        def clip(v):
            return np.clip(v, -5.0, 5.0)

        best_x = None
        best_f = np.inf
        evaluated = 0

        def eval_point(x):
            nonlocal evaluated, best_x, best_f
            x = np.asarray(x, dtype=float)
            x = clip(x)
            f = func(x)
            evaluated += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # CMA-ES implementation
        def cma_es(func, x0, budget, dim, sigma0=1.0, max_time=None):
            nonlocal evaluated, best_x, best_f
            
            if max_time is None:
                max_time = deadline
            
            def time_ok():
                return time.time() < max_time

            # CMA-ES parameters
            N = max(4, int(4 + 3 * np.log(dim)))  # population size
            # chiN = expected norm of standard normal
            chiN = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
            
            # Step size control
            c_sigma = (dim + 1.3) / (dim + 5 + N)
            d_sigma = 1 + 2*max(0, np.sqrt((N+1)/(dim+1)) - 1) + c_sigma
            hs = 1 + 2*max(0, np.sqrt((N+1)/(dim+1)) - 1)
            
            # Position update
            c_c = (4 + dim/N) / (dim + 4 + 2*N/dim)
            c_1 = 2 / ((dim + 1.3)**2 + N)
            c_mu = min(1 - c_1, 2 * ((mu_eff + 0.5) / ((dim + 2) * (N - mu_eff + 0.5)) - c_1))
            
            mu = N // 2  # number of recombination weights
            w = np.log(np.arange(1, mu + 1)) - np.log(mu)
            w /= w.sum()
            mu_eff = 1 / (w ** 2).sum()
            
            # State
            x_m = np.array(x0, dtype=float)
            sigma = sigma0
            C = np.eye(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            
            local_best_x = x_m.copy()
            local_best_f = np.inf
            
            total_evals = 0
            gen = 0
            
            while total_evals < budget and time_ok():
                # Generate population
                B = np.linalg.cholesky(C)
                ys = []
                xs = []
                for i in range(N):
                    z = rng.standard_normal(dim)
                    y = sigma * B @ z
                    x = x_m + y
                    x = clip(x)
                    ys.append(y)
                    xs.append(x)
                
                # Evaluate
                fs = np.array([eval_point(x) for x in xs])
                total_evals += N
                
                # Update best
                idx_min = np.argmin(fs)
                if fs[idx_min] < local_best_f:
                    local_best_f = fs[idx_min]
                    local_best_x = xs[idx_min].copy()
                
                # Sort
                idx_sort = np.argsort(fs)
                ys = np.array(ys)
                xs = np.array(xs)
                fs = fs[idx_sort]
                ys = ys[idx_sort]
                
                # Update mean
                y_w = (w[:, None] * ys[:mu]).sum(axis=0)
                x_m_new = x_m + y_w / sigma * sigma  # = x_m + y_w
                x_m_new = clip(x_m_new)
                
                # Update step size
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / np.linalg.norm(B) * (B @ y_w / sigma)
                if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**(2 * (gen + 1))) < (1.4 + 2 / (dim + 1)) * chiN:
                    sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chiN - 1))
                
                # Update covariance
                delta_c = (1 - c_c) if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**(2 * (gen + 1))) < (1.4 + 2 / (dim + 1)) * chiN else 0
                pc = (1 - c_c) * pc + delta_c * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_w / sigma
                C = (1 + c_1 + c_mu - c_1) * C - c_1 * np.outer(pc, pc)
                C += c_1 * np.outer(pc, pc)  # correction
                C = (1 - c_1 - c_mu) * C + c_1 * np.outer(pc, pc)
                for i in range(mu):
                    C += c_mu * w[i] * np.outer(ys[i], ys[i])
                
                # Ensure positive semi-definite
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-20)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                C = (C + C.T) / 2
                
                x_m = x_m_new
                gen += 1
                
                # Check if we should stop
                if sigma < 1e-10:
                    break
            
            return local_best_x, local_best_f, total_evals

        # Multi-start CMA-ES
        n_restarts = 3
        # Allocate budget: give more to later restarts as they start from better points
        # But keep first restart substantial to find the ridge
        remaining = budget
        restarts_budget = []
        for i in range(n_restarts):
            if i < n_restarts - 1:
                # First two restarts get smaller budgets
                b = int(remaining * 0.25)
            else:
                b = remaining
            restarts_budget.append(b)
            remaining -= b
        
        # Ensure we don't exceed total budget
        total_alloc = sum(restarts_budget)
        if total_alloc > budget:
            scale = budget / total_alloc
            restarts_budget = [int(b * scale) for b in restarts_budget]
        
        x_start = None
        for r in range(n_restarts):
            if not elapsed_ok() or evaluated >= budget:
                break
            
            # Determine starting point
            if x_start is None:
                # First start: random
                x0 = rng.uniform(-4.0, 4.0, size=dim)
            else:
                # Subsequent starts: near the best found
                x0 = best_x + rng.normal(0, 0.5, size=dim)
                x0 = clip(x0)
            
            # Run CMA-ES for this restart
            alloc = restarts_budget[r]
            if alloc < 10:
                alloc = remaining if remaining > 10 else remaining
            
            # Time limit for this restart: leave some time for others
            time_left = deadline - time.time()
            per_restart_time = time_left / (n_restarts - r)
            
            x_best_r, f_best_r, evals_used = cma_es(
                func, x0, alloc, dim, 
                sigma0=1.0 if r == 0 else 0.5,
                max_time=time.time() + per_restart_time
            )
            
            x_start = x_best_r
        
        return best_x, best_f
# EVOLVE-BLOCK-END
