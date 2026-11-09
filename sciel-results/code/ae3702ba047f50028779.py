"""Model-and-optimize approach for BBOB f4 (Büche-Rastrigin).
Uses CMA-ES with covariance adaptation for rotated multimodal landscape.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer with multi-restart for multimodal BBOB f4."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 290.0
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls = 0

        def evaluate(x):
            nonlocal calls
            if calls >= self.budget:
                return None
            x = np.clip(x, -5.0, 5.0)
            fx = float(func(x))
            calls += 1
            return fx

        dim = self.dim
        budget = self.budget

        # Phase 1: Broad exploration
        num_explore = min(10, budget // 10)
        for i in range(num_explore):
            if calls >= budget or time.time() > deadline:
                break
            if i == 0:
                x = np.zeros(dim)
            else:
                x = rng.uniform(-4.5, 4.5, dim)
            fx = evaluate(x)
            if fx is None:
                break
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 2: CMA-ES with restarts
        # CMA-ES parameters
        def cma_es(x0, sigma0, max_iter, time_limit):
            nonlocal best_f, best_x, calls
            
            n = dim
            # Population size
            lam = max(4, int(4 + 3 * np.log(n)))
            
            # Learning rates
            mu = lam // 2
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w = w / w.sum()
            mu_eff = 1.0 / (w @ w)
            cc = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
            cs = (mu_eff + 2) / (n + mu_eff + 5)
            c1 = 2 / ((n + 1.3) ** 2 + mu_eff)
            cmu = min(1 - c1, 2 * (mu_eff - 2 + 2 / mu_eff) / ((n + 2) ** 2 + mu_eff))
            
            # State
            xmean = x0.copy()
            sigma = sigma0
            C = np.eye(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
            
            best_local_f = np.inf
            best_local_x = None
            stall_count = 0
            max_stall = 30
            
            for it in range(max_iter):
                if calls >= budget or time.time() > deadline:
                    break
                
                # Generate population
                B = np.linalg.cholesky(C)
                z = rng.standard_normal((lam, n))
                y = z @ B.T
                pop = np.array([xmean + sigma * yi for yi in y])
                
                # Evaluate population
                evals = []
                for xi in pop:
                    xi_clipped = np.clip(xi, -5.0, 5.0)
                    if calls >= budget:
                        break
                    fx = float(func(xi_clipped))
                    calls += 1
                    evals.append((fx, xi_clipped))
                
                if len(evals) == 0:
                    break
                
                # Sort by fitness
                evals.sort(key=lambda t: t[0])
                sorted_f = np.array([e[0] for e in evals[:mu]])
                sorted_y = np.array([e[1] for e in evals[:mu]])
                
                # Update best
                if sorted_f[0] < best_local_f:
                    best_local_f = sorted_f[0]
                    best_local_x = sorted_y[0].copy()
                    stall_count = 0
                    if sorted_f[0] < best_f:
                        best_f = sorted_f[0]
                        best_x = sorted_y[0].copy()
                else:
                    stall_count += 1
                
                if stall_count >= max_stall:
                    break
                
                # Update mean
                xmean_new = xmean + sigma * (w @ (sorted_y - xmean / sigma))
                # Actually, we need to work in y-space
                ymean = w @ sorted_y
                xmean = xmean + sigma * ymean
                
                # Update covariance
                ydiff = (sorted_y - ymean[np.newaxis, :]) / sigma
                delta_s = (ps - pc) / (sigma * np.sqrt(1 - (1 - cc) ** (2 * (it + 1))))
                C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (1 - cc) * (np.eye(n) - C)) + \
                    cmu * (ydiff.T @ np.diag(w) @ ydiff)
                
                # Update evolution paths
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) / sigma * ymean
                pc = (1 - cc) * pc + cc * ymean
                
                # Update sigma
                sigma *= np.exp(cs / chi_n * (np.linalg.norm(ps) / chi_n - 1))
                
                # Clamp sigma
                sigma = np.clip(sigma, 1e-8, 10.0)
                
                # Ensure C is positive definite
                C = 0.5 * (C + C.T)
                try:
                    np.linalg.cholesky(C)
                except:
                    C += 1e-10 * np.eye(n)
            
            return best_local_x, best_local_f

        # Run CMA-ES with multiple restarts
        max_restarts = 15
        for restart in range(max_restarts):
            if calls >= budget or time.time() > deadline:
                break
            
            if restart == 0:
                x0 = best_x.copy() if best_x is not None else np.zeros(dim)
                sigma0 = 1.0
            elif restart % 3 == 0:
                x0 = rng.uniform(-4.5, 4.5, dim)
                sigma0 = 2.0
            elif restart % 3 == 1:
                x0 = best_x.copy() + rng.normal(0, 1.5, dim)
                x0 = np.clip(x0, -5, 5)
                sigma0 = 0.8
            else:
                x0 = best_x.copy() + rng.normal(0, 3.0, dim)
                x0 = np.clip(x0, -5, 5)
                sigma0 = 1.5
            
            # Budget per CMA-ES run
            remaining = budget - calls
            max_iter = max(5, remaining // (max(4, int(4 + 3 * np.log(dim))) + 1))
            
            local_x, local_f = cma_es(x0, sigma0, max_iter, deadline)
            
            if local_x is not None and local_f < best_f:
                best_f = local_f
                best_x = local_x.copy()

        # Phase 3: Fine-tuning
        if best_x is not None and calls < budget:
            sigma_fine = 0.05
            for _ in range(min(20, budget - calls)):
                if time.time() > deadline:
                    break
                y = best_x + sigma_fine * rng.standard_normal(dim)
                y = np.clip(y, -5.0, 5.0)
                fy = evaluate(y)
                if fy is None:
                    break
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
