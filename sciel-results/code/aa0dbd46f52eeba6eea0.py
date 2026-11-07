import numpy as np
import scipy.optimize
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.start_time = time.time()
        # 330s limit, use 310s for safety margin
        self.deadline = self.start_time + 310

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        eval_count = 0
        best_x = None
        best_f = np.inf
        
        def counted_func(x):
            nonlocal eval_count
            if time.time() > self.deadline:
                raise TimeoutError
            if eval_count >= budget:
                raise RuntimeError("Budget exceeded")
            eval_count += 1
            return func(x)

        # Phase 1: CMA-ES for global/geometric adaptation
        # We use a simplified CMA-ES for the first 60% of the budget
        n_cma = int(budget * 0.6)
        
        # CMA-ES Parameters
        lam = max(4, int(4 + 3 * np.log(dim)))
        mu = lam // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mu_eff = 1 / np.sum(w ** 2)
        
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + dim / mu_eff) / (dim + 4 + 2 * mu_eff)
        c_1 = 2 / (dim ** 2 + 18)
        c_mu = min(1 - c_1, 2 * ((mu_eff + 0.5) / (dim + 2) ** 2 - 1))
        damps = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        
        # Initial State
        x_mean = np.zeros(dim)
        sigma = 2.0
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
        
        # Precompute B and D
        eigvals, eigvecs = np.linalg.eigh(C)
        D = np.sqrt(np.clip(eigvals, 1e-12, None))
        B = eigvecs
        
        # Initial evaluation
        try:
            best_f = counted_func(x_mean)
            best_x = x_mean.copy()
        except Exception:
            return np.zeros(dim), np.inf

        gen = 0
        while eval_count + lam <= n_cma and time.time() < self.deadline:
            # Generate offspring
            off = []
            off_f = []
            for i in range(lam):
                z = rng.standard_normal(dim)
                y = x_mean + sigma * (B @ (D * z))
                y = np.clip(y, -5, 5)
                off.append(y)
                try:
                    fy = counted_func(y)
                    off_f.append(fy)
                    if fy < best_f:
                        best_f = fy
                        best_x = y.copy()
                except Exception:
                    break
            else:
                # All offspring evaluated successfully
                off_f = np.array(off_f)
                idx = np.argsort(off_f)
                off = [off[i] for i in idx]
                
                # Recombine
                x_new = np.zeros(dim)
                for i in range(mu):
                    x_new += w[i] * off[i]
                
                # Update paths
                x_old = x_mean.copy()
                x_mean = x_new
                
                pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) / sigma * (x_mean - x_old)
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (B @ (D * (x_mean - x_old)))
                
                # Adapt sigma
                delta_sig = 1 if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (gen + 1))) < (1.4 + 2 / (dim + 1)) * chi_n else 0
                sigma *= np.exp((c_sigma / damps) * (np.linalg.norm(ps) / chi_n - 1))
                sigma = np.clip(sigma, 1e-12, 10)
                
                # Adapt covariance (Rank-1 update only for simplicity and speed)
                C = (1 - c_1) * C + c_1 * (np.outer(pc, pc) + (1 - delta_sig) * c_c * (2 - c_c) * np.outer(pc, pc))
                
                # Ensure positive definiteness and recompute B, D
                C = (C + C.T) / 2 # Symmetrize
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-12, None)
                D = np.sqrt(eigvals)
                B = eigvecs
                
                gen += 1
            # If exception occurred in loop, break out

        # Phase 2: Local Refinement with L-BFGS-B
        remaining_budget = budget - eval_count
        if remaining_budget > 10 and best_x is not None:
            bounds = [(-5.0, 5.0)] * self.dim
            try:
                res = scipy.optimize.minimize(
                    counted_func, 
                    best_x, 
                    method='L-BFGS-B', 
                    bounds=bounds,
                    options={'maxfun': remaining_budget, 'maxiter': 100}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except Exception:
                pass

        return best_x, best_f
