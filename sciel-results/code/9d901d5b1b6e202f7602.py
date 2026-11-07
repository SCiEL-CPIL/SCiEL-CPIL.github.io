import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        budget = self.budget
        dim = self.dim
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls_used = 0

        def cma_es(x0, sigma0, max_calls):
            nonlocal best_x, best_f, calls_used
            
            n = dim
            # Population size
            lam = 4 + int(3 * np.log(n))
            mu = lam // 2
            # Weights
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mueff = 1 / np.sum(weights**2)
            
            # Strategy parameters
            cc = (4 + mueff/n) / (n + 4 + 2*mueff/n)
            cs = (mueff + 2) / (n + mueff + 5)
            c1 = 2 / ((n + 1.3)**2 + mueff)
            cmu = min(1 - c1, 2 * (mueff - 2 + 1/mueff) / ((n + 2)**2 + mueff))
            
            # State
            m = x0.copy()
            sigma = sigma0
            C = np.eye(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            chi_n = np.sqrt(n) * (1 - 1/(4*n) + 1/(20*n**2))
            
            for gen in range(max_calls // lam + 1):
                if time.time() > deadline:
                    return
                if calls_used >= budget:
                    return
                
                # Generate population
                y_points = []
                f_points = []
                
                for i in range(lam):
                    if calls_used >= budget:
                        return
                    if time.time() > deadline:
                        return
                        
                    z = rng.standard_normal(n)
                    y = m + sigma * z
                    # Clip to box
                    y = np.clip(y, -5, 5)
                    
                    fy = func(y)
                    calls_used += 1
                    
                    y_points.append(y)
                    f_points.append(fy)
                    
                    if fy < best_f:
                        best_f = fy
                        best_x = y.copy()
                
                # Sort by fitness
                order = np.argsort(f_points)
                y_sorted = np.array(y_points)[order]
                f_sorted = np.array(f_points)[order]
                
                # Update mean
                m_new = np.sum(weights[:, None] * y_sorted[:mu], axis=0)
                
                # Update covariance matrix
                y_diffs = (y_sorted[:mu] - m) / sigma
                yw = y_diffs.mean(axis=0)
                
                # Update path variables
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * yw
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2 * (gen + 1))) / chi_n < 1.4 + 2 / (n + 1))
                pc = (1 - cs) * pc + hsig * cs * mueff * yw
                
                # Update covariance matrix
                delta_hsig = (1 - hsig) * cs * (2 - cs)
                C = (1 + c1 * delta_hsig - c1 - cmu) * C
                C += c1 * (pc * pc.T + delta_hsig * chi_n**2 * np.eye(n))
                C += cmu * np.sum(weights[:, None, None] * np.outer(y_diffs, y_diffs), axis=0)
                
                m = m_new
                sigma *= np.exp(cs / (n + 1.3) * (np.linalg.norm(ps) / chi_n - 1))
                
                # Ensure symmetry and positive semi-definiteness
                C = (C + C.T) / 2
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T

        # Strategy:
        # 1. Run CMA-ES from origin.
        # 2. If budget remains, run CMA-ES from best found with larger sigma.
        # 3. If budget remains, run CMA-ES from random points.
        
        # Phase 1: Origin start
        x0 = np.zeros(dim)
        cma_es(x0, 2.0, int(budget * 0.5))
        
        # Phase 2: Restart from best with larger sigma if needed
        if calls_used < budget and best_x is not None:
            cma_es(best_x, 3.0, int(budget * 0.3))
            
        # Phase 3: Random starts
        remaining = budget - calls_used
        if remaining > 50:
            # Split remaining among 2 random starts
            calls_per_start = remaining // 2
            for _ in range(2):
                x0 = rng.uniform(-4, 4, dim)
                cma_es(x0, 2.0, calls_per_start)
                if calls_used >= budget:
                    break

        return best_x, best_f
