import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        self._eval_count = 0
        dim = self.dim
        budget = self.budget
        
        def eval_func(x):
            if self._eval_count >= budget:
                return np.inf
            self._eval_count += 1
            return func(x)

        # CMA-ES Parameters
        # lambda: number of offspring
        lam = 4 + int(3 * np.log(dim))
        # mu: number of best points for recombination
        mu = lam // 2
        # weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        # mu_eff
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Initial parameters
        x0 = np.zeros(dim)
        sigma = 2.0 # Initial step size, box is [-5,5]
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        
        # Adaptation parameters
        cc = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        cs = (mu_eff + 2) / (dim + mu_eff + 5)
        c1 = 2 / ((dim + 1.3)**2 + mu_eff)
        cmu = min(1 - c1, 2 * (weights[1] - 0.5) / ((dim + 2)**2 + mu_eff))
        # Damping for sigma update
        damps = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + cs
        
        best_x = x0.copy()
        best_fx = eval_func(x0)
        
        # If we used 1 call, we have budget-1 left.
        
        while self._eval_count < budget:
            if time.time() > deadline:
                break
            
            # Sample points
            ys = np.random.multivariate_normal(np.zeros(dim), C, lam)
            # Ensure we don't have NaNs/Infs from multivariate_normal if C becomes singular
            # CMA-ES usually handles this, but let's be safe.
            ys = np.nan_to_num(ys, nan=0.0, posinf=0.0, neginf=0.0)
            
            xs = x0 + sigma * ys
            # Clip to box? The problem says func clips. We can let func clip, 
            # but for CMA-ES statistics, it's better to work in the unbounded space 
            # if possible, or just accept that the function clips. 
            # The prompt says "func(x) clips a finite length-5 vector to the box".
            # So we can pass unclipped xs.
            
            fs = np.array([eval_func(x) for x in xs])
            
            if self._eval_count >= budget:
                break
                
            # Sort by fitness
            idx = np.argsort(fs)
            xs_sorted = xs[idx]
            fs_sorted = fs[idx]
            
            # Update best
            if fs_sorted[0] < best_fx:
                best_fx = fs_sorted[0]
                best_x = xs_sorted[0].copy()
                
            # Update mean (recombination)
            x0_old = x0.copy()
            x0 = np.sum(weights[:, np.newaxis] * xs_sorted[:mu], axis=0)
            
            # Update path evolution strategies
            y_mean = (x0 - x0_old) / sigma
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * y_mean
            ps_norm = np.linalg.norm(ps)
            
            # Update covariance matrix
            # Rank-one update
            delta_sigma = (ps_norm / np.sqrt(2 * dim) - 1) * cs * damps
            C = (1 + c1 * delta_sigma * cs * (2 - cs)) * C
            
            # Rank-mu update
            if mu_eff > 1:
                # Calculate rank-mu matrix
                # v = sum w_i * (x_i - x_old) / sigma
                v = np.sum(weights[:mu, np.newaxis] * (xs_sorted[:mu] - x0_old) / sigma, axis=0)
                # Actually, the standard update is:
                # C = (1 - c1 - cmu) * C + c1 * (ps*ps.T + (delta_sigma != 1) * cs*(2-cs)*C) + cmu * sum(wi * vi * vi.T)
                # Wait, the term for ps is usually handled separately or included in the rank-1 part.
                # Let's use the standard formula from Hansen's paper.
                
                # Recalculate C update properly
                C = (1 - c1 - cmu) * C
                # Rank 1
                C += c1 * (np.outer(ps, ps) + (delta_sigma != 1) * cs * (2 - cs) * np.eye(dim))
                # Rank mu
                for i in range(mu):
                    C += cmu * weights[i] * np.outer(v, v) # This is wrong, v should be per i?
                    # No, v_i = (x_i - x_old)/sigma
                    # The formula is sum_{i=1}^mu w_i * (x_i - x_old)/sigma * ((x_i - x_old)/sigma).T
                    pass
                
                # Let's restart the C update logic to be correct.
                C = (1 - c1 - cmu) * C
                C += c1 * (np.outer(ps, ps) + (delta_sigma != 1) * cs * (2 - cs) * np.eye(dim))
                for i in range(mu):
                    v_i = (xs_sorted[i] - x0_old) / sigma
                    C += cmu * weights[i] * np.outer(v_i, v_i)
            else:
                C = (1 - c1) * C + c1 * (np.outer(ps, ps) + (delta_sigma != 1) * cs * (2 - cs) * np.eye(dim))
                
            # Ensure C is symmetric
            C = (C + C.T) / 2.0
            
            # Update sigma
            sigma *= np.exp(cs / damps * (ps_norm / np.sqrt(2 * dim) - 1))
            
            # Check for convergence or singular C
            if sigma < 1e-10:
                break
            if np.any(np.isnan(C)) or np.any(np.isinf(C)):
                break

        return best_x, best_fx
