import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        timeout = 290.0
        d = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        lam = 4 + int(3 * np.log(d))
        mu = lam // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / np.sum(w)
        mueff = 1.0 / np.sum(w**2)
        
        cc = (4 + mueff / d) / (d + 4 + 2 * mueff / d)
        cs = (mueff + 2) / (d + mueff + 5)
        c1 = 2 / ((d + 1.3)**2 + mueff)
        cmu = min(1 - c1, 2 * ((mueff - 2 + 1 / mueff) / ((d + 2)**2 + mueff)))
        
        # Phase 0: Chebyshev Grid Initialization
        n_evals = 0
        if budget > 300:
            grid_coords = np.array([-4.0, 0.0, 4.0])
            grid_points = np.array(np.meshgrid(*([grid_coords]*d))).T.reshape(-1, d)
            grid_vals = np.array([func(p) for p in grid_points])
            n_evals = len(grid_points)
            best_idx = np.argmin(grid_vals)
            x_mean = grid_points[best_idx].copy()
            best_f = grid_vals[best_idx]
            best_x = x_mean.copy()
        else:
            x_mean = np.zeros(d)
            best_f = func(x_mean)
            best_x = x_mean.copy()
            n_evals = 1

        sigma = 1.0
        C = np.eye(d)
        B = np.eye(d)
        D = np.ones(d)
        pc = np.zeros(d)
        ps = np.zeros(d)
        
        it = 0
        
        # Phase 1: CMA-ES
        # Stop at 70% of budget to leave room for refinement
        cma_limit = 0.7 * budget
        while n_evals + lam <= cma_limit and time.time() - start_time < timeout:
            Z = np.random.randn(lam, d)
            Y = x_mean + sigma * (Z * D) @ B.T
            Y = np.clip(Y, -5, 5)
            
            F = np.array([func(y) for y in Y])
            n_evals += lam
            
            min_idx = np.argmin(F)
            if F[min_idx] < best_f:
                best_f = F[min_idx]
                best_x = Y[min_idx].copy()
                
            sorted_idx = np.argsort(F)
            Y_sorted = Y[sorted_idx]
            
            y_old = x_mean.copy()
            x_mean = np.dot(w, Y_sorted[:mu])
            
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) / sigma * (B.T @ (D * (x_mean - y_old)))
            
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2 * (it + 1)))) < (1 + 2 / (d + 1)) * cc
            it += 1
            
            if hsig:
                pc = (1 - cc) * pc + cc / sigma * (B.T @ (D * (x_mean - y_old)))
            
            delta_hebbian = c1 * (np.outer(ps, ps) + (1 - hsig) * (cc * (2 - cc) * np.outer(pc, pc)))
            
            for i in range(mu):
                delta_hebbian += (cmu * w[i]) * np.outer(Y_sorted[i] - y_old, Y_sorted[i] - y_old) / (sigma * sigma)
                
            C = (1 - c1 - cmu) * C + delta_hebbian
            C = (C + C.T) / 2
            
            D2, B = np.linalg.eigh(C)
            D = np.sqrt(np.clip(D2, 1e-20, None))
            
            chiN = np.sqrt(d) * (1 - 1/(4*d) + 1/(21*d**2))
            sigma *= np.exp(cs / chiN * (np.linalg.norm(ps) / chiN - 1))
            sigma = np.clip(sigma, 1e-6, 10.0)
            
            if sigma < 1e-5:
                break

        # Phase 2: Adaptive Coordinate Descent (Bisection-like)
        if n_evals < budget and time.time() - start_time < timeout:
            x_curr = best_x.copy()
            f_curr = best_f
            step = 1.0
            while n_evals < budget and time.time() - start_time < timeout:
                improved = False
                for i in range(d):
                    for direction in [1, -1]:
                        x_trial = x_curr.copy()
                        x_trial[i] += direction * step
                        x_trial = np.clip(x_trial, -5, 5)
                        f_trial = func(x_trial)
                        n_evals += 1
                        if f_trial < f_curr:
                            x_curr = x_trial
                            f_curr = f_trial
                            improved = True
                    if n_evals >= budget or time.time() - start_time > timeout:
                        break
                if not improved:
                    step *= 0.5
                    if step < 1e-4:
                        break
                if n_evals >= budget or time.time() - start_time > timeout:
                    break
            
            if f_curr < best_f:
                best_f = f_curr
                best_x = x_curr

        if best_x is None:
            best_x = np.zeros(d)
            best_f = func(best_x)
            
        return best_x, best_f
