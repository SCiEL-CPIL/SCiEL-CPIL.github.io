"""
Optimizer for BBOB f24 (Lunacek bi-Rastrigin) using tuned CMA-ES with warm start.
"""
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # 10s safety margin
        
        dim = self.dim
        budget = self.budget
        calls = 0
        best_x = None
        best_f = np.inf

        # Helper to update best
        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Phase 1: Warm Start - dense sampling to identify global basin
        # Lunacek bi-Rastrigin has two basins; we need to find the right one.
        n_warm = 25
        for _ in range(n_warm):
            if calls >= budget or time.time() > deadline:
                break
            x = np.random.uniform(-5, 5, size=dim)
            f = func(x)
            calls += 1
            update_best(x, f)
        
        # Additional structured sampling around the best point to refine basin
        if calls < budget and time.time() < deadline:
            for _ in range(15):
                if calls >= budget or time.time() > deadline:
                    break
                x = best_x + np.random.uniform(-2, 2, size=dim)
                x = np.clip(x, -5, 5)
                f = func(x)
                calls += 1
                update_best(x, f)

        # Initialize CMA-ES from the best warm start point
        x0 = best_x.copy()
        
        # CMA-ES Parameters
        # For dim=5, lambda=14 for better exploration in bi-modal landscape
        lambda_ = 14
        mu = lambda_ // 2
        
        # Weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / np.sum(w)
        mu_eff = 1.0 / np.sum(w**2)
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        # State
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        
        # Initial sigma: Start large to span the basin and navigate Rastrigin ridges
        sigma = 2.5
        
        gen = 0
        stagnation_count = 0
        prev_best_f = best_f

        def sqrtm(C):
            try:
                return np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                vals, vecs = np.linalg.eigh(C)
                vals = np.maximum(vals, 1e-10)
                return vecs @ np.diag(np.sqrt(vals)) @ vecs.T

        B = sqrtm(C)

        while calls < budget:
            if time.time() > deadline:
                break
                
            # Generate population
            ys = []
            fs = []
            
            for i in range(lambda_):
                if calls >= budget:
                    break
                if time.time() > deadline:
                    break
                
                z = np.random.standard_normal(dim)
                y = x0 + sigma * (B @ z)
                # Clip to bounds [-5, 5]
                y = np.clip(y, -5, 5)
                
                fy = func(y)
                calls += 1
                
                ys.append(y)
                fs.append(fy)
                update_best(y, fy)
            
            if len(fs) == 0:
                break
                
            fs = np.array(fs)
            order = np.argsort(fs)
            ys = [ys[i] for i in order]
            fs = fs[order]
            
            # Update best if any new point was better (already done in loop, but ensure consistency)
            # Actually update_best is called in loop, so best_x is updated.
            
            # Check stagnation
            if best_f < prev_best_f - 1e-8:
                stagnation_count = 0
            else:
                stagnation_count += 1
            
            prev_best_f = best_f

            # If stagnant for a long time, restart covariance adaptation
            # but keep the mean. This helps escape local minima in Rastrigin.
            if stagnation_count > 10:
                C = np.eye(dim)
                pc = np.zeros(dim)
                ps = np.zeros(dim)
                B = sqrtm(C)
                # Increase sigma to explore more broadly
                sigma *= 1.3
                stagnation_count = 0

            # Sort and select top mu
            ys_sel = ys[:mu]
            fs_sel = fs[:mu]
            
            # Update mean
            x_new = np.sum(np.array(ys_sel) * w[:, np.newaxis], axis=0)
            
            # Update step size control
            # C^{-1/2}
            vals, vecs = np.linalg.eigh(C)
            vals_inv_sqrt = 1.0 / np.sqrt(np.maximum(vals, 1e-10))
            C_inv_sqrt = (vecs * vals_inv_sqrt) @ vecs.T
            
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C_inv_sqrt @ (x_new - x0) / sigma
            
            h_sig = 1.0
            # Standard h_sig check is complex, simplified version:
            # If norm of ps is within expected range, h_sig=1 else 0
            # This check helps avoid sigma shrinking too fast when ps is random noise
            if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**(2 * (gen + 1))) < (1.4 + 2 / (dim + 1)) * np.sqrt(dim):
                sigma *= np.exp(c_sigma / dim * (np.linalg.norm(ps) / np.sqrt(dim) - 1))
            
            # Update covariance matrix
            pc = (1 - c_c) * pc + c_c * (x_new - x0) / sigma
            yk = x_new - x0
            rank1 = np.outer(yk, yk)
            rankmu = np.sum(np.array([np.outer(y - x0, y - x0) for y in ys_sel]) * w[:, np.newaxis, np.newaxis], axis=0)
            
            C = (1 + c_1 * (c_mu * mu_eff / dim - 1) - c_1 - c_mu) * C \
                + c_1 * (np.outer(pc, pc) + (1 - (1 - c_c)**2) * C) \
                + c_mu * rankmu
            
            # Symmetrize
            C = (C + C.T) / 2
            
            x0 = x_new
            gen += 1
            
            # Update B
            B = sqrtm(C)

        # Final polish phase: use remaining budget for local refinement
        if calls < budget and time.time() < deadline and best_x is not None:
            # Simple coordinate descent / local search to polish
            for _ in range(50):
                if calls >= budget or time.time() > deadline:
                    break
                
                # Try small perturbations in each dimension
                improved = False
                for d in range(dim):
                    if calls >= budget or time.time() > deadline:
                        break
                    
                    # Try shrinking step
                    for step in [0.1, 0.05, 0.02, 0.01, 0.005]:
                        if calls >= budget or time.time() > deadline:
                            break
                        
                        x_try = best_x.copy()
                        x_try[d] += step
                        x_try = np.clip(x_try, -5, 5)
                        f_try = func(x_try)
                        calls += 1
                        
                        if f_try < best_f - 1e-10:
                            best_x = x_try
                            best_f = f_try
                            improved = True
                            break
                        
                        x_try = best_x.copy()
                        x_try[d] -= step
                        x_try = np.clip(x_try, -5, 5)
                        f_try = func(x_try)
                        calls += 1
                        
                        if f_try < best_f - 1e-10:
                            best_x = x_try
                            best_f = f_try
                            improved = True
                            break
                    if improved:
                        break
                
                if not improved:
                    break
        
        return best_x, best_f
