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

        # Phase 1: Warm Start
        # Evaluate a small random population to find a good starting point.
        # This helps locate the correct global basin of the "bi" function.
        n_warm = 20
        for _ in range(n_warm):
            if calls >= budget or time.time() > deadline:
                break
            x = np.random.uniform(-5, 5, size=dim)
            f = func(x)
            calls += 1
            update_best(x, f)

        # Initialize CMA-ES from the best warm start point
        x0 = best_x.copy()
        
        # CMA-ES Parameters
        # For dim=5, lambda=15 for more exploration
        lambda_ = 15
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
        
        # Initial sigma: Start large to escape local minima and cross basin boundaries
        sigma = 2.0
        
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
            if stagnation_count > 15:
                C = np.eye(dim)
                pc = np.zeros(dim)
                ps = np.zeros(dim)
                B = sqrtm(C)
                # Slightly increase sigma to explore
                sigma *= 1.1
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

        # Local Polish Phase using Nelder-Mead
        if calls < budget and best_x is not None and time.time() < deadline:
            # Initialize simplex around best_x
            simplex = [best_x.copy()]
            step = 0.5
            for i in range(dim):
                p = best_x.copy()
                p[i] += step
                simplex.append(p)
            
            fs = []
            for p in simplex:
                if calls >= budget or time.time() > deadline:
                    break
                fp = func(p)
                calls += 1
                fs.append(fp)
                if fp < best_f:
                    best_f = fp
                    best_x = p.copy()
            
            # Nelder-Mead loop
            while calls < budget and time.time() < deadline:
                # Sort simplex by function value
                order = np.argsort(fs)
                simplex = [simplex[i] for i in order]
                fs = [fs[i] for i in order]
                
                # Calculate centroid of all but the worst point
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                alpha = 1.0
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                if calls >= budget or time.time() > deadline:
                    break
                fr = func(xr)
                calls += 1
                if fr < best_f:
                    best_f = fr
                    best_x = xr.copy()
                
                if fr < fs[0]:
                    # Expansion
                    gamma = 2.0
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5, 5)
                    if calls >= budget or time.time() > deadline:
                        break
                    fe = func(xe)
                    calls += 1
                    if fe < best_f:
                        best_f = fe
                        best_x = xe.copy()
                    if fe < fr:
                        simplex[-1] = xe
                        fs[-1] = fe
                    else:
                        simplex[-1] = xr
                        fs[-1] = fr
                elif fr < fs[-2]:
                    # Accept reflection
                    simplex[-1] = xr
                    fs[-1] = fr
                else:
                    # Contraction
                    rho = 0.5
                    xc = centroid + rho * (simplex[-1] - centroid)
                    xc = np.clip(xc, -5, 5)
                    if calls >= budget or time.time() > deadline:
                        break
                    fc = func(xc)
                    calls += 1
                    if fc < best_f:
                        best_f = fc
                        best_x = xc.copy()
                    if fc < fs[-1]:
                        simplex[-1] = xc
                        fs[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            if calls >= budget or time.time() > deadline:
                                break
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            fi = func(simplex[i])
                            calls += 1
                            fs[i] = fi
                            if fi < best_f:
                                best_f = fi
                                best_x = simplex[i].copy()

        return best_x, best_f
