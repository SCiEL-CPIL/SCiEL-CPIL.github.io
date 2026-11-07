import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.x_min = -5.0
        self.x_max = 5.0

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s kill
        rng = np.random.default_rng(42)
        
        # Phase 1: GP-guided exploration for first ~30 evaluations
        n_init = 10
        X_init = rng.uniform(self.x_min, self.x_max, size=(n_init, self.dim))
        Y_init = np.array([func(x) for x in X_init])
        
        X = X_init.copy()
        Y = Y_init.copy()
        best_x = X_init[np.argmin(Y_init)].copy()
        best_y = Y_init.min()
        calls_made = n_init
        
        # GP parameters
        l = 0.8
        sigma_f_sq = 1.0
        sigma_n_sq = 1e-6
        
        def compute_kernel(X1, X2):
            sq_dist = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * X1 @ X2.T
            sq_dist = np.maximum(sq_dist, 0)
            return np.exp(-sq_dist / (2 * l**2))
        
        # GP-guided phase: ~30 evaluations total
        gp_phase_end = 30
        while calls_made < gp_phase_end:
            if time.time() > deadline:
                break
            
            # Fit GP
            K = compute_kernel(X, X)
            K += (sigma_n_sq + 1e-8) * np.eye(len(X))
            
            y_mean = Y.mean()
            y_std = Y.std()
            if y_std < 1e-8:
                y_std = 1.0
            Y_std = (Y - y_mean) / y_std
            
            try:
                L = np.linalg.cholesky(K)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y_std))
            except np.linalg.LinAlgError:
                alpha = np.linalg.inv(K) @ Y_std
            
            # Generate candidates
            n_candidates = 500
            X_cand = rng.uniform(self.x_min, self.x_max, size=(n_candidates, self.dim))
            
            # Add local perturbations around best
            n_local = 100
            sigma_local = 0.5
            X_local = best_x + sigma_local * rng.standard_normal((n_local, self.dim))
            X_local = np.clip(X_local, self.x_min, self.x_max)
            X_cand = np.vstack([X_cand, X_local])
            
            # Predict
            K_s = compute_kernel(X_cand, X)
            mu = (K_s @ alpha) * y_std + y_mean
            v = np.linalg.solve(L, K_s.T)
            var = sigma_f_sq - np.sum(v**2, axis=0)
            var = np.maximum(var, 1e-10)
            std = np.sqrt(var)
            
            # Adaptive beta
            progress = calls_made / gp_phase_end
            beta = 3.0 + (0.5 - 3.0) * progress
            acq = mu - beta * std
            
            best_idx = np.argmin(acq)
            x_new = X_cand[best_idx]
            y_new = func(x_new)
            calls_made += 1
            
            X = np.vstack([X, x_new])
            Y = np.append(Y, y_new)
            
            if y_new < best_y:
                best_y = y_new
                best_x = x_new.copy()
        
        # Phase 2: Multi-restart (1+1)-ES with adaptive sigma
        # Use remaining budget for local search with restarts
        remaining = self.budget - calls_made
        
        # Track global best
        global_best_x = best_x.copy()
        global_best_y = best_y
        
        # (1+1)-ES with restarts
        sigma = 1.5
        no_improve = 0
        max_no_improve = 40  # Restart if no improvement for 40 steps
        restart_count = 0
        max_restarts = 5
        
        while calls_made < self.budget:
            if time.time() > deadline:
                break
            
            # Propose new point
            y = np.clip(best_x + sigma * rng.standard_normal(self.dim), self.x_min, self.x_max)
            fy = func(y)
            calls_made += 1
            
            if fy < best_y:
                best_x = y
                best_y = fy
                sigma *= 1.2  # Slightly increase on success
                no_improve = 0
                
                if fy < global_best_y:
                    global_best_y = fy
                    global_best_x = y.copy()
            else:
                sigma *= 0.85
                no_improve += 1
            
            # Restart logic: if stuck, perturb from global best
            if no_improve >= max_no_improve and restart_count < max_restarts:
                # Perturb global best with larger step
                sigma = 2.0
                perturbation = 1.5 * rng.standard_normal(self.dim)
                best_x = np.clip(global_best_x + perturbation, self.x_min, self.x_max)
                best_y = func(best_x)
                calls_made += 1
                no_improve = 0
                restart_count += 1
                
                if best_y < global_best_y:
                    global_best_y = best_y
                    global_best_x = best_x.copy()
        
        return global_best_x, global_best_y
