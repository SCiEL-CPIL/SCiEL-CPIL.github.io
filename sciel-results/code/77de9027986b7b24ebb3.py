import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        
        # 5 distinct trajectories with different step sizes
        n_traj = 5
        step_sizes = [0.1, 0.5, 1.0, 2.0, 4.0]
        
        # Initialize trajectories
        x = [rng.uniform(-5, 5, dim) for _ in range(n_traj)]
        sigma = step_sizes.copy()
        best_f_traj = [np.inf] * n_traj
        recent_hist = [[] for _ in range(n_traj)] # List of (call_idx, f_val)
        calls_since_global_improve = 0
        
        global_best_x = None
        global_best_f = np.inf
        calls_made = 0
        
        # Initial evaluations
        for i in range(n_traj):
            if calls_made >= budget:
                break
            f_val = func(x[i])
            calls_made += 1
            best_f_traj[i] = f_val
            recent_hist[i].append((calls_made, f_val))
            
            if f_val < global_best_f:
                global_best_f = f_val
                global_best_x = x[i].copy()
                calls_since_global_improve = 0
            else:
                calls_since_global_improve += 1
                
            if time.time() > deadline:
                return global_best_x, global_best_f

        # Main loop
        while calls_made < budget:
            if time.time() > deadline:
                break
            
            # 1. Determine which trajectory to update
            # Calculate scores based on recent improvement (last 20 calls in each traj's history)
            scores = []
            for i in range(n_traj):
                if len(recent_hist[i]) == 0:
                    scores.append(0.0)
                    continue
                hist = recent_hist[i][-20:]
                if len(hist) >= 2:
                    f_start = hist[0][1]
                    f_end = hist[-1][1]
                    imp = f_start - f_end
                else:
                    imp = 0.0
                scores.append(imp)
            
            # Add noise to prevent deterministic cycling
            scores_noisy = [s + rng.normal(0, 0.1) for s in scores]
            
            # Selection: 80% best, 20% random
            if rng.random() < 0.8:
                idx = np.argmax(scores_noisy)
            else:
                idx = rng.integers(0, n_traj)
            
            # 2. Generate offspring for selected trajectory
            z = rng.standard_normal(dim)
            y = x[idx] + sigma[idx] * z
            y = np.clip(y, -5, 5)
            
            # 3. Evaluate
            fy = func(y)
            calls_made += 1
            
            # Update history
            recent_hist[idx].append((calls_made, fy))
            if len(recent_hist[idx]) > 20:
                recent_hist[idx] = recent_hist[idx][-20:]
                
            # 4. Update Global Best
            if fy < global_best_f:
                global_best_f = fy
                global_best_x = y.copy()
                calls_since_global_improve = 0
            else:
                calls_since_global_improve += 1
                
            # 5. Update Trajectory State
            if fy < best_f_traj[idx]:
                # Improvement
                best_f_traj[idx] = fy
                x[idx] = y.copy()
                sigma[idx] *= 1.05 # Expand
            else:
                # No improvement
                sigma[idx] *= 0.95 # Contract
                
            sigma[idx] = np.clip(sigma[idx], 0.01, 10.0)
            
            # 6. Global Stagnation Handling
            # If no global improvement for 100 calls, restart a random trajectory from global best + noise
            if calls_since_global_improve >= 100:
                restart_idx = rng.integers(0, n_traj)
                perturbation = 2.0 * rng.standard_normal(dim)
                x[restart_idx] = np.clip(global_best_x + perturbation, -5, 5)
                sigma[restart_idx] = 1.0 # Reset to medium scale
                best_f_traj[restart_idx] = np.inf
                recent_hist[restart_idx] = []
                calls_since_global_improve = 0

        return global_best_x, global_best_f
