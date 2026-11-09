"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-restart CMA-ES with stall detection for Büche-Rastrigin."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 300s limit, 20s safety margin
        rng = np.random.default_rng(12345)
        
        best_x = None
        best_f = np.inf
        calls_made = 0
        budget = self.budget
        dim = self.dim

        # Multi-restart strategy: use budget in chunks, restart if stalled
        # Büche-Rastrigin has many local minima, so restarts help escape
        max_restarts = 5
        calls_per_restart = budget // max_restarts  # 100 calls per restart
        
        for restart in range(max_restarts):
            if calls_made >= budget:
                break
            if time.time() > deadline:
                break
                
            # Initialize: first restart at origin, subsequent near best or random
            if restart == 0:
                x = np.zeros(dim)
            elif best_x is not None and rng.random() < 0.5:
                # Perturb around current best
                x = np.clip(best_x + rng.standard_normal(dim) * 0.5, -5, 5)
            else:
                # Random restart
                x = rng.uniform(-5, 5, dim)
            
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f, best_x = fx, x.copy()
            
            # Simple CMA-ES-like adaptive step size
            sigma = 1.0
            stall_count = 0
            max_stall = 15
            
            while calls_made < budget:
                if time.time() > deadline:
                    break
                    
                # Generate candidate
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_made += 1
                
                if fy < fx:
                    # Accept improvement
                    x, fx = y, fy
                    sigma *= 1.1
                    stall_count = 0
                    if fx < best_f:
                        best_f, best_x = fx, x.copy()
                else:
                    # Reject, adapt step size
                    sigma *= 0.9
                    stall_count += 1
                    
                # Restart within this chunk if stalled too much
                if stall_count > max_stall:
                    break
            
            # If we used fewer calls than allocated to this restart, continue
            if calls_made >= budget:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
