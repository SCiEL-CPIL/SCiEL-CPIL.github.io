"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Coordinate Descent with Adaptive Step Sizes and Stagnation-Triggered Jumps."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        
        # Initialize
        x_best = rng.uniform(-5, 5, self.dim)
        f_best = func(x_best)
        budget_used = 1
        
        # Adaptive step sizes per dimension
        step_sizes = np.ones(self.dim)
        
        # Stagnation tracking
        stall_count = 0
        stall_threshold = 10 # Trigger jump if no improvement in 10 evals
        
        # Time management
        start_time = time.time()
        time_limit = 290.0 # Safety margin for 300s limit
        
        while budget_used < self.budget:
            # Check time limit
            if time.time() - start_time > time_limit:
                break
            
            improved = False
            
            # Coordinate Descent Phase
            for i in range(self.dim):
                if budget_used >= self.budget:
                    break
                if time.time() - start_time > time_limit:
                    break
                
                h = step_sizes[i]
                
                # Try positive step
                x_trial = x_best.copy()
                x_trial[i] += h
                x_trial = np.clip(x_trial, -5, 5)
                f_trial = func(x_trial)
                budget_used += 1
                
                if f_trial < f_best:
                    x_best = x_trial
                    f_best = f_trial
                    improved = True
                    step_sizes[i] *= 1.5
                    stall_count = 0
                else:
                    # Try negative step
                    x_trial[i] = x_best[i] - h
                    x_trial = np.clip(x_trial, -5, 5)
                    f_trial = func(x_trial)
                    budget_used += 1
                    
                    if f_trial < f_best:
                        x_best = x_trial
                        f_best = f_trial
                        improved = True
                        step_sizes[i] *= 1.5
                        stall_count = 0
                    else:
                        # No improvement in this direction
                        step_sizes[i] *= 0.5
                        # Prevent step size from becoming too small
                        if step_sizes[i] < 1e-3:
                            step_sizes[i] = 1.0
            
            if not improved:
                stall_count += 1
            else:
                stall_count = 0
            
            # Stagnation Triggered Jump
            if stall_count >= stall_threshold:
                # Decide on jump strategy
                # 50% Large Local Jump, 50% Global Restart
                if rng.random() < 0.5:
                    # Large local jump
                    sigma_jump = 2.0
                    x_jump = x_best + rng.normal(0, sigma_jump, self.dim)
                    x_jump = np.clip(x_jump, -5, 5)
                else:
                    # Global restart
                    x_jump = rng.uniform(-5, 5, self.dim)
                
                f_jump = func(x_jump)
                budget_used += 1
                
                if f_jump < f_best:
                    x_best = x_jump
                    f_best = f_jump
                    # Reset step sizes to default
                    step_sizes = np.ones(self.dim)
                else:
                    # Keep best, but perhaps increase step sizes to explore more broadly
                    step_sizes = np.ones(self.dim) * 2.0
                
                stall_count = 0

        return x_best, f_best
# EVOLVE-BLOCK-END
