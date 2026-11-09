"""Multi-Start Coordinate Descent optimizer for BBOB-style functions."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Pure Multi-Start Coordinate Descent with fixed partitioning."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        time_limit = 325.0  # Safety margin before 330s kill
        
        dim = self.dim
        budget = self.budget
        
        best_x_global = None
        best_f_global = np.inf
        
        # Strategy: Pure parallel Multi-Start
        # Partition 300 calls into 30 restarts of 10 calls each.
        # No communication between restarts except updating the global best.
        # This maximizes coverage and avoids getting stuck in a single local basin
        # for too long, which is beneficial for AOCC (early progress).
        
        n_restarts = 30
        calls_per_restart = 10
        
        # Fixed step size for coordinate descent
        # 0.5 is a reasonable step for [-5, 5] box, allowing exploration within a restart
        step_size = 0.5
        
        # We will use a deterministic sequence of random seeds for reproducibility 
        # if needed, but np.random is fine.
        
        for r in range(n_restarts):
            if budget <= 0:
                break
            if time.time() - start_time > time_limit:
                break
            
            # 1. Random start
            x = np.random.uniform(-4, 4, dim)
            f = func(x)
            budget -= 1
            if budget < 0: budget = 0
            
            if f < best_f_global:
                best_f_global = f
                best_x_global = x.copy()
            
            # 2. Coordinate Descent for remaining calls in this restart
            # We have calls_per_restart - 1 calls left for this restart
            remaining_in_restart = calls_per_restart - 1
            
            # Cycle through dimensions
            dim_idx = 0
            while remaining_in_restart > 0:
                if time.time() - start_time > time_limit:
                    break
                
                d = dim_idx % dim
                dim_idx += 1
                
                # Try positive step
                x_pos = x.copy()
                x_pos[d] += step_size
                x_pos = np.clip(x_pos, -5, 5)
                f_pos = func(x_pos)
                budget -= 1
                remaining_in_restart -= 1
                
                if f_pos < f:
                    x = x_pos
                    f = f_pos
                    if f < best_f_global:
                        best_f_global = f
                        best_x_global = x.copy()
                    # If improved, we could try negative step next or reset cycle?
                    # Standard coordinate descent often tries both directions.
                    # To save budget, we only try the other direction if we have budget.
                    # But for simplicity and strict budget partitioning, we just move to next dim.
                    # Actually, trying both directions in 10 calls is hard.
                    # Let's stick to trying +step. If it fails, try -step?
                    # That would consume 2 calls per dimension check.
                    # With 10 calls, we can check 5 dimensions.
                    # Let's do: Try +step. If no improvement, try -step.
                    # This consumes 2 calls per dimension.
                    # So in 9 calls (after initial), we can check 4.5 dimensions.
                    
                    if remaining_in_restart > 0:
                        # Try negative step
                        x_neg = x.copy()
                        x_neg[d] -= step_size
                        x_neg = np.clip(x_neg, -5, 5)
                        f_neg = func(x_neg)
                        budget -= 1
                        remaining_in_restart -= 1
                        
                        if f_neg < f:
                            x = x_neg
                            f = f_neg
                            if f < best_f_global:
                                best_f_global = f
                                best_x_global = x.copy()
                else:
                    # No improvement with +step, try -step
                    if remaining_in_restart > 0:
                        x_neg = x.copy()
                        x_neg[d] -= step_size
                        x_neg = np.clip(x_neg, -5, 5)
                        f_neg = func(x_neg)
                        budget -= 1
                        remaining_in_restart -= 1
                        
                        if f_neg < f:
                            x = x_neg
                            f = f_neg
                            if f < best_f_global:
                                best_f_global = f
                                best_x_global = x.copy()
                
                # Move to next dimension
                # If we used 2 calls, we advanced 1 dimension.
                # If we used 1 call (improvement found, skipped neg), we advanced 1 dimension.
                # The loop structure above handles the call counting.
                # We just increment dim_idx in the while loop condition implicitly?
                # No, dim_idx is incremented at the start of the while loop iteration.
                # So each iteration of the while loop checks one dimension (using 1 or 2 calls).
                # This is correct.

        return best_x_global, best_f_global
# EVOLVE-BLOCK-END
