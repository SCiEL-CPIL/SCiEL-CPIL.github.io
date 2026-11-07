"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start (1+1)-ES with interleaved execution for AOCC optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_starts = 5  # Number of parallel independent optimizers

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        
        # Initialize multiple independent (1+1)-ES instances
        # Using different seeds for diversity
        rng_seeds = [100 + i for i in range(self.num_starts)]
        optimizers = []
        global_best_x = None
        global_best_fx = float('inf')
        
        calls_made = 0
        max_calls = self.budget
        
        # Initialize states
        for i in range(self.num_starts):
            rng = np.random.default_rng(rng_seeds[i])
            x = rng.uniform(-5, 5, self.dim)
            fx = func(x)
            calls_made += 1
            
            if fx < global_best_fx:
                global_best_fx = fx
                global_best_x = x.copy()
                
            optimizers.append({
                'x': x,
                'fx': fx,
                'sigma': 2.0,
                'rng': rng
            })
            
            if calls_made >= max_calls:
                break

        # If we used all budget just initializing, return
        if calls_made >= max_calls:
            return global_best_x, global_best_fx

        # Interleaved execution to maximize early progress visibility
        current_opt_idx = 0
        while calls_made < max_calls:
            if time.time() > deadline:
                break
                
            opt = optimizers[current_opt_idx % self.num_starts]
            current_opt_idx += 1
            
            # Generate mutant
            y = np.clip(opt['x'] + opt['sigma'] * opt['rng'].standard_normal(self.dim), -5, 5)
            fy = func(y)
            calls_made += 1
            
            # Update global best
            if fy < global_best_fx:
                global_best_fx = fy
                global_best_x = y.copy()
                
            # Update local optimizer state
            if fy < opt['fx']:
                opt['x'] = y
                opt['fx'] = fy
                opt['sigma'] *= 1.5
            else:
                opt['sigma'] *= 0.87
                
            # Prevent sigma from becoming too small or too large
            opt['sigma'] = np.clip(opt['sigma'], 1e-6, 10.0)

        return global_best_x, global_best_fx
# EVOLVE-BLOCK-END
