"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Derivative-Free Coordinate Descent (DFCD) with multi-resolution grid scan
    tailored for the bi-modal Lunacek bi-Rastrigin function f24."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        # We will track the best x and fx found so far
        best_x = None
        best_fx = float('inf')
        calls_made = 0

        def eval_point(x):
            nonlocal calls_made, best_x, best_fx
            fx = func(x)
            calls_made += 1
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
            return fx

        # --- Phase 1: Coarse global scan ---
        # For each dimension, scan a few points to find a reasonable starting region.
        # Lunacek bi-Rastrigin has two basins. A coarse scan helps avoid missing one.
        # We use a fixed grid per dimension. Let's use 5 points per dimension initially.
        # That's 5^5 = 3125 points, which exceeds budget 500.
        # So we need a smarter approach.
        
        # Strategy: 
        # 1. Start with a random point (or center) and do coordinate descent.
        # 2. Use multiple restarts with different resolutions.
        # 3. For each coordinate, scan a small set of offsets around the current value.
        
        # Since budget is 500 and dim is 5, we have about 100 evaluations per dimension on average.
        # Let's do a few rounds of coordinate descent with decreasing step sizes.
        
        # Initialize with a few random starting points to handle bi-modality.
        rng = np.random.default_rng(42)
        
        # We'll maintain a list of candidate points. Start with center and a few randoms.
        # But to be deterministic and robust, let's use a structured approach.
        
        # Phase 1: Coarse coordinate descent from center
        x = np.zeros(dim)
        eval_point(x)
        
        # Define step sizes for coordinate descent
        # Start with large steps to cover the domain, then refine
        step_sizes = [4.0, 2.0, 1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01, 0.005]
        
        # For each step size, do coordinate descent
        for step in step_sizes:
            # Check if we have enough budget left
            if calls_made >= budget - 10:
                break
                
            # Coordinate descent: for each dimension, try moving in +/- step
            improved = True
            while improved and calls_made < budget - 10:
                improved = False
                for i in range(dim):
                    if calls_made >= budget - 10:
                        break
                    
                    # Try positive direction
                    x_new = x.copy()
                    x_new[i] = min(5.0, x_new[i] + step)
                    if not np.allclose(x_new, x):
                        fx_new = eval_point(x_new)
                        if fx_new < best_fx:
                            x = x_new
                            improved = True
                            continue
                    
                    # Try negative direction
                    x_new = x.copy()
                    x_new[i] = max(-5.0, x_new[i] - step)
                    if not np.allclose(x_new, x):
                        fx_new = eval_point(x_new)
                        if fx_new < best_fx:
                            x = x_new
                            improved = True
        
        # Phase 2: Local refinement with finer steps around the best point
        if best_x is not None and calls_made < budget:
            x = best_x.copy()
            fx = best_fx
            
            # Very fine steps for final polish
            fine_steps = [0.002, 0.001, 0.0005, 0.0002]
            for step in fine_steps:
                if calls_made >= budget - 5:
                    break
                
                improved = True
                while improved and calls_made < budget - 5:
                    improved = False
                    for i in range(dim):
                        if calls_made >= budget - 5:
                            break
                        
                        # Try positive direction
                        x_new = x.copy()
                        x_new[i] = min(5.0, x_new[i] + step)
                        if not np.allclose(x_new, x):
                            fx_new = eval_point(x_new)
                            if fx_new < best_fx:
                                x = x_new
                                improved = True
                                continue
                        
                        # Try negative direction
                        x_new = x.copy()
                        x_new[i] = max(-5.0, x_new[i] - step)
                        if not np.allclose(x_new, x):
                            fx_new = eval_point(x_new)
                            if fx_new < best_fx:
                                x = x_new
                                improved = True
        
        return best_x, best_fx

# EVOLVE-BLOCK-END
