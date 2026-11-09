import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bound_lo = -5.0
        self.bound_hi = 5.0
        self.start_time = None
        self.deadline = 325.0  # 5s margin

    def __call__(self, func):
        self.start_time = time.time()
        d = self.dim
        budget = self.budget
        lo = self.bound_lo
        hi = self.bound_hi

        rng = np.random.default_rng(42)

        # Phase 1: Scout Phase with structured + random points
        scout_budget = 30
        best_x = None
        best_f = np.inf
        scout_points = []

        points = []
        # Center
        points.append(np.zeros(d))
        # Axis-aligned points at various distances
        for dim_idx in range(d):
            for val in [1.0, -1.0, 2.0, -2.0, 3.0, -3.0]:
                pt = np.zeros(d)
                pt[dim_idx] = val
                points.append(pt)
        # Some random points to fill remaining scout budget
        n_random = max(0, scout_budget - len(points))
        for _ in range(n_random):
            points.append(rng.uniform(lo, hi, d))

        for x in points:
            if time.time() - self.start_time > self.deadline:
                break
            fx = func(x)
            scout_points.append((x, fx))
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        if best_x is None:
            best_x = np.zeros(d)
            best_f = func(best_x)

        # Phase 2: Deterministic Multi-Resolution Grid Descent
        # Coarse-to-fine coordinate-wise grid search with progressive step size reduction
        remaining_budget = budget - len(scout_points)
        best_x_grid = best_x.copy()
        best_f_grid = best_f

        # Define resolution levels: step sizes to try
        # Start coarse, get finer
        base_steps = [4.0, 2.0, 1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 0.00390625, 0.001953125, 0.0009765625, 0.00048828125]
        
        # For each resolution level, do coordinate-wise search
        # At each coordinate, try moving left and right by the step size
        # Keep track of budget usage
        
        calls_used = len(scout_points)
        
        for step_size in base_steps:
            if time.time() - self.start_time > self.deadline:
                break
            if calls_used >= budget:
                break
            
            # For this resolution, do coordinate-wise search
            # Each coordinate gets 2 evaluations (left and right) per iteration
            # We'll do multiple passes to allow convergence
            
            # Number of passes we can afford at this resolution
            # Each pass costs d*2 evaluations
            evals_per_pass = d * 2
            max_passes = max(1, (budget - calls_used) // evals_per_pass)
            
            # Limit passes to avoid over-focusing on one resolution
            max_passes = min(max_passes, 5)
            
            for pass_num in range(max_passes):
                if time.time() - self.start_time > self.deadline:
                    break
                if calls_used >= budget:
                    break
                
                improved = False
                for coord in range(d):
                    if time.time() - self.start_time > self.deadline:
                        break
                    if calls_used >= budget:
                        break
                    
                    # Try moving in positive direction
                    x_plus = best_x_grid.copy()
                    x_plus[coord] += step_size
                    if x_plus[coord] > hi:
                        x_plus[coord] = hi
                    
                    fx_plus = func(x_plus)
                    calls_used += 1
                    
                    if fx_plus < best_f_grid:
                        best_f_grid = fx_plus
                        best_x_grid = x_plus.copy()
                        improved = True
                    else:
                        # Try moving in negative direction
                        x_minus = best_x_grid.copy()
                        x_minus[coord] -= step_size
                        if x_minus[coord] < lo:
                            x_minus[coord] = lo
                        
                        fx_minus = func(x_minus)
                        calls_used += 1
                        
                        if fx_minus < best_f_grid:
                            best_f_grid = fx_minus
                            best_x_grid = x_minus.copy()
                            improved = True
                
                # If no improvement in this pass, we can stop early for this resolution
                # But only if we've done at least one pass
                if not improved and pass_num >= 1:
                    break
            
            # If we've used most of the budget, stop
            if calls_used >= budget:
                break

        # Phase 3: Final Polish with very fine steps
        if calls_used < budget and best_x_grid is not None:
            fine_steps = [0.0002, 0.0001, 0.00005, 0.00002, 0.00001]
            for step_size in fine_steps:
                if time.time() - self.start_time > self.deadline:
                    break
                if calls_used >= budget:
                    break
                
                # Do one coordinate-wise pass with this fine step
                for coord in range(d):
                    if time.time() - self.start_time > self.deadline:
                        break
                    if calls_used >= budget:
                        break
                    
                    # Try moving in positive direction
                    x_plus = best_x_grid.copy()
                    x_plus[coord] += step_size
                    if x_plus[coord] > hi:
                        x_plus[coord] = hi
                    
                    fx_plus = func(x_plus)
                    calls_used += 1
                    
                    if fx_plus < best_f_grid:
                        best_f_grid = fx_plus
                        best_x_grid = x_plus.copy()
                    
                    # Try moving in negative direction
                    if calls_used < budget:
                        x_minus = best_x_grid.copy()
                        x_minus[coord] -= step_size
                        if x_minus[coord] < lo:
                            x_minus[coord] = lo
                        
                        fx_minus = func(x_minus)
                        calls_used += 1
                        
                        if fx_minus < best_f_grid:
                            best_f_grid = fx_minus
                            best_x_grid = x_minus.copy()
                
                # Stop if no budget left
                if calls_used >= budget:
                    break
        
        # Ensure we have a valid result
        if best_x_grid is None:
            best_x_grid = best_x
            best_f_grid = best_f

        return best_x_grid, best_f_grid
