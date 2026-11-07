import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280.0
        budget = self.budget
        dim = self.dim
        
        calls = [0]
        best_x = np.zeros(dim)
        best_f = float('inf')
        
        def evaluate(x):
            nonlocal best_x, best_f
            if calls[0] >= budget:
                return float('inf')
            if time.time() > deadline:
                return float('inf')
            calls[0] += 1
            x = np.clip(x, -5, 5)
            f = func(x)
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        rng = np.random.default_rng(42)
        
        # Phase 1: Coarse Global Grid Scan
        # 3 points per dimension: -5, 0, 5
        # Total points: 3^5 = 243
        # This fits within the 500 budget and provides global coverage.
        grid_points = []
        coords = [-5.0, 0.0, 5.0]
        
        # Generate all combinations
        for i in range(len(coords)):
            for j in range(len(coords)):
                for k in range(len(coords)):
                    for l in range(len(coords)):
                        for m in range(len(coords)):
                            grid_points.append(np.array([coords[i], coords[j], coords[k], coords[l], coords[m]]))
                            
        # Evaluate all coarse grid points
        for x in grid_points:
            if calls[0] >= budget or time.time() > deadline:
                break
            evaluate(x)
            
        # Phase 2: Local Derivative-Free Coordinate Descent (DFCD)
        # Start from the best point found in Phase 1.
        # We will perform coordinate-wise scans to refine the solution.
        # This structure avoids the trajectory traps of (1+1)-ES by explicitly
        # scanning each dimension in a local neighborhood.
        
        current_x = best_x.copy()
        current_f = best_f
        
        # Configuration for coordinate descent
        # We have approx 250 calls left.
        # 5 dims.
        # Let's do 2 iterations of coordinate scans.
        # Iteration 1: Range +/- 1.0, Step 0.1 (21 points/dim -> 105 calls)
        # Iteration 2: Range +/- 0.5, Step 0.05 (21 points/dim -> 105 calls)
        # Total 210 calls. Fits perfectly.
        
        # Iteration 1
        range_1 = 1.0
        step_1 = 0.1
        n_steps_1 = int(range_1 / step_1) * 2 + 1 # 21
        
        for _ in range(2): # 2 iterations
            for d in range(dim):
                if calls[0] >= budget or time.time() > deadline:
                    break
                
                # Scan dimension d
                # Local grid centered at current_x[d]
                # Clip to bounds [-5, 5]
                start_val = max(-5.0, current_x[d] - range_1)
                end_val = min(5.0, current_x[d] + range_1)
                
                # Generate points
                # Ensure we include the current best value in the scan
                scan_vals = np.arange(start_val, end_val + step_1/2, step_1)
                
                # If current_x[d] is not in scan_vals (due to floating point or bounds), add it
                # Actually, np.arange might miss it. Let's just evaluate the grid.
                # To save calls, we can skip if we are at the bound and the range is clipped, but simple is better.
                
                for val in scan_vals:
                    if calls[0] >= budget or time.time() > deadline:
                        break
                    new_x = current_x.copy()
                    new_x[d] = val
                    f_new = evaluate(new_x)
                    if f_new < current_f:
                        current_f = f_new
                        current_x = new_x.copy()
                        best_f = f_new
                        best_x = new_x.copy()
                        
            # Shrink range for next iteration if any improvement
            if current_f < best_f: # This check is redundant as best_f is updated, but logical for flow
                pass
            range_1 = range_1 * 0.5
            step_1 = step_1 * 0.5
            n_steps_1 = int(range_1 / step_1) * 2 + 1

        # Final check
        if calls[0] == 0:
            evaluate(np.zeros(dim))

        return best_x, best_f
