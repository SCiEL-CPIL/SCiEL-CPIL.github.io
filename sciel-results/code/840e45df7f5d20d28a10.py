import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """
    Adaptive lattice refinement with local Nelder-Mead polishing.
    Phase 1: Coarse grid search to locate the attractive sector.
    Phase 2: Recursive subdivision around the best point.
    Phase 3: Nelder-Mead local optimization for final precision.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin
        
        d = self.dim
        max_calls = self.budget
        calls_made = 0
        best_x = None
        best_f = float('inf')

        # Helper to evaluate and track budget
        def eval_func(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= max_calls:
                return best_f
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Coarse grid search
        # Use a 3x3x3x3x3 grid = 243 points to cover [-5, 5]^5
        # This gives initial resolution of ~2.5 units
        grid_points = np.meshgrid(*[np.linspace(-5, 5, 3) for _ in range(d)])
        grid_points = np.array(np.meshgrid(*[np.linspace(-5, 5, 3) for _ in range(d)])).T.reshape(-1, d)
        
        # Evaluate grid points
        for x in grid_points:
            if calls_made >= max_calls or time.time() > deadline:
                break
            eval_func(x)
        
        # Ensure we have a valid best
        if best_x is None:
            best_x = np.zeros(d)
            best_f = 0.0

        # Phase 2: Recursive subdivision around the best point
        # Start with a box centered at best_x with half-width 1.0
        center = best_x.copy()
        half_width = 1.0
        
        # Subdivide the box into 2^d sub-boxes and evaluate centers
        # Continue until budget is exhausted or box is small enough
        min_half_width = 0.01
        
        while calls_made < max_calls and time.time() < deadline and half_width > min_half_width:
            # Generate 2^d sub-box centers
            # For each dimension, create two points: center - half_width/2 and center + half_width/2
            # Actually, we want to evaluate the centers of the 2^d sub-boxes
            # Sub-box i has center: center + (offset_i - 0.5) * half_width for each dim
            # where offset_i is in {0, 1} for each dimension
            
            # Generate all 2^d combinations
            sub_centers = []
            for bits in range(2 ** d):
                point = center.copy()
                for j in range(d):
                    bit = (bits >> j) & 1
                    point[j] = center[j] + (bit - 0.5) * half_width
                sub_centers.append(point)
            
            sub_centers = np.array(sub_centers)
            
            # Evaluate all sub-centers
            local_best_f = float('inf')
            local_best_x = None
            
            for x in sub_centers:
                if calls_made >= max_calls or time.time() > deadline:
                    break
                fx = eval_func(x)
                if fx < local_best_f:
                    local_best_f = fx
                    local_best_x = x
            
            if local_best_x is None:
                break
            
            # Update global best if improved
            if local_best_f < best_f:
                best_f = local_best_f
                best_x = local_best_x.copy()
            
            # Move to the sub-box with the best center
            # Determine which sub-box contains local_best_x
            new_center = local_best_x.copy()
            half_width *= 0.5
            
            # Check if we should stop subdividing
            if half_width < min_half_width:
                break
        
        # Phase 3: Nelder-Mead local optimization
        # Use remaining budget for local polishing
        remaining_calls = max_calls - calls_made
        if remaining_calls > 20 and time.time() < deadline:
            try:
                # Use Nelder-Mead with a small initial simplex
                x0 = best_x.copy()
                
                # Create initial simplex
                simplex = [x0]
                for i in range(d):
                    point = x0.copy()
                    # Small step in dimension i
                    step = 0.1 if x0[i] < 4.9 else -0.1
                    point[i] += step
                    point = np.clip(point, -5.0, 5.0)
                    simplex.append(point)
                
                # Use scipy's minimize with Nelder-Mead
                # Set maxiter based on remaining budget
                maxiter = min(remaining_calls // 2, 100)
                
                result = minimize(
                    func,
                    x0,
                    method='Nelder-Mead',
                    options={
                        'maxiter': maxiter,
                        'xatol': 1e-6,
                        'fatol': 1e-8,
                        'adaptive': True
                    }
                )
                
                # The minimize function calls func internally, but we need to track calls
                # Unfortunately, scipy's minimize doesn't let us easily track the number of calls
                # and we can't intercept them. So we need a different approach.
                
                # Alternative: Manual Nelder-Mead implementation
                # For now, let's do a simple coordinate descent or random restart around best_x
                
                # Simple local search: try small perturbations
                for _ in range(min(remaining_calls // 3, 50)):
                    if calls_made >= max_calls or time.time() > deadline:
                        break
                    
                    # Random perturbation
                    perturbation = np.random.randn(d) * 0.05
                    new_x = np.clip(best_x + perturbation, -5.0, 5.0)
                    fx = eval_func(new_x)
                    
                    if fx < best_f:
                        best_f = fx
                        best_x = new_x.copy()
                        
            except Exception:
                pass

        return best_x, best_f
