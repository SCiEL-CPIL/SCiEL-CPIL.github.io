"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Deterministic Global-Local Hybrid Optimizer.
    
    Strategy:
    1. Coarse-to-fine deterministic grid search to locate the global basin.
       This is highly efficient for Schaffer's F7 (single global minimum, 
       ill-conditioned but not multimodal in the global sense) and avoids 
       the random sampling overhead of evolutionary strategies.
    2. Adaptive Coordinate-wise Line Search (1D optimization per dimension).
       This exploits the axis-aligned nature of the search space and 
       handles ill-conditioning by adapting step sizes per dimension 
       based on local curvature estimates (parabolic interpolation).
    3. Deadline guards to ensure validity.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin before 300s limit
        dim = self.dim
        budget = self.budget
        
        best_x = np.zeros(dim)
        best_f = float('inf')
        calls = 0

        def timed_call(x):
            nonlocal calls, best_x, best_f
            if calls >= budget:
                return float('inf')
            fx = func(np.asarray(x, dtype=float))
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = np.array(x, dtype=float)
            return fx

        def time_left():
            return deadline - time.time()

        # --- Phase 1: Coarse-to-Fine Deterministic Grid Search ---
        # Schaffer's F7 is ill-conditioned. A coarse grid helps identify the 
        # approximate location of the minimum without wasting calls on 
        # fine-grained random exploration in the wrong basin.
        
        # We use a hierarchical grid. 
        # Level 1: 3 points per dimension (27 points for dim=5)
        # Level 2: 5 points per dimension (3125 points) - might be too many, 
        #          so we adapt based on budget and time.
        # Level 3: 9 points per dimension (59049 points) - too many.
        
        # Let's use a smarter approach: 
        # 1. Sample corners and center (2^dim + 1 points)
        # 2. If budget allows, do a 3^dim grid.
        # 3. If budget allows, do a 5^dim grid around the best point.
        
        # Generate grid points
        def generate_grid(points_per_dim, center, range_val):
            # points_per_dim: number of points along each axis
            # center: center of the grid
            # range_val: half-width of the grid
            if points_per_dim == 1:
                return [center]
            
            # Create 1D grid
            # Use linspace but ensure endpoints are included
            # For odd number of points, center is exactly in the middle
            grid_1d = np.linspace(center - range_val, center + range_val, points_per_dim)
            
            # Generate all combinations
            grids = np.meshgrid(*([grid_1d] * dim), indexing='ij')
            points = np.array(np.vstack([g.ravel() for g in grids]).T)
            return points

        # Initial broad search
        # The minimum is in [-4, 4]^5. The box is [-5, 5]^5.
        # Start with a 3x3x... grid over the entire box.
        # 3^5 = 243 points. This is affordable.
        
        grid_1 = generate_grid(3, np.zeros(dim), 5.0)
        # Add center explicitly if not present (linspace( -5, 5, 3) is [-5, 0, 5])
        # Actually linspace(-5, 5, 3) is [-5, 0, 5]. Good.
        
        for p in grid_1:
            if calls >= budget or time_left() < 10.0:
                break
            timed_call(p)
        
        # If we have budget left, refine around the best point found so far.
        # Use a 5x5x... grid around best_x with a smaller range.
        # 5^5 = 3125 points. If budget is 500, we can't do all.
        # We will do a partial grid or a smaller grid.
        
        # Let's do a 3x3x... grid around best_x with range 2.5
        # 3^5 = 243 points.
        if calls < budget and time_left() > 10.0:
            grid_2 = generate_grid(3, best_x, 2.5)
            for p in grid_2:
                if calls >= budget or time_left() < 10.0:
                    break
                timed_call(p)
                
        # If we have budget left, refine further.
        # 3x3x... grid around best_x with range 1.0
        # 3^5 = 243 points.
        if calls < budget and time_left() > 10.0:
            grid_3 = generate_grid(3, best_x, 1.0)
            for p in grid_3:
                if calls >= budget or time_left() < 10.0:
                    break
                timed_call(p)

        # --- Phase 2: Adaptive Coordinate-wise Line Search ---
        # Now that we are close to the minimum, use 1D line searches to 
        # exploit the ill-conditioned structure.
        # We iterate over dimensions, performing a 1D optimization along 
        # that axis while holding others fixed.
        
        # Initial step size for line search
        step_sizes = np.full(dim, 0.5)
        
        # We will perform multiple sweeps.
        # Each sweep goes through all dimensions.
        # Within each dimension, we use a simple parabolic/golden section 
        # hybrid to find the minimum along that axis.
        
        max_sweeps = 50  # Safety limit
        sweep_count = 0
        
        while calls < budget and time_left() > 5.0 and sweep_count < max_sweeps:
            improved = False
            for d in range(dim):
                if calls >= budget or time_left() < 5.0:
                    break
                
                # Perform 1D search along dimension d
                # Current point: best_x
                # We try to minimize f(x) where x[d] varies, others fixed.
                
                # Define 1D function
                def f_1d(t):
                    x_temp = best_x.copy()
                    x_temp[d] = t
                    return timed_call(x_temp)
                
                # Initial interval
                x0 = best_x[d]
                h = step_sizes[d]
                
                # Check neighbors to find a valley
                f0 = f_1d(x0)
                f1 = f_1d(x0 + h)
                f2 = f_1d(x0 - h)
                
                # Determine direction
                if f1 < f0 and f1 < f2:
                    # Move right
                    xl, xr = x0, x0 + 2*h
                    fxl, fxr = f0, f1
                    # Refine
                    # Use parabolic interpolation if possible
                    # For simplicity, use golden section or binary search on slope
                    # Since we have f0, f1, we can estimate slope.
                    # Let's just do a few bisection steps or parabolic fit.
                    # Parabolic fit with 3 points: (x0-h, f2), (x0, f0), (x0+h, f1)
                    # If f1 < f0 < f2, min is to the right.
                    # If f2 < f0 < f1, min is to the left.
                    
                    # Let's use a robust 3-point parabolic interpolation
                    # Points: (x0-h, f2), (x0, f0), (x0+h, f1)
                    # Note: f2 is at x0-h, f0 at x0, f1 at x0+h
                    
                    # If we have a valley, we can fit a parabola.
                    # We need to ensure we don't go out of bounds [-5, 5]
                    
                    # Simple approach: Binary search on the derivative (numerical)
                    # Or just use Golden Section Search on the interval [x0, x0+h] or [x0-h, x0]
                    
                    # Let's use a simple adaptive step reduction/expansion
                    # and bisection.
                    
                    # Determine bracket
                    if f1 < f2:
                        # Min is likely between x0 and x0+h
                        a, b = x0, x0 + h
                        fa, fb = f0, f1
                    else:
                        # Min is likely between x0-h and x0
                        a, b = x0 - h, x0
                        fa, fb = f2, f0
                    
                    # Perform 5-10 iterations of bisection/golden section
                    # Golden Section Search
                    gr = (np.sqrt(5) - 1) / 2  # 0.618
                    c = b - gr * (b - a)
                    d = a + gr * (b - a)
                    fc = f_1d(c)
                    fd = f_1d(d)
                    
                    for _ in range(8):
                        if calls >= budget or time_left() < 5.0:
                            break
                        if fc < fd:
                            b = d
                            fb = fd
                            d = c
                            fd = fc
                            c = b - gr * (b - a)
                            fc = f_1d(c)
                        else:
                            a = c
                            fa = fc
                            c = d
                            fc = fd
                            d = a + gr * (b - a)
                            fd = f_1d(d)
                    
                    # Best point in this 1D search
                    if fc < fd:
                        best_1d_x = c
                        best_1d_f = fc
                    else:
                        best_1d_x = d
                        best_1d_f = fd
                    
                    # Update best_x
                    if best_1d_f < best_f:
                        best_x[d] = best_1d_x
                        improved = True
                        # Adapt step size: reduce if we found a better point
                        # This helps in ill-conditioned regions
                        step_sizes[d] *= 0.7
                    else:
                        # No improvement, maybe step was too small or we are at min
                        # Try increasing step size slightly to escape flat regions
                        step_sizes[d] *= 1.2
                        # Cap step size
                        step_sizes[d] = min(step_sizes[d], 2.0)

                elif f2 < f0 and f2 < f1:
                    # Move left
                    # Similar logic, but direction is negative
                    a, b = x0 - h, x0
                    fa, fb = f2, f0
                    # Same golden section logic
                    gr = (np.sqrt(5) - 1) / 2
                    c = b - gr * (b - a)
                    d = a + gr * (b - a)
                    fc = f_1d(c)
                    fd = f_1d(d)
                    
                    for _ in range(8):
                        if calls >= budget or time_left() < 5.0:
                            break
                        if fc < fd:
                            b = d
                            fb = fd
                            d = c
                            fd = fc
                            c = b - gr * (b - a)
                            fc = f_1d(c)
                        else:
                            a = c
                            fa = fc
                            c = d
                            fc = fd
                            d = a + gr * (b - a)
                            fd = f_1d(d)
                    
                    if fc < fd:
                        best_1d_x = c
                        best_1d_f = fc
                    else:
                        best_1d_x = d
                        best_1d_f = fd
                        
                    if best_1d_f < best_f:
                        best_x[d] = best_1d_x
                        improved = True
                        step_sizes[d] *= 0.7
                    else:
                        step_sizes[d] *= 1.2
                        step_sizes[d] = min(step_sizes[d], 2.0)

                else:
                    # Local minimum or flat region
                    # Try a larger step to see if we can escape
                    h_new = h * 2.0
                    if h_new < 5.0:
                        f1_new = f_1d(x0 + h_new)
                        f2_new = f_1d(x0 - h_new)
                        if f1_new < f0:
                            # Restart 1D search with larger h
                            # For simplicity, just update step size and let next sweep try
                            step_sizes[d] = h_new
                        elif f2_new < f0:
                            step_sizes[d] = h_new
                        else:
                            # Truly stuck? Reduce step size to refine
                            step_sizes[d] = h * 0.5
                    else:
                        # At boundary or large step, just reduce
                        step_sizes[d] = h * 0.5

            sweep_count += 1
            if not improved and sweep_count > 5:
                # Stagnation, try a random perturbation to escape local min
                # This is a fallback to ensure global coverage if grid missed
                # But Schaffer F7 is single modality, so this is less likely needed
                # unless we are stuck in a flat region.
                # Add a small random jitter
                jitter = np.random.normal(0, 0.1, dim)
                x_jitter = np.clip(best_x + jitter, -5, 5)
                f_jitter = timed_call(x_jitter)
                if f_jitter < best_f:
                    improved = True

        return best_x, best_f

# EVOLVE-BLOCK-END
