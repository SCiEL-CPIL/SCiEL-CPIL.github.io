import numpy as np
import time
import sys
from scipy.optimize import differential_evolution, minimize

def compute_volume_2d(positions_2d):
    """
    Computes the exact volume of the union of 64 sheared tubes.
    positions_2d is shape (64, 2).
    Tube k is defined by i = k // 8, j = k % 8.
    At height c, tube k is [x_k + c*i/8, x_k + c*i/8 + 1/8] x [y_k + c*j/8, y_k + c*j/8 + 1/8].
    
    The volume is the integral over c in [0,1] of the area of the union of the 64 squares at that height.
    The area of the union of axis-aligned rectangles can be computed exactly using a sweep line or 
    by identifying all critical x and y coordinates (breakpoints) and summing the areas of the 
    resulting grid cells that are covered.
    
    However, the breakpoints depend on c. 
    x-coord of left edge of tube k at c: Lx_k(c) = x_k + c*i_k/8
    x-coord of right edge: Rx_k(c) = Lx_k(c) + 1/8
    y-coord of bottom edge: Ly_k(c) = y_k + c*j_k/8
    y-coord of top edge: Ry_k(c) = Ly_k(c) + 1/8
    
    The area of the union of these rectangles as a function of c is piecewise linear? No, 
    the area is piecewise constant? No, the coordinates are linear in c, so the area is 
    piecewise quadratic? Actually, the area of the union of rectangles with linearly moving 
    edges is a piecewise linear function of c? 
    Let's check: Area = sum of widths * heights. If the combinatorial structure of overlaps 
    doesn't change, the area is a linear function of c? 
    Actually, the area of a rectangle is (Rx-Lx)*(Ry-Ly) = (1/8)*(1/8) = 1/64. 
    Wait, the area of the union is not just the sum of areas.
    
    Let's use a different approach. The volume is the measure of the set of points (a,b,c) 
    inside at least one tube.
    This is equivalent to the integral over c of the area of the union of the 2D projections 
    (which are just the squares themselves, shifted).
    
    For a fixed c, we have 64 axis-aligned squares.
    The area of the union of axis-aligned squares can be computed exactly.
    The function A(c) is the area of the union at height c.
    The volume is integral_0^1 A(c) dc.
    
    A(c) is a piecewise linear function of c? 
    The boundaries of the squares are linear in c.
    The area of the union of such moving squares is a piecewise linear function of c?
    Actually, the area of the union of rectangles with linearly varying coordinates is 
    a piecewise linear function of the parameter if the combinatorial structure of 
    overlaps remains constant. But the overlaps change when edges cross.
    Edge crossing times are linear equations in c.
    So A(c) is piecewise linear.
    We can find all breakpoints c where any two vertical edges (Lx or Rx) coincide, 
    or any two horizontal edges (Ly or Ry) coincide.
    There are 128 vertical edges and 128 horizontal edges.
    Number of pairwise comparisons: 128*127/2 ~ 8000 for x, same for y. Total ~16000 breakpoints.
    This is feasible to compute exactly.
    
    Steps:
    1. Collect all c values in (0,1) where any Lx_i(c) == Lx_j(c), Lx_i(c) == Rx_j(c), 
       Rx_i(c) == Lx_j(c), Rx_i(c) == Rx_j(c).
       Similarly for y.
    2. Sort these breakpoints.
    3. For each interval between consecutive breakpoints, pick a midpoint c_mid.
    4. Compute the area of the union of the 64 squares at c_mid exactly.
    5. Integrate A(c) using the trapezoidal rule? No, A(c) is linear in each interval, 
       so the integral over the interval is exactly (c_end - c_start) * A(c_mid).
       Wait, is A(c) linear?
       Area of union = sum over cells in grid formed by edges.
       For a fixed combinatorial structure, the area is a linear function of c?
       Let's verify. 
       Area = ∫ dx ∫ dy I(x,y in union).
       This is not obviously linear.
       However, the area of the union of rectangles with coordinates linear in t 
       is a piecewise linear function of t.
       Reference: The area of the union of moving segments/rectangles.
       Actually, the area of the union of axis-aligned rectangles whose edges move 
       linearly with time is a piecewise linear function of time.
       Why? The area is the sum of areas of disjoint cells in the arrangement.
       The boundaries of these cells are segments of the rectangle edges.
       The area of a cell is (x_right - x_left) * (y_top - y_bottom).
       x_right and x_left are linear in c. y_top and y_bottom are linear in c.
       So the area of each cell is a quadratic function of c? 
       (a c + b)(d c + e) = ad c^2 + ...
       So A(c) is piecewise QUADRATIC.
       
       Therefore, we should use Simpson's rule or integrate the quadratic exactly.
       To integrate exactly, we need the values at the endpoints and the midpoint?
       If it's quadratic, we can fit a parabola through 3 points.
       Or simply, since we have the breakpoints, we can evaluate A(c) at the breakpoints 
       and midpoints and use Simpson's 1/3 rule?
       Actually, if we know it's quadratic on each interval, we can just evaluate at 
       c_start, c_mid, c_end and use the formula for integrating a quadratic.
       Or simpler: just use a fine grid if the number of breakpoints is small?
       No, exact integration is better.
       
       Let's implement exact integration assuming piecewise quadratic.
       For each interval [c_k, c_{k+1}], compute A(c_k), A(c_mid), A(c_{k+1}).
       Integral = (c_{k+1} - c_k) / 6 * (A(c_k) + 4*A(c_mid) + A(c_{k+1})).
       
    """
    N = 8
    i_idx = np.arange(N)
    j_idx = np.arange(N)
    i_grid, j_grid = np.meshgrid(i_idx, j_idx, indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    # positions_2d is (64, 2)
    x_base = positions_2d[:, 0]
    y_base = positions_2d[:, 1]
    
    # Breakpoints for x-coordinates
    # Lx_k(c) = x_base[k] + c * i_flat[k] / 8
    # Rx_k(c) = x_base[k] + 1/8 + c * i_flat[k] / 8
    # We need c where Lx_i = Lx_j, Lx_i = Rx_j, Rx_i = Lx_j, Rx_i = Rx_j.
    # x_i + c*i_i/8 = x_j + c*i_j/8  => c*(i_i - i_j)/8 = x_j - x_i
    # x_i + c*i_i/8 = x_j + 1/8 + c*i_j/8 => c*(i_i - i_j)/8 = x_j + 1/8 - x_i
    # etc.
    
    breakpoints = [0.0, 1.0]
    
    # Helper to add breakpoint if valid
    def add_bp(c_val):
        if 0.0 < c_val < 1.0:
            breakpoints.append(c_val)
            
    # X-breakpoints
    for k1 in range(64):
        for k2 in range(k1+1, 64):
            di = i_flat[k1] - i_flat[k2]
            if di == 0: continue
            denom = di / 8.0
            
            # L1 = L2
            c_val = (x_base[k2] - x_base[k1]) / denom
            add_bp(c_val)
            
            # L1 = R2
            c_val = (x_base[k2] + 1/8 - x_base[k1]) / denom
            add_bp(c_val)
            
            # R1 = L2
            c_val = (x_base[k2] - x_base[k1] - 1/8) / denom
            add_bp(c_val)
            
            # R1 = R2
            c_val = (x_base[k2] + 1/8 - x_base[k1] - 1/8) / denom
            add_bp(c_val)
            
    # Y-breakpoints
    for k1 in range(64):
        for k2 in range(k1+1, 64):
            dj = j_flat[k1] - j_flat[k2]
            if dj == 0: continue
            denom = dj / 8.0
            
            # B1 = B2
            c_val = (y_base[k2] - y_base[k1]) / denom
            add_bp(c_val)
            
            # B1 = T2
            c_val = (y_base[k2] + 1/8 - y_base[k1]) / denom
            add_bp(c_val)
            
            # T1 = B2
            c_val = (y_base[k2] - y_base[k1] - 1/8) / denom
            add_bp(c_val)
            
            # T1 = T2
            c_val = (y_base[k2] + 1/8 - y_base[k1] - 1/8) / denom
            add_bp(c_val)
            
    breakpoints = sorted(set(breakpoints))
    
    total_volume = 0.0
    
    def compute_area_at_c(c_val):
        # Compute union area of 64 squares at height c_val
        # Left x: Lx_k = x_base[k] + c_val * i_flat[k] / 8
        # Right x: Rx_k = Lx_k + 1/8
        # Bottom y: Ly_k = y_base[k] + c_val * j_flat[k] / 8
        # Top y: Ry_k = Ly_k + 1/8
        
        Lx = x_base + c_val * i_flat / 8.0
        Rx = Lx + 1/8.0
        Ly = y_base + c_val * j_flat / 8.0
        Ry = Ly + 1/8.0
        
        # Collect all unique x and y coordinates
        x_coords = np.unique(np.concatenate([Lx, Rx]))
        y_coords = np.unique(np.concatenate([Ly, Ry]))
        
        # Create grid
        # Area of union = sum over cells (dx * dy) if cell is covered
        # A cell is covered if it intersects any rectangle.
        # Since rectangles are axis aligned, a cell [x_i, x_{i+1}] x [y_j, y_{j+1}] 
        # is covered if there exists a rectangle k such that 
        # Lx_k <= x_i and Rx_k >= x_{i+1} and Ly_k <= y_j and Ry_k >= y_{j+1}?
        # No, that's for full containment. 
        # A cell is covered if its interior is in the union.
        # Since the grid is formed by the rectangle edges, any cell is either 
        # fully inside the union or fully outside?
        # Yes, because the boundaries of the union are subsets of the grid lines.
        # So we just need to check if the center of the cell is in any rectangle.
        
        x_mid = (x_coords[:-1] + x_coords[1:]) / 2.0
        y_mid = (y_coords[:-1] + y_coords[1:]) / 2.0
        dx = x_coords[1:] - x_coords[:-1]
        dy = y_coords[1:] - y_coords[:-1]
        
        # Grid of midpoints
        X_mid, Y_mid = np.meshgrid(x_mid, y_mid, indexing="ij")
        
        # Check coverage
        # For each cell (idx_x, idx_y), check if any rectangle covers it.
        # Rectangle k covers cell if Lx_k <= X_mid[idx] and Rx_k >= X_mid[idx] and Ly_k <= Y_mid[idx] and Ry_k >= Y_mid[idx].
        # Since X_mid is in (Lx_k, Rx_k) if covered?
        # Actually, if the cell is in the union, its center is in the union.
        
        # Efficient check:
        # For each x, find which rectangles cover it.
        # For each y, find which rectangles cover it.
        # This is 2D range query.
        # With 64 rects and small grid, brute force is fine.
        
        area = 0.0
        # Iterate over cells
        # x_coords size <= 128, y_coords size <= 128. Cells <= 16000.
        # 16000 * 64 checks = 1M ops. Fast enough in Python? Maybe slow.
        # Vectorize:
        
        # X_mid shape (Nx, Ny), Y_mid shape (Nx, Ny)
        # Lx shape (64,), Rx shape (64,)
        # We want to know for each cell if ANY rect covers it.
        
        # Condition: Lx_k <= X_mid AND Rx_k >= X_mid AND Ly_k <= Y_mid AND Ry_k >= Y_mid
        # Let's create boolean arrays.
        
        # To avoid 64 * Nx * Ny loops, we can do:
        # For each x-index, determine which rects cover that x-slice.
        # Then for each y-index, determine which rects cover that y-slice.
        # Intersection?
        
        # Simpler: 
        # covered = np.zeros((Nx, Ny), dtype=bool)
        # for k in range(64):
        #     mask_x = (Lx[k] <= X_mid) & (Rx[k] >= X_mid)
        #     mask_y = (Ly[k] <= Y_mid) & (Ry[k] >= Y_mid)
        #     covered |= (mask_x & mask_y)
        # area = np.sum(covered) * np.sum(dx * dy)? No, dx and dy vary.
        # area = np.sum(covered * dx[:, None] * dy[None, :])
        
        # This is 64 iterations of Nx*Ny boolean ops.
        # Nx, Ny <= 128. 128*128 = 16k. 64*16k = 1M. 
        # This will be called ~16000 times (breakpoints). 16000 * 1M = 16 billion ops. TOO SLOW.
        
        # We need a faster area calculation or fewer evaluations.
        
        # Alternative: 
        # The area of the union of rectangles can be computed in O(N^3) or O(N^2 log N) 
        # using sweep line.
        # But we need to do it for many c values.
        
        # Is there a way to avoid so many breakpoints?
        # Maybe we don't need exact integration. 
        # The task says V is a fixed-sample Monte Carlo estimate.
        # But the score is 1/V.
        # If we minimize the true volume, we maximize the score.
        # But the evaluator uses MC.
        # The MC estimate has noise.
        # However, the "seed" program used 1D cost.
        # If we can compute the true volume faster, we can optimize it directly.
        
        # Let's optimize the area calculation.
        # Use the fact that the grid is small.
        # Instead of looping k, can we vectorize over k?
        # Lx shape (64, 1, 1), X_mid shape (1, Nx, Ny) -> broadcast?
        # Lx <= X_mid: (64, Nx, Ny)
        # This creates a 64*128*128 array = 1M bools. 
        # 1M bools is 1MB. 
        # Doing this 16000 times is 16GB of memory traffic. 
        # Might be borderline but probably too slow in Python/NumPy overhead.
        
        # Let's try to reduce breakpoints.
        # Do we need all pairwise breakpoints?
        # The area function A(c) is piecewise quadratic.
        # The breakpoints are where the combinatorial structure changes.
        # Maybe we can sample fewer points?
        # If we just use Simpson's rule with, say, 100 intervals, is it accurate enough?
        # The score difference between 1.33 and 1.34 is small.
        # Maybe approximate integration is good enough for optimization, 
        # and we only compute exact volume at the end?
        
        # Let's use a fixed grid of c values for the cost function during optimization.
        # This makes the cost function smooth and fast to evaluate.
        
        return 0.0 # Placeholder
        
    # Revised Strategy:
    # 1. Define a fast cost function that approximates the volume using a fixed grid of c values.
    #    For each c, compute area of union of 64 rectangles.
    #    To speed up area calculation, use a sweep line or simple grid coverage with vectorization.
    # 2. Use Differential Evolution or Basin Hopping to minimize this cost.
    # 3. At the end, compute the exact volume for the best candidate to report? 
    #    No, the evaluator computes V itself. We just need to return positions.
    #    So we just need to find positions that minimize the true volume.
    #    If our approximate cost correlates well with true volume, we are good.
    
    # Let's implement the fast area calculation.
    def area_of_union(x_base, y_base, i_flat, j_flat, c_val):
        Lx = x_base + c_val * i_flat / 8.0
        Rx = Lx + 1/8.0
        Ly = y_base + c_val * j_flat / 8.0
        Ry = Ly + 1/8.0
        
        # Sweep line algorithm for area of union of rectangles
        # Events: (x, type, y1, y2)
        # type: 0 for left edge, 1 for right edge
        events = []
        for k in range(64):
            events.append((Lx[k], 0, Ly[k], Ry[k]))
            events.append((Rx[k], 1, Ly[k], Ry[k]))
            
        events.sort(key=lambda x: x[0])
        
        active_intervals = []
        area = 0.0
        last_x = events[0][0]
        
        for x, typ, y1, y2 in events:
            if x > last_x:
                # Calculate height of union of active intervals
                # Merge active intervals
                if active_intervals:
                    active_intervals.sort(key=lambda i: i[0])
                    merged = []
                    cur_l, cur_r = active_intervals[0]
                    for l, r in active_intervals[1:]:
                        if l <= cur_r:
                            if r > cur_r:
                                cur_r = r
                        else:
                            merged.append((cur_l, cur_r))
                            cur_l, cur_r = l, r
                    merged.append((cur_l, cur_r))
                    height = sum(r - l for l, r in merged)
                else:
                    height = 0.0
                    
                area += height * (x - last_x)
                last_x = x
                
            if typ == 0:
                active_intervals.append((y1, y2))
            else:
                # Remove interval
                # Find and remove
                for idx, (l, r) in enumerate(active_intervals):
                    if l == y1 and r == y2:
                        active_intervals.pop(idx)
                        break
                        
        return area

    # This sweep line is O(N log N) per c.
    # If we use 50 c-values, 50 * 64 log 64 ~ 2000 ops. Very fast.
    
    def cost_function(positions_2d):
        # positions_2d is (64, 2)
        x_base = positions_2d[:, 0]
        y_base = positions_2d[:, 1]
        
        # Fixed grid for c
        c_vals = np.linspace(0.0, 1.0, 21) # 21 points
        
        total_area = 0.0
        for c in c_vals:
            total_area += area_of_union(x_base, y_base, i_flat, j_flat, c)
            
        # Approximate integral (Simpson or Trapezoid)
        # Trapezoid: sum * (1/20)
        volume = total_area * (1.0 / 20.0)
        
        # Box area
        X0 = np.min(x_base)
        X1 = np.max(x_base + (i_flat + 1) / 8.0) # Wait, i_flat is the slope index?
        # The task says: X1 = max_k (x_k + (i+1)/8). 
        # Here i is the row index of the tube in the 8x8 grid?
        # "Row k is tube (i, j) with i = k // 8 and j = k % 8."
        # "tube k is ... [x_k + c*i/8, ...]"
        # So the shift is c*i/8.
        # The box X1 is max_k (x_k + (i+1)/8). 
        # Why (i+1)/8? 
        # Maybe it's a typo in my understanding or the task?
        # Let's re-read: "X1 = max_k (x_k + (i+1)/8)".
        # If i=7, (i+1)/8 = 1.
        # If i=0, (i+1)/8 = 1/8.
        # This seems to define the bounding box of the *sheared* tubes?
        # At c=1, x range is [x_k + i/8, x_k + i/8 + 1/8].
        # Max x is x_k + i/8 + 1/8.
        # The task says X1 = max_k (x_k + (i+1)/8).
        # Note that i/8 + 1/8 = (i+1)/8.
        # So X1 is the max right edge at c=1.
        # Similarly Y1 = max_k (y_k + (j+1)/8).
        
        X0 = np.min(x_base)
        X1 = np.max(x_base + (i_flat + 1) / 8.0)
        Y0 = np.min(y_base)
        Y1 = np.max(y_base + (j_flat + 1) / 8.0)
        
        box_area = (X1 - X0) * (Y1 - Y0)
        
        if box_area <= 1e-9:
            return 1e10
            
        return volume / box_area # Fraction of box covered? 
        # Wait, the score is 1/V. V is the volume estimate.
        # V = fraction * box_area.
        # So minimizing V is equivalent to minimizing fraction * box_area.
        # My cost function returns volume (integral of areas).
        # So I should return the volume directly.
        
    def cost_function_volume(positions_2d):
        x_base = positions_2d[:, 0]
        y_base = positions_2d[:, 1]
        
        c_vals = np.linspace(0.0, 1.0, 21)
        
        total_area = 0.0
        for c in c_vals:
            total_area += area_of_union(x_base, y_base, i_flat, j_flat, c)
            
        volume = total_area * (1.0 / 20.0)
        
        X0 = np.min(x_base)
        X1 = np.max(x_base + (i_flat + 1) / 8.0)
        Y0 = np.min(y_base)
        Y1 = np.max(y_base + (j_flat + 1) / 8.0)
        
        box_area = (X1 - X0) * (Y1 - Y0)
        
        if box_area <= 1e-9:
            return 1e10
            
        return volume

    # Optimization
    start_time = time.time()
    deadline = 630.0
    
    # Initial guess: Keich or random?
    # Let's try random restarts with Differential Evolution.
    
    best_pos = None
    best_vol = 1e10
    
    # Use a few random starts
    rng = np.random.default_rng(42)
    
    while time.time() - start_time < deadline - 30.0:
        # Random initial positions in a reasonable range
        # x, y in [-1, 2]
        init_pos = rng.uniform(-1.0, 2.0, (64, 2))
        
        # Flatten for DE
        init_flat = init_pos.flatten()
        
        # Bounds for DE
        bounds = [(-2.0, 3.0)] * 128
        
        # Run Differential Evolution
        try:
            res = differential_evolution(
                cost_function_volume, 
                init_flat, 
                bounds=bounds,
                maxiter=100, 
                tol=1e-8,
                polish=False, # Polish later
                seed=42,
                workers=-1 # Use all cores? Might be risky. Let's use 1 for safety.
            )
            
            if res.fun < best_vol:
                best_vol = res.fun
                best_pos = res.x.reshape(64, 2)
                
                # Polish with Nelder-Mead
                if time.time() - start_time < deadline - 10.0:
                    res2 = minimize(
                        cost_function_volume, 
                        best_pos.flatten(), 
                        method='Nelder-Mead',
                        options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10}
                    )
                    if res2.fun < best_vol:
                        best_vol = res2.fun
                        best_pos = res2.x.reshape(64, 2)
                        
        except Exception as e:
            print(e)
            
    if best_pos is None:
        # Fallback
        return np.zeros((64, 2))
        
    return best_pos

def kakeya_needle_3d8() -> np.ndarray:
    return compute_volume_2d(np.zeros((64,2))) # Placeholder, need to fix structure

# Wait, I nested the logic. Let's rewrite cleanly.

import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

def kakeya_needle_3d8() -> np.ndarray:
    start_time = time.time()
    deadline = 630.0
    
    N = 8
    i_idx = np.arange(N)
    j_idx = np.arange(N)
    i_grid, j_grid = np.meshgrid(i_idx, j_idx, indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    def area_of_union(x_base, y_base, c_val):
        Lx = x_base + c_val * i_flat / 8.0
        Rx = Lx + 1/8.0
        Ly = y_base + c_val * j_flat / 8.0
        Ry = Ly + 1/8.0
        
        events = []
        for k in range(64):
            events.append((Lx[k], 0, Ly[k], Ry[k]))
            events.append((Rx[k], 1, Ly[k], Ry[k]))
            
        events.sort(key=lambda x: x[0])
        
        active_intervals = []
        area = 0.0
        last_x = events[0][0]
        
        for x, typ, y1, y2 in events:
            if x > last_x:
                if active_intervals:
                    active_intervals.sort(key=lambda i: i[0])
                    merged = []
                    cur_l, cur_r = active_intervals[0]
                    for l, r in active_intervals[1:]:
                        if l <= cur_r:
                            if r > cur_r:
                                cur_r = r
                        else:
                            merged.append((cur_l, cur_r))
                            cur_l, cur_r = l, r
                    merged.append((cur_l, cur_r))
                    height = sum(r - l for l, r in merged)
                else:
                    height = 0.0
                    
                area += height * (x - last_x)
                last_x = x
                
            if typ == 0:
                active_intervals.append((y1, y2))
            else:
                for idx, (l, r) in enumerate(active_intervals):
                    if l == y1 and r == y2:
                        active_intervals.pop(idx)
                        break
                        
        return area

    def cost_function_volume(positions_2d):
        x_base = positions_2d[:, 0]
        y_base = positions_2d[:, 1]
        
        c_vals = np.linspace(0.0, 1.0, 21)
        
        total_area = 0.0
        for c in c_vals:
            total_area += area_of_union(x_base, y_base, c)
            
        volume = total_area * (1.0 / 20.0)
        
        X0 = np.min(x_base)
        X1 = np.max(x_base + (i_flat + 1) / 8.0)
        Y0 = np.min(y_base)
        Y1 = np.max(y_base + (j_flat + 1) / 8.0)
        
        box_area = (X1 - X0) * (Y1 - Y0)
        
        if box_area <= 1e-9:
            return 1e10
            
        return volume

    best_pos = None
    best_vol = 1e10
    
    rng = np.random.default_rng(42)
    
    # Initial Keich-like guess? Or just random.
    # Let's use a structured initial guess to help DE.
    # x_k = i/8 * 0.1, y_k = j/8 * 0.1 ?
    
    while time.time() - start_time < deadline - 30.0:
        init_pos = rng.uniform(-1.0, 2.0, (64, 2))
        init_flat = init_pos.flatten()
        
        bounds = [(-2.0, 3.0)] * 128
        
        try:
            res = differential_evolution(
                cost_function_volume, 
                init_flat, 
                bounds=bounds,
                maxiter=50, # Keep it fast
                tol=1e-8,
                polish=False,
                seed=42,
                workers=1
            )
            
            if res.fun < best_vol:
                best_vol = res.fun
                best_pos = res.x.reshape(64, 2)
                
                if time.time() - start_time < deadline - 10.0:
                    res2 = minimize(
                        cost_function_volume, 
                        best_pos.flatten(), 
                        method='Nelder-Mead',
                        options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10}
                    )
                    if res2.fun < best_vol:
                        best_vol = res2.fun
                        best_pos = res2.x.reshape(64, 2)
                        
        except Exception as e:
            pass

    if best_pos is None:
        return np.zeros((64, 2))
        
    return best_pos

if __name__ == "__main__":
    res = kakeya_needle_3d8()
    print(res.shape)
