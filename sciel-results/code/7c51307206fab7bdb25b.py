import numpy as np
import time
from scipy.optimize import minimize
from scipy.optimize import milp, LinearConstraint, Bounds
import itertools

def _construct_analytical_layout():
    """
    Construct a high-quality analytical layout for 10 squares in unit square.
    
    Strategy: Use a 3x3 grid of equal squares in the bottom-left 9/10 of the square,
    plus one strip square in the remaining space.
    
    Layout:
    - 3x3 grid of squares with side s in [0, 3s] x [0, 3s]
    - One square in the remaining L-shaped region
    
    For a 3x3 grid of side s, we need 3s <= 1, so s <= 1/3.
    The remaining region is [3s, 1] x [0, 1] union [0, 1] x [3s, 1].
    
    Let's try: 3x3 grid with s = 0.3 (so grid is 0.9 x 0.9)
    Remaining: right strip [0.9, 1] x [0, 1] (width 0.1) and top strip [0, 1] x [0.9, 1] (height 0.1)
    
    Better: Use asymmetric layout.
    Let me try: 4 squares in bottom row, 3 in middle, 3 in top.
    
    Actually, let's use the known good layout:
    - Bottom: 2 squares of side 0.45 (total width 0.9, leaves 0.1)
    - Middle: 3 squares of side 0.3 (total width 0.9)
    - Top: 3 squares of side 0.3 (total width 0.9)
    - Right strip: 2 squares stacked vertically in the 0.1-wide strip? No, too narrow.
    
    Let me think differently. Known result for 10 squares: sum ~3.0
    
    Try: 
    - 4 squares of side 0.25 in a 2x2 block (total 0.5x0.5)
    - 4 squares of side 0.25 in another 2x2 block
    - 2 squares of side 0.5
    
    Actually, let's use a simpler constructive approach:
    Divide unit square into regions and place squares greedily.
    
    Best known: Pack 10 squares with total side length ~3.0
    
    Layout:
    - Left column: 2 squares stacked, each 0.5 x 0.5 (occupies [0,0.5] x [0,1])
    - Right area [0.5, 1] x [0, 1]: 
      - Bottom: 3 squares of side 1/3 in [0.5, 1] x [0, 1/3]
      - Middle: 3 squares of side 1/3 in [0.5, 1] x [1/3, 2/3]  
      - Top: 2 squares of side 1/6 in [0.5, 1] x [2/3, 1]
    
    That's 2 + 3 + 3 + 2 = 10 squares.
    Sum = 2*0.5 + 6*(1/3) + 2*(1/6) = 1.0 + 2.0 + 0.333 = 3.333
    
    Let me verify:
    - Square 1: center (0.25, 0.25), side 0.5 -> [0, 0.5] x [0, 0.5]
    - Square 2: center (0.25, 0.75), side 0.5 -> [0, 0.5] x [0.5, 1.0]
    - Square 3: center (0.5+1/6, 1/6) = (0.667, 0.167), side 1/3 -> [0.5, 0.833] x [0, 0.333]
    - Square 4: center (0.5+1/6+1/3, 1/6) = (0.833, 0.167), side 1/3 -> [0.667, 1.0] x [0, 0.333]
    Wait, that's only 2 in the bottom right. Let me redo.
    
    Right area is [0.5, 1.0] x [0, 1.0], width 0.5, height 1.0.
    Divide into 3 horizontal strips of height 1/3 each:
    - Strip 1 [0.5, 1.0] x [0, 1/3]: width 0.5, can fit 1 square of side 1/3? No, 1/3 < 0.5, so can fit floor(0.5/(1/3)) = 1 square of side 1/3, leaving space.
    
    This is getting complex. Let me just use a well-known packing.
    
    Known optimal or near-optimal for 10 squares:
    According to literature, the best packing of 10 equal squares in a unit square has side ~0.24, sum ~2.4.
    But we want to MAXIMIZE sum of sides, so we want unequal squares.
    
    Let me try a different approach: Use the MILP result if available, otherwise use a good heuristic.
    """
    n = 10
    # Layout: 2 large squares on left, 8 smaller squares on right
    # Left: 2 squares of side 0.5, stacked vertically
    # Right: 4x2 grid of squares, each side 0.25
    centers = np.array([
        [0.25, 0.25],   # Large 1
        [0.25, 0.75],   # Large 2
        [0.625, 0.125], # Small 1
        [0.875, 0.125], # Small 2
        [0.625, 0.375], # Small 3
        [0.875, 0.375], # Small 4
        [0.625, 0.625], # Small 5
        [0.875, 0.625], # Small 6
        [0.625, 0.875], # Small 7
        [0.875, 0.875], # Small 8
    ])
    sides = np.array([
        0.5, 0.5,
        0.25, 0.25, 0.25, 0.25,
        0.25, 0.25, 0.25, 0.25
    ])
    return centers, sides

def _construct_better_layout():
    """
    Try to construct a layout with higher sum of sides.
    
    Idea: Use fewer, larger squares where possible.
    
    Layout attempt:
    - 1 square of side 0.5 in bottom-left
    - 1 square of side 0.5 in bottom-right  
    - 1 square of side 0.5 in top-left
    - Remaining top-right area [0.5, 1] x [0.5, 1]: 0.5 x 0.5
      - Can fit 4 squares of side 0.25
    - That's 7 squares, need 3 more.
    
    Better:
    - Bottom row: 2 squares of side 0.4 (total width 0.8, leaves 0.2)
    - Middle row: 3 squares of side 0.3 (total width 0.9)
    - Top row: 3 squares of side 0.3 (total width 0.9)
    - Right strip [0.8, 1] x [0, 1]: width 0.2, can fit 2 squares of side 0.2 stacked
    - That's 2+3+3+2 = 10 squares
    Sum = 2*0.4 + 6*0.3 + 2*0.2 = 0.8 + 1.8 + 0.4 = 3.0
    
    Let me verify this layout:
    - Square 1: center (0.2, 0.2), side 0.4 -> [0, 0.4] x [0, 0.4]
    - Square 2: center (0.6, 0.2), side 0.4 -> [0.4, 0.8] x [0, 0.4]
    - Square 3: center (0.15, 0.55), side 0.3 -> [0, 0.3] x [0.4, 0.7]
    - Square 4: center (0.45, 0.55), side 0.3 -> [0.3, 0.6] x [0.4, 0.7]
    - Square 5: center (0.75, 0.55), side 0.3 -> [0.6, 0.9] x [0.4, 0.7]
    - Square 6: center (0.15, 0.85), side 0.3 -> [0, 0.3] x [0.7, 1.0]
    - Square 7: center (0.45, 0.85), side 0.3 -> [0.3, 0.6] x [0.7, 1.0]
    - Square 8: center (0.75, 0.85), side 0.3 -> [0.6, 0.9] x [0.7, 1.0]
    - Square 9: center (0.9, 0.2), side 0.2 -> [0.8, 1.0] x [0.1, 0.3]
    - Square 10: center (0.9, 0.6), side 0.2 -> [0.8, 1.0] x [0.5, 0.7]
    
    Check overlaps:
    - Squares 1,2 in bottom: no overlap (touch at x=0.4)
    - Squares 3,4,5 in middle: no overlap (touch at x=0.3, x=0.6)
    - Squares 6,7,8 in top: no overlap (touch at x=0.3, x=0.6)
    - Square 9 in right strip: [0.8,1]x[0.1,0.3]
      - vs Square 2: [0.4,0.8]x[0,0.4] -> touch at x=0.8, no overlap
      - vs Square 5: [0.6,0.9]x[0.4,0.7] -> x ranges [0.8,0.9] overlap, y ranges [0.1,0.3] vs [0.4,0.7] no overlap
    - Square 10 in right strip: [0.8,1]x[0.5,0.7]
      - vs Square 5: [0.6,0.9]x[0.4,0.7] -> x overlap [0.8,0.9], y overlap [0.5,0.7] -> OVERLAP!
    
    Fix: Move square 10 to [0.8,1]x[0.7,0.9]
    - center (0.9, 0.8), side 0.2
    - vs Square 8: [0.6,0.9]x[0.7,1.0] -> x overlap [0.8,0.9], y overlap [0.7,0.9] -> OVERLAP!
    
    The right strip conflicts with the middle/top rows. Let me adjust.
    
    New layout:
    - Bottom: 2 squares of side 0.45 (total width 0.9)
    - Middle: 3 squares of side 0.3 (total width 0.9)  
    - Top: 3 squares of side 0.3 (total width 0.9)
    - Right: 2 squares of side 0.1 in the 0.1-wide strip
    Sum = 2*0.45 + 6*0.3 + 2*0.1 = 0.9 + 1.8 + 0.2 = 2.9
    
    Or:
    - Left column: 2 squares of side 0.5
    - Right area: 8 squares in a 4x2 grid, each side 0.25
    Sum = 2*0.5 + 8*0.25 = 1.0 + 2.0 = 3.0
    
    This is the same as the first layout. Let me stick with that and let local search improve it.
    """
    n = 10
    centers = np.array([
        [0.25, 0.25],
        [0.25, 0.75],
        [0.625, 0.125],
        [0.875, 0.125],
        [0.625, 0.375],
        [0.875, 0.375],
        [0.625, 0.625],
        [0.875, 0.625],
        [0.625, 0.875],
        [0.875, 0.875],
    ])
    sides = np.array([
        0.5, 0.5,
        0.25, 0.25, 0.25, 0.25,
        0.25, 0.25, 0.25, 0.25
    ])
    return centers, sides

def squares_in_square10() -> tuple:
    n = 10
    start_time = time.time()
    deadline = start_time + 595  # Safety margin before 630s kill
    
    # Try MILP approach first
    try:
        result = _solve_milp(n, deadline)
        if result is not None:
            return result
    except Exception:
        pass
    
    # Fallback to local search with analytical initial layout
    centers_init, sides_init = _construct_better_layout()

    # Helper: convert from (centers, sides) to flat vector
    def to_flat(centers, sides):
        x = np.zeros(3 * n)
        for i in range(n):
            x[3*i] = centers[i, 0]
            x[3*i+1] = centers[i, 1]
            x[3*i+2] = sides[i]
        return x

    # Helper: convert from flat vector to (centers, sides)
    def from_flat(x):
        cx = x[0::3]
        cy = x[1::3]
        s = np.maximum(x[2::3], 1e-8)
        centers = np.stack([cx, cy], axis=1)
        return centers, s

    # Penalty function: -sum(sides) + boundary_penalty + overlap_penalty
    def penalty_function(x):
        cx = x[0::3]
        cy = x[1::3]
        s = np.maximum(x[2::3], 1e-8)

        # Boundary violations
        pen = 0.0
        for i in range(n):
            lo_x = -1e-7
            hi_x = 1 + 1e-7
            lo_y = -1e-7
            hi_y = 1 + 1e-7
            if cx[i] - s[i]/2 < lo_x:
                pen += 1000 * (lo_x - (cx[i] - s[i]/2))
            if cx[i] + s[i]/2 > hi_x:
                pen += 1000 * ((cx[i] + s[i]/2) - hi_x)
            if cy[i] - s[i]/2 < lo_y:
                pen += 1000 * (lo_y - (cy[i] - s[i]/2))
            if cy[i] + s[i]/2 > hi_y:
                pen += 1000 * ((cy[i] + s[i]/2) - hi_y)

        # Overlap penalties
        for i in range(n):
            for j in range(i+1, n):
                dx = abs(cx[i] - cx[j])
                dy = abs(cy[i] - cy[j])
                required = (s[i] + s[j]) / 2
                if dx < required and dy < required:
                    # Overlap: penalize the product of penetrations
                    pen += 100 * (required - dx) * (required - dy)

        return -(np.sum(s)) + pen

    # Clamp and validate
    def clamp_and_fix(centers, sides):
        s = np.maximum(sides, 1e-8)
        cx = centers[:, 0].copy()
        cy = centers[:, 1].copy()
        for i in range(n):
            # Clamp to boundary
            cx[i] = np.clip(cx[i], s[i]/2, 1 - s[i]/2)
            cy[i] = np.clip(cy[i], s[i]/2, 1 - s[i]/2)
        centers_fixed = np.stack([cx, cy], axis=1)

        # Check validity
        tol = 1e-7
        for i in range(n):
            if centers_fixed[i, 0] - s[i]/2 < -tol or centers_fixed[i, 0] + s[i]/2 > 1 + tol:
                return False, centers_init, sides_init, 0.0
            if centers_fixed[i, 1] - s[i]/2 < -tol or centers_fixed[i, 1] + s[i]/2 > 1 + tol:
                return False, centers_init, sides_init, 0.0
        for i in range(n):
            for j in range(i+1, n):
                dx = abs(centers_fixed[i, 0] - centers_fixed[j, 0])
                dy = abs(centers_fixed[i, 1] - centers_fixed[j, 1])
                t = (s[i] + s[j])/2 - tol
                if dx < t and dy < t:
                    return False, centers_init, sides_init, 0.0
        score = np.sum(s)
        return True, centers_fixed, s, score

    # Local search using Nelder-Mead with time check
    def local_search(x_flat, max_iter=2000):
        best_x = x_flat.copy()
        best_f = penalty_function(x_flat)
        for _ in range(max_iter):
            if time.time() > deadline - 5:
                break
            res = minimize(penalty_function, best_x, method='Nelder-Mead',
                          options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-6})
            f_new = res.fun
            if f_new < best_f:
                best_f = f_new
                best_x = res.x
            else:
                # No improvement, try a small random perturbation
                if np.random.rand() < 0.3:
                    best_x = best_x + np.random.normal(0, 0.01, 3*n)
                else:
                    break
        return best_x, -best_f if best_f < 0 else best_f

    # Multi-start local search with focus on the analytical layout
    x0_flat = to_flat(centers_init, sides_init)
    best_centers, best_sides = centers_init.copy(), sides_init.copy()
    valid_init, c_init_fixed, s_init_fixed, score_init = clamp_and_fix(centers_init, sides_init)
    if valid_init:
        best_score = score_init
        best_centers = c_init_fixed
        best_sides = s_init_fixed
    else:
        best_score = 0.0

    # Try different initial layouts, prioritizing the analytical one
    layouts = [
        (centers_init, sides_init),  # Analytical layout (highest priority)
        # Symmetric 4-3-3 layout
        (np.array([[0.25,0.25],[0.75,0.25],[0.2,0.65],[0.5,0.65],[0.8,0.65],[0.15,0.9],[0.35,0.9],[0.65,0.9],[0.85,0.9],[0.5,0.9]]),
         np.array([0.5,0.5,0.3,0.3,0.3,0.2,0.2,0.2,0.2,0.2])),
        # Another layout: 3 large + 7 small
        (np.array([[0.167,0.167],[0.5,0.167],[0.833,0.167],[0.167,0.5],[0.5,0.5],[0.833,0.5],[0.333,0.833],[0.667,0.833],[0.167,0.833],[0.833,0.833]]),
         np.array([0.333,0.333,0.333,0.333,0.333,0.333,0.333,0.333,0.333,0.333])),
    ]

    for c_init, s_init in layouts:
        if time.time() > deadline - 10:
            break
        # Clamp initial to be valid
        c_fix = c_init.copy()
        s_fix = np.maximum(s_init, 1e-8).copy()
        for i in range(n):
            c_fix[i, 0] = np.clip(c_fix[i, 0], s_fix[i]/2, 1 - s_fix[i]/2)
            c_fix[i, 1] = np.clip(c_fix[i, 1], s_fix[i]/2, 1 - s_fix[i]/2)
        x_flat = to_flat(c_fix, s_fix)
        
        # Use more iterations for the first (analytical) layout
        max_iter = 3000 if c_init is centers_init else 1000
        x_res, f_res = local_search(x_flat, max_iter=max_iter)
        centers_res, sides_res = from_flat(x_res)
        valid, c_final, s_final, score_final = clamp_and_fix(centers_res, sides_res)
        if valid and score_final > best_score:
            best_score = score_final
            best_centers = c_final
            best_sides = s_final

    # Final refinement with high-precision polish
    if time.time() < deadline - 20:
        x_flat = to_flat(best_centers, best_sides)
        
        # Use tighter tolerances for the final polish
        def polish_penalty(x):
            cx = x[0::3]
            cy = x[1::3]
            s = np.maximum(x[2::3], 1e-8)
            
            pen = 0.0
            for i in range(n):
                if cx[i] - s[i]/2 < -1e-7:
                    pen += 10000 * (-1e-7 - (cx[i] - s[i]/2))
                if cx[i] + s[i]/2 > 1 + 1e-7:
                    pen += 10000 * ((cx[i] + s[i]/2) - (1 + 1e-7))
                if cy[i] - s[i]/2 < -1e-7:
                    pen += 10000 * (-1e-7 - (cy[i] - s[i]/2))
                if cy[i] + s[i]/2 > 1 + 1e-7:
                    pen += 10000 * ((cy[i] + s[i]/2) - (1 + 1e-7))
            
            for i in range(n):
                for j in range(i+1, n):
                    dx = abs(cx[i] - cx[j])
                    dy = abs(cy[i] - cy[j])
                    required = (s[i] + s[j]) / 2 - 1e-7
                    if dx < required and dy < required:
                        pen += 10000 * (required - dx) * (required - dy)
            
            return -(np.sum(s)) + pen
        
        res = minimize(polish_penalty, x_flat, method='L-BFGS-B',
                      options={'maxiter': 5000, 'ftol': 1e-12, 'gtol': 1e-10})
        if res.fun < penalty_function(x_flat):
            centers_res, sides_res = from_flat(res.x)
            valid, c_final, s_final, score_final = clamp_and_fix(centers_res, sides_res)
            if valid and score_final > best_score:
                best_score = score_final
                best_centers = c_final
                best_sides = s_final

    # Final validation
    valid, best_centers, best_sides, final_score = clamp_and_fix(best_centers, best_sides)
    if not valid:
        return centers_init, sides_init

    return best_centers, best_sides

def _solve_milp(n, deadline):
    """
    Solve the square packing problem using MILP.
    
    Variables:
    - x_i, y_i: center coordinates (continuous)
    - s_i: side length (continuous)
    - For each pair (i,j) and each of 4 directions (left, right, above, below):
      binary variable indicating that square i is in that direction relative to square j
    
    Objective: maximize sum(s_i)
    
    Constraints:
    - Boundary: 0 <= x_i - s_i/2, x_i + s_i/2 <= 1, same for y
    - For each pair (i,j), at least one of the 4 separation constraints must hold
    """
    num_pairs = n * (n - 1) // 2
    # Variables: n x, n y, n s, then 4 * num_pairs binaries
    num_binaries = 4 * num_pairs
    num_vars = 3 * n + num_binaries
    
    # Objective: maximize sum(s_i) -> minimize -sum(s_i)
    c = np.zeros(num_vars)
    for i in range(n):
        c[2 * n + i] = -1.0  # s_i variable index
    
    # Bounds
    lb = np.zeros(num_vars)
    ub = np.zeros(num_vars)
    for i in range(n):
        lb[i] = 0.0
        ub[i] = 1.0  # x_i
        lb[n + i] = 0.0
        ub[n + i] = 1.0  # y_i
        lb[2 * n + i] = 1e-8
        ub[2 * n + i] = 1.0  # s_i
    # Binary variables
    for k in range(num_binaries):
        idx = 3 * n + k
        lb[idx] = 0
        ub[idx] = 1
    
    # Linear constraints
    A_rows = []
    lb_rows = []
    ub_rows = []
    
    # Boundary constraints:
    # x_i - s_i/2 >= 0  =>  x_i - 0.5*s_i >= 0
    # x_i + s_i/2 <= 1  =>  x_i + 0.5*s_i <= 1
    # Same for y
    for i in range(n):
        # x_i - 0.5*s_i >= 0
        row = np.zeros(num_vars)
        row[i] = 1.0
        row[2 * n + i] = -0.5
        A_rows.append(row)
        lb_rows.append(0.0)
        ub_rows.append(np.inf)
        
        # x_i + 0.5*s_i <= 1
        row = np.zeros(num_vars)
        row[i] = 1.0
        row[2 * n + i] = 0.5
        A_rows.append(row)
        lb_rows.append(-np.inf)
        ub_rows.append(1.0)
        
        # y_i - 0.5*s_i >= 0
        row = np.zeros(num_vars)
        row[n + i] = 1.0
        row[2 * n + i] = -0.5
        A_rows.append(row)
        lb_rows.append(0.0)
        ub_rows.append(np.inf)
        
        # y_i + 0.5*s_i <= 1
        row = np.zeros(num_vars)
        row[n + i] = 1.0
        row[2 * n + i] = 0.5
        A_rows.append(row)
        lb_rows.append(-np.inf)
        ub_rows.append(1.0)
    
    # Separation constraints for each pair (i,j)
    # For squares i and j to not overlap, at least one of:
    #   x_i + s_i/2 <= x_j - s_j/2  (i is left of j)
    #   x_j + s_j/2 <= x_i - s_i/2  (i is right of j)
    #   y_i + s_i/2 <= y_j - s_j/2  (i is below j)
    #   y_j + s_j/2 <= y_i - s_i/2  (i is above j)
    # 
    # Using big-M formulation with binary variables:
    # For direction d (0=left, 1=right, 2=below, 3=above):
    #   constraint_d: (separation_expr) <= M * (1 - b_d)
    #   sum(b_d) >= 1
    #
    # M = 1 (since all values are in [0,1])
    
    M = 1.0
    pair_idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            # Direction 0: i is left of j: x_i + s_i/2 <= x_j - s_j/2
            # => x_i + 0.5*s_i - x_j + 0.5*s_j <= 0
            # With binary b0: x_i + 0.5*s_i - x_j + 0.5*s_j <= M * (1 - b0)
            row = np.zeros(num_vars)
            row[i] = 1.0
            row[2 * n + i] = 0.5
            row[j] = -1.0
            row[2 * n + j] = 0.5
            row[3 * n + 4 * pair_idx + 0] = -M
            A_rows.append(row)
            lb_rows.append(-np.inf)
            ub_rows.append(M)
            
            # Direction 1: i is right of j: x_j + s_j/2 <= x_i - s_i/2
            # => x_j + 0.5*s_j - x_i + 0.5*s_i <= 0
            row = np.zeros(num_vars)
            row[j] = 1.0
            row[2 * n + j] = 0.5
            row[i] = -1.0
            row[2 * n + i] = 0.5
            row[3 * n + 4 * pair_idx + 1] = -M
            A_rows.append(row)
            lb_rows.append(-np.inf)
            ub_rows.append(M)
            
            # Direction 2: i is below j: y_i + s_i/2 <= y_j - s_j/2
            # => y_i + 0.5*s_i - y_j + 0.5*s_j <= 0
            row = np.zeros(num_vars)
            row[n + i] = 1.0
            row[2 * n + i] = 0.5
            row[n + j] = -1.0
            row[2 * n + j] = 0.5
            row[3 * n + 4 * pair_idx + 2] = -M
            A_rows.append(row)
            lb_rows.append(-np.inf)
            ub_rows.append(M)
            
            # Direction 3: i is above j: y_j + s_j/2 <= y_i - s_i/2
            # => y_j + 0.5*s_j - y_i + 0.5*s_i <= 0
            row = np.zeros(num_vars)
            row[n + j] = 1.0
            row[2 * n + j] = 0.5
            row[n + i] = -1.0
            row[2 * n + i] = 0.5
            row[3 * n + 4 * pair_idx + 3] = -M
            A_rows.append(row)
            lb_rows.append(-np.inf)
            ub_rows.append(M)
            
            # Sum of binaries >= 1
            row = np.zeros(num_vars)
            row[3 * n + 4 * pair_idx + 0] = 1.0
            row[3 * n + 4 * pair_idx + 1] = 1.0
            row[3 * n + 4 * pair_idx + 2] = 1.0
            row[3 * n + 4 * pair_idx + 3] = 1.0
            A_rows.append(row)
            lb_rows.append(1.0)
            ub_rows.append(np.inf)
            
            pair_idx += 1
    
    A = np.array(A_rows)
    lb_arr = np.array(lb_rows)
    ub_arr = np.array(ub_rows)
    
    constraint = LinearConstraint(A, lb_arr, ub_arr)
    bounds = Bounds(lb, ub)
    
    # Integer variables: the binary separation variables
    integrality = np.zeros(num_vars)
    for k in range(num_binaries):
        integrality[3 * n + k] = 1
    
    result = milp(
        c=c,
        constraints=constraint,
        integrality=integrality,
        bounds=bounds,
        options={"time_limit": int(deadline - time.time() - 5)}
    )
    
    if result.status != 0 and result.status != 1:
        return None
    
    x_sol = result.x
    centers = np.zeros((n, 2))
    sides = np.zeros(n)
    for i in range(n):
        centers[i, 0] = x_sol[i]
        centers[i, 1] = x_sol[n + i]
        sides[i] = x_sol[2 * n + i]
    
    # Clamp to ensure validity
    tol = 1e-7
    for i in range(n):
        centers[i, 0] = np.clip(centers[i, 0], sides[i]/2, 1 - sides[i]/2)
        centers[i, 1] = np.clip(centers[i, 1], sides[i]/2, 1 - sides[i]/2)
    
    # Validate
    for i in range(n):
        if centers[i, 0] - sides[i]/2 < -tol or centers[i, 0] + sides[i]/2 > 1 + tol:
            return None
        if centers[i, 1] - sides[i]/2 < -tol or centers[i, 1] + sides[i]/2 > 1 + tol:
            return None
    for i in range(n):
        for j in range(i + 1, n):
            dx = abs(centers[i, 0] - centers[j, 0])
            dy = abs(centers[i, 1] - centers[j, 1])
            t = (sides[i] + sides[j]) / 2 - tol
            if dx < t and dy < t:
                return None
    
    score = np.sum(sides)
    if score <= 0:
        return None
    
    return centers, sides
