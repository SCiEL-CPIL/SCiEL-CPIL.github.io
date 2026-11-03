import numpy as np
import time
from scipy.optimize import linprog, minimize

N = 21

def _is_valid(result, tol=1e-7):
    """Check if the packing is valid: disjoint interiors, within bounds, perimeter <= 4."""
    if result is None or len(result) != N:
        return False
    centers = result[:, :2]
    radii = result[:, 2]
    w = np.max(centers[:, 0]) + np.max(radii) - (np.min(centers[:, 0]) - np.min(radii))
    h = np.max(centers[:, 1]) + np.max(radii) - (np.min(centers[:, 1]) - np.min(radii))
    # Actually, the bounding box is determined by min/max of (center +/- radius)
    x_min = np.min(centers[:, 0] - radii)
    x_max = np.max(centers[:, 0] + radii)
    y_min = np.min(centers[:, 1] - radii)
    y_max = np.max(centers[:, 1] + radii)
    w = x_max - x_min
    h = y_max - y_min
    if w + h > 4.0 + tol:
        return False
    
    # Check overlaps
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(centers[i] - centers[j])
            if d < radii[i] + radii[j] - tol:
                return False
    return True

def _solve_qp_subproblem(centers, radii, w, h, time_limit):
    """
    Solve a QP subproblem to adjust centers and radii to maximize sum(radii)
    subject to linearized constraints.
    Variables: [dx_1, dy_1, dr_1, ..., dx_N, dy_N, dr_N]
    Objective: maximize sum(dr_i) -> minimize -sum(dr_i)
    Constraints:
    1. dx_i <= w - x_i - r_i - dr_i  (Right wall) -> dx_i + dr_i <= w - x_i - r_i
       -dx_i <= x_i - r_i - dr_i     (Left wall)  -> -dx_i + dr_i <= x_i - r_i
       dy_i + dr_i <= h - y_i - r_i
       -dy_i + dr_i <= y_i - r_i
    2. For each pair (i,j):
       (x_i - x_j + dx_i - dx_j)^2 + (y_i - y_j + dy_i - dy_j)^2 >= (r_i + r_j + dr_i + dr_j)^2
       Linearize around current state:
       2(x_i - x_j)(dx_i - dx_j) + 2(y_i - y_j)(dy_i - dy_j) >= 2(r_i + r_j)(dr_i + dr_j)
       + (dx_i - dx_j)^2 + (dy_i - dy_j)^2 - (dr_i + dr_j)^2 ... ignore quadratic terms in linearization
       So: 2(x_i - x_j)(dx_i - dx_j) + 2(y_i - y_j)(dy_i - dy_j) - 2(r_i + r_j)(dr_i + dr_j) >= 0
       Or: -2(x_i - x_j)(dx_i - dx_j) - 2(y_i - y_j)(dy_i - dy_j) + 2(r_i + r_j)(dr_i + dr_j) <= 0
    """
    n = N
    # Variables: 3*n
    # 0..n-1: dx, n..2n-1: dy, 2n..3n-1: dr
    
    c = np.zeros(3*n)
    c[2*n:] = -1.0 # Minimize -sum(dr)
    
    A_ub = []
    b_ub = []
    
    # Wall constraints
    # Right: dx_i + dr_i <= w - x_i - r_i
    for i in range(n):
        row = np.zeros(3*n)
        row[i] = 1.0
        row[2*n + i] = 1.0
        A_ub.append(row)
        b_ub.append(w - centers[i, 0] - radii[i])
        
        # Left: -dx_i + dr_i <= x_i - r_i
        row = np.zeros(3*n)
        row[i] = -1.0
        row[2*n + i] = 1.0
        A_ub.append(row)
        b_ub.append(centers[i, 0] - radii[i])
        
        # Top: dy_i + dr_i <= h - y_i - r_i
        row = np.zeros(3*n)
        row[n + i] = 1.0
        row[2*n + i] = 1.0
        A_ub.append(row)
        b_ub.append(h - centers[i, 1] - radii[i])
        
        # Bottom: -dy_i + dr_i <= y_i - r_i
        row = np.zeros(3*n)
        row[n + i] = -1.0
        row[2*n + i] = 1.0
        A_ub.append(row)
        b_ub.append(centers[i, 1] - radii[i])
        
    # Pair constraints
    for i in range(n):
        for j in range(i + 1, n):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            r_sum = radii[i] + radii[j]
            
            # -2*dx*(dx_i - dx_j) - 2*dy*(dy_i - dy_j) + 2*r_sum*(dr_i + dr_j) <= 0
            row = np.zeros(3*n)
            row[i] = -2 * dx
            row[j] = 2 * dx
            row[n + i] = -2 * dy
            row[n + j] = 2 * dy
            row[2*n + i] = 2 * r_sum
            row[2*n + j] = 2 * r_sum
            A_ub.append(row)
            b_ub.append(0.0)
            
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    # Bounds: dr_i >= -radii[i] (can't go negative), dx, dy unbounded? 
    # Let's bound dx, dy to small values to keep local.
    bounds = []
    for i in range(n):
        bounds.append((None, None)) # dx
        bounds.append((None, None)) # dy
        bounds.append((-radii[i], None)) # dr
        
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if res.success:
        new_centers = centers + np.column_stack([res.x[:n], res.x[n:2*n]])
        new_radii = radii + res.x[2*n:]
        # Clip radii to be non-negative
        new_radii = np.maximum(new_radii, 0)
        return new_centers, new_radii
    else:
        return centers, radii

def _refine(centers, radii, w, h, time_limit):
    """
    Perform successive convex relaxation.
    """
    start_time = time.time()
    best_score = np.sum(radii)
    best_centers = centers.copy()
    best_radii = radii.copy()
    
    for it in range(100):
        if time.time() - start_time > time_limit:
            break
            
        new_centers, new_radii = _solve_qp_subproblem(centers, radii, w, h, time_limit)
        
        # Check validity of the new solution
        if _is_valid(np.column_stack([new_centers, new_radii])):
            score = np.sum(new_radii)
            if score > best_score:
                best_score = score
                best_centers = new_centers.copy()
                best_radii = new_radii.copy()
            centers = new_centers
            radii = new_radii
        else:
            # If invalid, we might have stepped too far.
            # Try a smaller step? Or break?
            # For now, break to be safe, or try to repair.
            break
            
    return np.column_stack([best_centers, best_radii]), best_score

def _pack_for_aspect(w, h, time_limit, seed=0):
    """
    Pack circles in a w x h rectangle.
    """
    np.random.seed(seed)
    
    # Initial guess: Grid
    # 21 circles. 5x5 grid = 25, so 5x4 + 1? Or 4x5 + 1?
    # Let's try 5 cols, 4 rows = 20, plus 1 in the middle.
    # Or just random init.
    
    # Random init
    centers = np.random.uniform(0.2, w - 0.2, (N, 2))
    centers[:, 0] = np.clip(centers[:, 0], 0.1, w - 0.1)
    centers[:, 1] = np.clip(centers[:, 1], 0.1, h - 0.1)
    
    # Initial radii: small
    radii = np.full(N, 0.05)
    
    # Run refinement
    result, score = _refine(centers, radii, w, h, time_limit)
    
    # Validate final result
    if _is_valid(result):
        return result, score
    else:
        # Fallback: return the small radius packing if valid, else None
        init_result = np.column_stack([centers, radii])
        if _is_valid(init_result):
            return init_result, np.sum(radii)
        return None, 0.0

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    best_result = None
    best_score = 0.0
    
    # Aspect ratios to try
    # Perimeter 4 => w + h = 2
    # Try a range of aspect ratios
    aspect_ratios = [1.0, 0.9, 1.1, 0.8, 1.2, 0.85, 1.15, 0.75, 1.25]
    
    for w in aspect_ratios:
        if time.time() > deadline:
            break
            
        h = 2.0 - w
        
        time_remaining = deadline - time.time()
        if time_remaining < 5:
            break
            
        # Allocate time for this aspect ratio
        # Try a few seeds for good aspect ratios
        num_seeds = 2 if w in [1.0, 0.9, 1.1] else 1
        time_per_seed = (time_remaining - 2) / num_seeds
        
        for s in range(num_seeds):
            if time.time() > deadline:
                break
                
            seed = 42 + s + int(w * 100)
            try:
                result, score = _pack_for_aspect(w, h, time_per_seed, seed)
                if result is not None and score > best_score:
                    best_score = score
                    best_result = result
            except Exception:
                pass

    if best_result is None:
        # Fallback: Simple grid
        w, h = 1.0, 1.0
        positions = []
        radii = []
        cols = 5
        rows = 5
        count = 0
        for i in range(cols):
            for j in range(rows):
                if count >= N:
                    break
                pos = [(i + 0.5) * w / cols, (j + 0.5) * h / rows]
                r = 0.5 * min(w/cols, h/rows) * 0.9
                positions.append(pos)
                radii.append(r)
                count += 1
        positions = np.array(positions)
        radii = np.array(radii)
        best_result = np.column_stack([positions, radii])
        
    return best_result
