# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, linprog
from itertools import combinations

def _is_valid(centers, sides, tol=1e-7):
    n = len(sides)
    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
        return False
    if np.any(sides <= 0):
        return False
    for i in range(n):
        cx, cy = centers[i]
        s = sides[i]
        if cx - s / 2 < -tol or cx + s / 2 > 1 + tol:
            return False
        if cy - s / 2 < -tol or cy + s / 2 > 1 + tol:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            dx = abs(centers[i, 0] - centers[j, 0])
            dy = abs(centers[i, 1] - centers[j, 1])
            t = (sides[i] + sides[j]) / 2.0 - tol
            if t > 0:
                if dx < t and dy < t:
                    return False
    return True

def _objective(x, n=10):
    """
    Objective function for optimization.
    x = [cx_0..cx_{n-1}, cy_0..cy_{n-1}, s_0..s_{n-1}]
    Returns: -sum(s_i) + penalty_terms
    """
    centers = x[:2*n].reshape(n, 2)
    sides = x[2*n:]
    
    # Negative sum of sides (we minimize)
    obj = -np.sum(sides)
    
    # Boundary penalties
    for i in range(n):
        cx, cy = centers[i]
        s = sides[i]
        # Left boundary
        if cx - s/2 < 0:
            obj += 1e6 * (cx - s/2)**2
        # Right boundary
        if cx + s/2 > 1:
            obj += 1e6 * (cx + s/2 - 1)**2
        # Bottom boundary
        if cy - s/2 < 0:
            obj += 1e6 * (cy - s/2)**2
        # Top boundary
        if cy + s/2 > 1:
            obj += 1e6 * (cy + s/2 - 1)**2
    
    # Overlap penalties
    for i in range(n):
        for j in range(i+1, n):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            t = (sides[i] + sides[j]) / 2.0
            # If overlapping in both x and y
            if abs(dx) < t and abs(dy) < t:
                # Penalty proportional to overlap
                overlap_x = t - abs(dx)
                overlap_y = t - abs(dy)
                obj += 1e6 * (overlap_x * overlap_y)
    
    return obj

def _lp_pack_fixed_sides(sides, orientations):
    """
    Given fixed side lengths and orientation choices, solve LP for centers.
    orientations: dict mapping (i,j) -> 0,1,2,3
    Returns (centers, success)
    """
    n = len(sides)
    c_obj = np.zeros(2*n)
    
    A_ub = []
    b_ub = []
    
    # Boundary constraints
    for i in range(n):
        # -cx_i <= -s_i/2
        row = np.zeros(2*n)
        row[i] = -1.0
        A_ub.append(row)
        b_ub.append(-sides[i]/2.0)
        
        # cx_i <= 1 - s_i/2
        row = np.zeros(2*n)
        row[i] = 1.0
        A_ub.append(row)
        b_ub.append(1.0 - sides[i]/2.0)
        
        # -cy_i <= -s_i/2
        row = np.zeros(2*n)
        row[n+i] = -1.0
        A_ub.append(row)
        b_ub.append(-sides[i]/2.0)
        
        # cy_i <= 1 - s_i/2
        row = np.zeros(2*n)
        row[n+i] = 1.0
        A_ub.append(row)
        b_ub.append(1.0 - sides[i]/2.0)
    
    # Disjunctive constraints
    for i in range(n):
        for j in range(i+1, n):
            o = orientations.get((i,j), 0)
            sep = (sides[i] + sides[j]) / 2.0
            row = np.zeros(2*n)
            if o == 0:  # i left of j
                row[i] = 1.0
                row[j] = -1.0
                A_ub.append(row)
                b_ub.append(-sep)
            elif o == 1:  # i right of j
                row[i] = -1.0
                row[j] = 1.0
                A_ub.append(row)
                b_ub.append(-sep)
            elif o == 2:  # i below j
                row[n+i] = 1.0
                row[n+j] = -1.0
                A_ub.append(row)
                b_ub.append(-sep)
            else:  # i above j
                row[n+i] = -1.0
                row[n+j] = 1.0
                A_ub.append(row)
                b_ub.append(-sep)
    
    if not A_ub:
        return np.zeros((n, 2)), True
    
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub,
                  bounds=[(0, 1)] * (2*n), method='highs')
    if res.status == 0:
        centers = res.x.reshape(n, 2)
        return centers, True
    return None, False

def _local_search(centers, sides, max_iters=500, deadline=None):
    """
    Local search: try moving individual squares to improve sum of sides.
    Uses gradient-free approach with random perturbations.
    """
    n = len(sides)
    best_score = np.sum(sides)
    best_c = centers.copy()
    best_s = sides.copy()
    
    rng = np.random.RandomState(42)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Randomly pick a square to modify
        i = rng.randint(n)
        
        # Try changing side length
        s_new = best_s[i] * (1 + rng.uniform(-0.05, 0.1))
        s_new = max(0.01, min(0.5, s_new))
        
        # Try adjusting position
        cx_new = best_c[i, 0] + rng.uniform(-0.05, 0.05)
        cy_new = best_c[i, 1] + rng.uniform(-0.05, 0.05)
        
        # Clamp to valid range
        cx_new = max(s_new/2, min(1 - s_new/2, cx_new))
        cy_new = max(s_new/2, min(1 - s_new/2, cy_new))
        
        # Create new configuration
        c_new = best_c.copy()
        s_new_arr = best_s.copy()
        c_new[i] = [cx_new, cy_new]
        s_new_arr[i] = s_new
        
        # Check validity
        if _is_valid(c_new, s_new_arr):
            score = np.sum(s_new_arr)
            if score > best_score:
                best_score = score
                best_c = c_new
                best_s = s_new_arr
    
    return best_c, best_s

def _coordinate_descent(centers, sides, max_iters=2000, deadline=None):
    """
    Coordinate descent: optimize one variable at a time.
    For each square, try to increase its side length while adjusting
    positions to maintain non-overlap. Uses smooth penalty for violations.
    """
    n = len(sides)
    best_c = centers.copy()
    best_s = sides.copy()
    best_score = np.sum(best_s)
    
    def smooth_penalty(c, s):
        """Smooth penalty for boundary and overlap violations."""
        pen = 0.0
        for i in range(n):
            cx, cy = c[i]
            si = s[i]
            # Boundary penalties (smooth)
            if cx - si/2 < 0:
                pen += 1000.0 * (cx - si/2)**2
            if cx + si/2 > 1:
                pen += 1000.0 * (cx + si/2 - 1)**2
            if cy - si/2 < 0:
                pen += 1000.0 * (cy - si/2)**2
            if cy + si/2 > 1:
                pen += 1000.0 * (cy + si/2 - 1)**2
        
        # Smooth overlap penalty
        for i in range(n):
            for j in range(i+1, n):
                dx = c[i, 0] - c[j, 0]
                dy = c[i, 1] - c[j, 1]
                t = (s[i] + s[j]) / 2.0
                # Smooth approximation of max(0, t-|dx|) * max(0, t-|dy|)
                ox = max(0, t - abs(dx))
                oy = max(0, t - abs(dy))
                # Smooth: use soft version
                ox_soft = ox + 1e-8 * np.exp(-ox/1e-4)
                oy_soft = oy + 1e-8 * np.exp(-oy/1e-4)
                pen += 1000.0 * ox_soft * oy_soft
        
        return pen
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        improved = False
        
        for i in range(n):
            # Try to increase side length of square i
            for delta_s in [0.001, 0.005, 0.01, 0.02, 0.05]:
                s_try = best_s.copy()
                s_try[i] = min(0.5, best_s[i] + delta_s)
                
                # Try to adjust position of square i to avoid overlaps
                c_try = best_c.copy()
                found_valid = False
                
                # Try small position adjustments
                for dcx in [-0.02, -0.01, 0, 0.01, 0.02]:
                    for dcy in [-0.02, -0.01, 0, 0.01, 0.02]:
                        cx_new = best_c[i, 0] + dcx
                        cy_new = best_c[i, 1] + dcy
                        
                        # Clamp
                        si = s_try[i]
                        cx_new = max(si/2, min(1 - si/2, cx_new))
                        cy_new = max(si/2, min(1 - si/2, cy_new))
                        
                        c_try[i] = [cx_new, cy_new]
                        
                        if _is_valid(c_try, s_try):
                            score = np.sum(s_try)
                            if score > best_score:
                                best_score = score
                                best_c = c_try.copy()
                                best_s = s_try.copy()
                                improved = True
                            found_valid = True
                            break
                    if found_valid:
                        break
                
                if found_valid:
                    break
        
        if not improved and it > 100:
            # No improvement for a while, try smaller steps
            pass
    
    return best_c, best_s

def _nelder_mead_polish(centers, sides, max_iter=500, deadline=None):
    """
    Nelder-Mead polish: use smooth penalty objective to refine the solution.
    """
    n = len(sides)
    
    def objective(x):
        c = x[:2*n].reshape(n, 2)
        s = x[2*n:]
        
        # Ensure sides are positive
        s = np.maximum(s, 0.001)
        
        obj = -np.sum(s)
        
        # Smooth boundary penalties
        for i in range(n):
            cx, cy = c[i]
            si = s[i]
            if cx - si/2 < 0:
                obj += 500.0 * (cx - si/2)**2
            if cx + si/2 > 1:
                obj += 500.0 * (cx + si/2 - 1)**2
            if cy - si/2 < 0:
                obj += 500.0 * (cy - si/2)**2
            if cy + si/2 > 1:
                obj += 500.0 * (cy + si/2 - 1)**2
        
        # Smooth overlap penalties
        for i in range(n):
            for j in range(i+1, n):
                dx = c[i, 0] - c[j, 0]
                dy = c[i, 1] - c[j, 1]
                t = (s[i] + s[j]) / 2.0
                ox = max(0.0, t - abs(dx))
                oy = max(0.0, t - abs(dy))
                obj += 500.0 * ox * oy
        
        return obj
    
    x0 = np.concatenate([centers.flatten(), sides])
    
    res = minimize(objective, x0, method='Nelder-Mead',
                   options={'maxiter': max_iter, 'xatol': 1e-10, 'fatol': 1e-10})
    
    if res.success:
        c_opt = res.x[:2*n].reshape(n, 2)
        s_opt = res.x[2*n:]
        
        # Clamp to valid range
        s_opt = np.clip(s_opt, 0.01, 0.5)
        for i in range(n):
            si = s_opt[i]
            c_opt[i, 0] = max(si/2, min(1 - si/2, c_opt[i, 0]))
            c_opt[i, 1] = max(si/2, min(1 - si/2, c_opt[i, 1]))
        
        if _is_valid(c_opt, s_opt):
            return c_opt, s_opt
    
    return centers, sides

def squares_in_square10() -> tuple:
    n = 10
    start_time = time.time()
    deadline = start_time + 580  # Safety margin
    
    best_centers = None
    best_sides = None
    best_score = -np.inf
    
    def update_best(centers, sides):
        nonlocal best_centers, best_sides, best_score
        if _is_valid(centers, sides):
            score = np.sum(sides)
            if score > best_score:
                best_score = score
                best_centers = centers.copy()
                best_sides = sides.copy()
    
    # Fallback: 2x5 grid
    fallback_c = np.zeros((n, 2))
    fallback_s = np.zeros(n)
    for i in range(n):
        r = i // 5
        c = i % 5
        fallback_c[i] = [(c + 0.5) * 0.2, (r + 0.5) * 0.5]
        fallback_s[i] = 0.2
    update_best(fallback_c, fallback_s)
    
    # Strategy 1: Try different grid-like initializations with varying sizes
    # Then use local search to improve
    
    # Initial configurations to try
    configs = []
    
    # 2x5 grid with equal sides
    c1 = np.zeros((n, 2))
    s1 = np.zeros(n)
    for i in range(n):
        r = i // 5
        c = i % 5
        c1[i] = [(c + 0.5) * 0.2, (r + 0.5) * 0.5]
        s1[i] = 0.2
    configs.append((c1, s1))
    
    # 1x10 strip
    c2 = np.zeros((n, 2))
    s2 = np.zeros(n)
    for i in range(n):
        c2[i] = [(i + 0.5) * 0.1, 0.5]
        s2[i] = 0.1
    configs.append((c2, s2))
    
    # 5x2 grid
    c3 = np.zeros((n, 2))
    s3 = np.zeros(n)
    for i in range(n):
        r = i % 2
        c = i // 2
        c3[i] = [(c + 0.5) * 0.2, (r + 0.5) * 0.5]
        s3[i] = 0.2
    configs.append((c3, s3))
    
    # Random initializations
    rng = np.random.RandomState(123)
    for trial in range(20):
        if time.time() > deadline:
            break
        
        # Random sides
        s_rand = rng.uniform(0.1, 0.3, n)
        
        # Try to place using greedy approach
        c_rand = np.zeros((n, 2))
        placed = np.zeros(n, dtype=bool)
        
        # Sort by side length descending
        order = np.argsort(-s_rand)
        
        for idx in order:
            s = s_rand[idx]
            # Try random positions
            found = False
            for _ in range(100):
                cx = rng.uniform(s/2, 1 - s/2)
                cy = rng.uniform(s/2, 1 - s/2)
                c_rand[idx] = [cx, cy]
                
                # Check if overlaps with already placed
                ok = True
                for j in range(n):
                    if not placed[j] and j != idx:
                        continue
                    if placed[j] and j != idx:
                        dx = abs(c_rand[idx, 0] - c_rand[j, 0])
                        dy = abs(c_rand[idx, 1] - c_rand[j, 1])
                        t = (s_rand[idx] + s_rand[j]) / 2.0
                        if dx < t and dy < t:
                            ok = False
                            break
                if ok:
                    placed[idx] = True
                    found = True
                    break
            
            if not found:
                # Shrink the square
                s_rand[idx] = max(0.05, s_rand[idx] * 0.8)
        
        if _is_valid(c_rand, s_rand):
            configs.append((c_rand, s_rand))
    
    # For each config, run local search
    for c_init, s_init in configs:
        if time.time() > deadline:
            break
        
        update_best(c_init, s_init)
        
        # Run local search
        c_ls, s_ls = _local_search(c_init, s_init, max_iters=200, deadline=deadline)
        update_best(c_ls, s_ls)
    
    # Strategy 2: Use LP relaxation with different M values
    M_values = [0.5, 1.0, 1.5, 2.0, 3.0]
    
    for M in M_values:
        if time.time() > deadline:
            break
        
        # Build LP relaxation
        num_pairs = n * (n - 1) // 2
        total_vars = 2 * n + n + 4 * num_pairs
        
        c_obj = np.zeros(total_vars)
        for i in range(n):
            c_obj[2 * n + i] = -1.0
        
        A_ub = []
        b_ub = []
        
        # Boundary constraints
        for i in range(n):
            row = np.zeros(total_vars)
            row[i] = -1.0
            row[2 * n + i] = -0.5
            A_ub.append(row)
            b_ub.append(0.0)
            
            row = np.zeros(total_vars)
            row[i] = 1.0
            row[2 * n + i] = 0.5
            A_ub.append(row)
            b_ub.append(1.0)
            
            row = np.zeros(total_vars)
            row[n + i] = -1.0
            row[2 * n + i] = -0.5
            A_ub.append(row)
            b_ub.append(0.0)
            
            row = np.zeros(total_vars)
            row[n + i] = 1.0
            row[2 * n + i] = 0.5
            A_ub.append(row)
            b_ub.append(1.0)
        
        # Disjunctive constraints
        pair_counter = 0
        for i in range(n):
            for j in range(i + 1, n):
                z_base = 2 * n + n + 4 * pair_counter
                
                # Orientation 0: i left of j
                row = np.zeros(total_vars)
                row[i] = 1.0
                row[j] = -1.0
                row[2 * n + i] = 0.5
                row[2 * n + j] = 0.5
                row[z_base + 0] = M
                A_ub.append(row)
                b_ub.append(M)
                
                # Orientation 1: i right of j
                row = np.zeros(total_vars)
                row[i] = -1.0
                row[j] = 1.0
                row[2 * n + i] = 0.5
                row[2 * n + j] = 0.5
                row[z_base + 1] = M
                A_ub.append(row)
                b_ub.append(M)
                
                # Orientation 2: i below j
                row = np.zeros(total_vars)
                row[n + i] = 1.0
                row[n + j] = -1.0
                row[2 * n + i] = 0.5
                row[2 * n + j] = 0.5
                row[z_base + 2] = M
                A_ub.append(row)
                b_ub.append(M)
                
                # Orientation 3: i above j
                row = np.zeros(total_vars)
                row[n + i] = -1.0
                row[n + j] = 1.0
                row[2 * n + i] = 0.5
                row[2 * n + j] = 0.5
                row[z_base + 3] = M
                A_ub.append(row)
                b_ub.append(M)
                
                pair_counter += 1
        
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        
        bounds = [(0.0, 1.0)] * (2 * n)
        bounds += [(0.01, 0.5)] * n
        bounds += [(0.0, 1.0)] * (4 * num_pairs)
        
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if res.status == 0:
            x = res.x
            centers_lp = x[:2 * n].reshape(n, 2)
            sides_lp = x[2 * n:3 * n]
            
            # Extract orientations
            orientations = {}
            pair_counter = 0
            for i in range(n):
                for j in range(i + 1, n):
                    z_base = 3 * n + 4 * pair_counter
                    z_vals = x[z_base:z_base + 4]
                    # The z variable closest to 1 indicates the active constraint
                    o = int(np.argmax(z_vals))
                    orientations[(i, j)] = o
                    pair_counter += 1
            
            update_best(centers_lp, sides_lp)
            
            # Polish with fixed orientations
            c_polished, ok = _lp_pack_fixed_sides(sides_lp, orientations)
            if ok and _is_valid(c_polished, sides_lp):
                update_best(c_polished, sides_lp)
            
            # Local search from LP result
            c_ls, s_ls = _local_search(centers_lp, sides_lp, max_iters=100, deadline=deadline)
            update_best(c_ls, s_ls)
            
            # Coordinate descent polish
            if time.time() < deadline:
                c_cd, s_cd = _coordinate_descent(centers_lp, sides_lp, max_iters=500, deadline=deadline)
                update_best(c_cd, s_cd)
    
    # Strategy 3: Final high-precision polish on best solution
    if best_centers is not None and time.time() < deadline:
        # Nelder-Mead polish
        c_nm, s_nm = _nelder_mead_polish(best_centers, best_sides, max_iter=300, deadline=deadline)
        update_best(c_nm, s_nm)
        
        # Coordinate descent on polished result
        if time.time() < deadline:
            c_cd2, s_cd2 = _coordinate_descent(best_centers, best_sides, max_iters=1000, deadline=deadline)
            update_best(c_cd2, s_cd2)
    
    if best_centers is None:
        best_centers = fallback_c
        best_sides = fallback_s
    
    return best_centers, best_sides

# EVOLVE-BLOCK-END
