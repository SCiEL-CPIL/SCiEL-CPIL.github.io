import numpy as np
import time
from scipy.optimize import minimize, differential_evolution


def _is_feasible(centers, sides, tol=1e-7):
    """Check if the configuration is feasible."""
    n = len(sides)
    for i in range(n):
        cx, cy = centers[i]
        s = sides[i]
        if s <= 0:
            return False
        if not (np.isfinite(cx) and np.isfinite(cy) and np.isfinite(s)):
            return False
        if cx - s/2 < -tol or cx + s/2 > 1 + tol:
            return False
        if cy - s/2 < -tol or cy + s/2 > 1 + tol:
            return False
    for i in range(n):
        for j in range(i+1, n):
            t = (sides[i] + sides[j])/2 - tol
            dx = abs(centers[i, 0] - centers[j, 0])
            dy = abs(centers[i, 1] - centers[j, 1])
            if dx < t and dy < t:
                return False
    return True


def _make_2big_8small():
    """Construct 2 large (0.5) + 8 small (0.25) tiling. Sum = 3.0."""
    centers = np.array([
        # 2 large squares at top
        [0.25, 0.75],  # large 1: [0,0.5]x[0.5,1]
        [0.75, 0.75],  # large 2: [0.5,1]x[0.5,1]
        # 8 small squares in 2x4 grid below
        [0.125, 0.125], [0.375, 0.125], [0.625, 0.125], [0.875, 0.125],
        [0.125, 0.375], [0.375, 0.375], [0.625, 0.375], [0.875, 0.375]
    ])
    sides = np.array([0.5, 0.5] + [0.25]*8)
    return centers, sides


def _make_3big_7small():
    """Try 3 large squares + 7 smaller ones. Sum potentially > 3.0."""
    # 3 squares of size ~0.45 in a row at bottom, 7 smaller above
    s_big = 0.45
    s_small = 0.32
    centers = np.array([
        [0.225, 0.225], [0.5, 0.225], [0.775, 0.225],  # 3 big at bottom
        [0.15, 0.65], [0.35, 0.65], [0.55, 0.65], [0.75, 0.65],  # 4 small top row
        [0.25, 0.88], [0.5, 0.88], [0.75, 0.88]  # 3 small top-top
    ])
    sides = np.array([s_big]*3 + [s_small]*7)
    return centers, sides


def _make_mixed_packing():
    """Mixed packing with varied sizes."""
    # Try a more organic arrangement
    centers = np.array([
        [0.2, 0.2], [0.8, 0.2], [0.5, 0.15],  # bottom row
        [0.2, 0.6], [0.8, 0.6], [0.5, 0.55],  # middle row  
        [0.2, 0.85], [0.5, 0.85], [0.8, 0.85], [0.5, 0.7]  # top
    ])
    sides = np.array([0.35, 0.35, 0.3, 0.35, 0.35, 0.3, 0.3, 0.3, 0.3, 0.25])
    return centers, sides


def _move_and_grow(centers, sides, tol=1e-7, deadline=None, max_iter=500):
    """
    Local search that can both move and grow squares.
    Tries random perturbations of position and size.
    """
    n = len(sides)
    best_centers = centers.copy()
    best_sides = sides.copy()
    best_val = np.sum(sides)
    
    rng = np.random.default_rng(42)
    
    for iteration in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        improved = False
        # Try modifying one square at a time
        for i in range(n):
            # Generate several random perturbations
            for attempt in range(20):
                # Random size change
                size_factor = 1.0 + rng.uniform(-0.05, 0.15)
                new_s = best_sides[i] * size_factor
                new_s = np.clip(new_s, 1e-6, 1.0)
                
                # Random position change
                dx = rng.uniform(-0.05, 0.05)
                dy = rng.uniform(-0.05, 0.05)
                
                new_centers = best_centers.copy()
                new_sides = best_sides.copy()
                new_sides[i] = new_s
                new_cx = np.clip(best_centers[i, 0] + dx, new_s/2, 1 - new_s/2)
                new_cy = np.clip(best_centers[i, 1] + dy, new_s/2, 1 - new_s/2)
                new_centers[i] = [new_cx, new_cy]
                
                if _is_feasible(new_centers, new_sides, tol):
                    new_val = np.sum(new_sides)
                    if new_val > best_val + 1e-10:
                        best_centers = new_centers
                        best_sides = new_sides
                        best_val = new_val
                        improved = True
                        break
        
        if not improved and iteration > 100:
            # Reduce perturbation size
            break
        if not improved and iteration > 300:
            break
    
    return best_centers, best_sides


def _simulated_annealing(centers, sides, tol=1e-7, deadline=None):
    """
    Simulated annealing to escape local optima.
    """
    n = len(sides)
    current_centers = centers.copy()
    current_sides = sides.copy()
    current_val = np.sum(sides)
    
    best_centers = current_centers.copy()
    best_sides = current_sides.copy()
    best_val = current_val
    
    rng = np.random.default_rng(123)
    
    T_start = 0.1
    T_end = 0.001
    n_iter = 2000
    
    for it in range(n_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        T = T_start * (T_end / T_start) ** (it / n_iter)
        
        # Pick a random square to modify
        i = rng.integers(n)
        
        # Random perturbation
        size_change = rng.uniform(-0.03, 0.05)
        dx = rng.uniform(-0.03, 0.03)
        dy = rng.uniform(-0.03, 0.03)
        
        new_s = np.clip(current_sides[i] + size_change, 1e-6, 1.0)
        new_cx = np.clip(current_centers[i, 0] + dx, new_s/2, 1 - new_s/2)
        new_cy = np.clip(current_centers[i, 1] + dy, new_s/2, 1 - new_s/2)
        
        new_centers = current_centers.copy()
        new_sides = current_sides.copy()
        new_sides[i] = new_s
        new_centers[i] = [new_cx, new_cy]
        
        if _is_feasible(new_centers, new_sides, tol):
            new_val = np.sum(new_sides)
            delta_val = new_val - current_val
            
            # Accept if better or with probability exp(-delta/T) for worse
            if delta_val > 0 or rng.random() < np.exp(-delta_val / max(T, 1e-10)):
                current_centers = new_centers
                current_sides = new_sides
                current_val = new_val
                
                if current_val > best_val:
                    best_centers = current_centers.copy()
                    best_sides = current_sides.copy()
                    best_val = current_val
    
    return best_centers, best_sides


def _coordinate_ascent_v2(centers, sides, tol=1e-7, deadline=None):
    """
    Improved coordinate ascent that considers moving squares.
    """
    n = len(sides)
    best_centers = centers.copy()
    best_sides = sides.copy()
    
    for iteration in range(200):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        
        for i in range(n):
            current_s = best_sides[i]
            cx, cy = best_centers[i]
            
            # Try to grow square i by finding max possible size
            max_s = 2 * min(cx, 1 - cx, cy, 1 - cy)
            
            for j in range(n):
                if i == j:
                    continue
                oj_cx, oj_cy = best_centers[j]
                oj_s = best_sides[j]
                
                dx = abs(cx - oj_cx)
                dy = abs(cy - oj_cy)
                
                # Max size before overlapping in x or y
                # If we grow i, the half-width becomes s/2
                # Overlap in x when: cx_i - s_i/2 < oj_cx + oj_s/2 AND cx_i + s_i/2 > oj_cx - oj_s/2
                # i.e., s_i > 2*dx - oj_s (if cx_i > oj_cx) or similar
                
                # Simpler: just check if growing to max_s would cause overlap
                # and reduce accordingly
                
                # Distance from edge of i to edge of j in x
                if cx >= oj_cx:
                    sep_x = cx - oj_cx - (current_s + oj_s)/2
                    # Can grow by at most sep_x in x direction
                    max_from_j_x = current_s + 2*sep_x if sep_x > 0 else current_s
                else:
                    sep_x = oj_cx - cx - (current_s + oj_s)/2
                    max_from_j_x = current_s + 2*sep_x if sep_x > 0 else current_s
                
                if cy >= oj_cy:
                    sep_y = cy - oj_cy - (current_s + oj_s)/2
                    max_from_j_y = current_s + 2*sep_y if sep_y > 0 else current_s
                else:
                    sep_y = oj_cy - cy - (current_s + oj_s)/2
                    max_from_j_y = current_s + 2*sep_y if sep_y > 0 else current_s
                
                max_s = min(max_s, max_from_j_x, max_from_j_y)
            
            # Also try moving the square to find more space
            # Try shifting in 4 directions
            for dcx in [-0.02, 0.02]:
                for dcy in [-0.02, 0.02]:
                    new_cx = np.clip(cx + dcx, current_s/2, 1 - current_s/2)
                    new_cy = np.clip(cy + dcy, current_s/2, 1 - current_s/2)
                    
                    # Recalculate max size from new position
                    max_s_try = 2 * min(new_cx, 1 - new_cx, new_cy, 1 - new_cy)
                    
                    for j in range(n):
                        if i == j:
                            continue
                        oj_cx, oj_cy = best_centers[j]
                        oj_s = best_sides[j]
                        
                        dx = abs(new_cx - oj_cx)
                        dy = abs(new_cy - oj_cy)
                        
                        if new_cx >= oj_cx:
                            sep_x = new_cx - oj_cx - (current_s + oj_s)/2
                            max_from_j_x = current_s + 2*sep_x if sep_x > 0 else current_s
                        else:
                            sep_x = oj_cx - new_cx - (current_s + oj_s)/2
                            max_from_j_x = current_s + 2*sep_x if sep_x > 0 else current_s
                        
                        if new_cy >= oj_cy:
                            sep_y = new_cy - oj_cy - (current_s + oj_s)/2
                            max_from_j_y = current_s + 2*sep_y if sep_y > 0 else current_s
                        else:
                            sep_y = oj_cy - new_cy - (current_s + oj_s)/2
                            max_from_j_y = current_s + 2*sep_y if sep_y > 0 else current_s
                        
                        max_s_try = min(max_s_try, max_from_j_x, max_from_j_y)
                    
                    if max_s_try > max_s:
                        max_s = max_s_try
                        cx, cy = new_cx, new_cy
            
            if max_s > current_s + 1e-8:
                new_s = max_s
                new_sides = best_sides.copy()
                new_sides[i] = new_s
                new_centers = best_centers.copy()
                
                new_cx = np.clip(cx, new_s/2, 1 - new_s/2)
                new_cy = np.clip(cy, new_s/2, 1 - new_s/2)
                new_centers[i] = [new_cx, new_cy]
                
                if _is_feasible(new_centers, new_sides, tol):
                    best_centers = new_centers
                    best_sides = new_sides
                    improved = True
        
        if not improved:
            break
    
    return best_centers, best_sides


def _differential_evolution_optimize(deadline):
    """
    Use differential evolution with a feasibility-based fitness.
    """
    n = 10
    
    # Bounds: cx, cy in [0,1], s in [0.01, 0.6]
    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0))  # cx
    for i in range(n):
        bounds.append((0.0, 1.0))  # cy
    for i in range(n):
        bounds.append((0.01, 0.6))  # s
    
    def objective(x):
        centers = np.array([[x[i], x[n+i]] for i in range(n)])
        sides = x[2*n:3*n]
        
        # Check feasibility
        if not _is_feasible(centers, sides):
            # Penalty for infeasibility
            penalty = 0
            for i in range(n):
                cx, cy = centers[i]
                s = sides[i]
                if cx - s/2 < 0:
                    penalty += abs(cx - s/2)
                if cx + s/2 > 1:
                    penalty += abs(cx + s/2 - 1)
                if cy - s/2 < 0:
                    penalty += abs(cy - s/2)
                if cy + s/2 > 1:
                    penalty += abs(cy + s/2 - 1)
            for i in range(n):
                for j in range(i+1, n):
                    t = (sides[i] + sides[j])/2
                    dx = abs(centers[i, 0] - centers[j, 0])
                    dy = abs(centers[i, 1] - centers[j, 1])
                    if dx < t and dy < t:
                        penalty += min(t - dx, t - dy)
            return -np.sum(sides) + 100 * penalty
        
        return -np.sum(sides)
    
    # Start from a good initial point
    c0, s0 = _make_2big_8small()
    x0 = np.concatenate([c0.flatten(), s0])
    
    result = differential_evolution(
        objective, bounds, 
        x0=x0,
        seed=42,
        maxiter=100,
        tol=1e-8,
        polish=True,
        workers=1,
        init='sobol'
    )
    
    if np.all(np.isfinite(result.x)):
        centers = np.array([[result.x[i], result.x[n+i]] for i in range(n)])
        sides = result.x[2*n:3*n]
        
        # Clip to valid range
        for i in range(n):
            sides[i] = max(1e-6, sides[i])
            centers[i, 0] = np.clip(centers[i, 0], sides[i]/2, 1 - sides[i]/2)
            centers[i, 1] = np.clip(centers[i, 1], sides[i]/2, 1 - sides[i]/2)
        
        if _is_feasible(centers, sides):
            return centers, sides
    
    return None, None


def squares_in_square10():
    start_time = time.time()
    deadline = start_time + 580
    
    best_centers = None
    best_sides = None
    best_val = -np.inf
    
    # Try multiple initial configurations
    configs = [
        _make_2big_8small(),
        _make_3big_7small(),
        _make_mixed_packing()
    ]
    
    for c, s in configs:
        if not _is_feasible(c, s):
            continue
        val = np.sum(s)
        if val > best_val:
            best_val = val
            best_centers = c.copy()
            best_sides = s.copy()
    
    if best_centers is None:
        best_centers, best_sides = _make_2big_8small()
        best_val = np.sum(best_sides)
    
    # Coordinate ascent v2 (with movement)
    if time.time() < deadline - 180:
        try:
            ca_c, ca_s = _coordinate_ascent_v2(
                best_centers, best_sides, deadline=deadline - 150
            )
            if _is_feasible(ca_c, ca_s):
                ca_val = np.sum(ca_s)
                if ca_val > best_val:
                    best_val = ca_val
                    best_centers = ca_c
                    best_sides = ca_s
        except Exception:
            pass
    
    # Move and grow local search
    if time.time() < deadline - 120:
        try:
            mg_c, mg_s = _move_and_grow(
                best_centers, best_sides, deadline=deadline - 100
            )
            if _is_feasible(mg_c, mg_s):
                mg_val = np.sum(mg_s)
                if mg_val > best_val:
                    best_val = mg_val
                    best_centers = mg_c
                    best_sides = mg_s
        except Exception:
            pass
    
    # Simulated annealing
    if time.time() < deadline - 80:
        try:
            sa_c, sa_s = _simulated_annealing(
                best_centers, best_sides, deadline=deadline - 60
            )
            if _is_feasible(sa_c, sa_s):
                sa_val = np.sum(sa_s)
                if sa_val > best_val:
                    best_val = sa_val
                    best_centers = sa_c
                    best_sides = sa_s
        except Exception:
            pass
    
    # Differential evolution
    if time.time() < deadline - 40:
        try:
            de_c, de_s = _differential_evolution_optimize(deadline - 20)
            if de_c is not None and _is_feasible(de_c, de_s):
                de_val = np.sum(de_s)
                if de_val > best_val:
                    best_val = de_val
                    best_centers = de_c
                    best_sides = de_s
        except Exception:
            pass
    
    # Final coordinate ascent pass
    if time.time() < deadline - 10:
        try:
            final_c, final_s = _coordinate_ascent_v2(
                best_centers, best_sides, deadline=deadline - 5
            )
            if _is_feasible(final_c, final_s):
                final_val = np.sum(final_s)
                if final_val > best_val:
                    best_val = final_val
                    best_centers = final_c
                    best_sides = final_s
        except Exception:
            pass
    
    return best_centers, best_sides
