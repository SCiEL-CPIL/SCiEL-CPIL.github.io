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


def _local_refine(centers, sides, tol=1e-7, max_iter=2000):
    """
    Local refinement: try to grow squares while maintaining feasibility.
    """
    n = len(sides)
    best_centers = centers.copy()
    best_sides = sides.copy()
    best_val = np.sum(sides)
    
    deltas = [0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]
    
    for iteration in range(max_iter):
        improved = False
        for i in range(n):
            for delta in deltas:
                new_sides = best_sides.copy()
                new_sides[i] = best_sides[i] + delta
                new_centers = best_centers.copy()
                
                s = new_sides[i]
                cx = np.clip(best_centers[i, 0], s/2, 1 - s/2)
                cy = np.clip(best_centers[i, 1], s/2, 1 - s/2)
                new_centers[i] = [cx, cy]
                
                if _is_feasible(new_centers, new_sides, tol):
                    best_centers = new_centers
                    best_sides = new_sides
                    best_val = np.sum(best_sides)
                    improved = True
                    break
        
        if not improved:
            break
    
    return best_centers, best_sides


def _penalty_objective(x, tol=1e-7, penalty_weight=1000.0):
    """
    Penalty-based objective for packing 10 squares.
    x = [cx_0..cx_9, cy_0..cy_9, s_0..s_9]
    Minimizes: -sum(s_i) + penalty * (boundary_violations + overlap_violations)
    """
    n = 10
    cx = x[:n]
    cy = x[n:2*n]
    s = x[2*n:3*n]
    
    # Objective: maximize sum of sides (minimize negative)
    obj = -np.sum(s)
    
    # Boundary violations
    for i in range(n):
        if s[i] <= 0:
            obj += penalty_weight * 10.0
        else:
            if cx[i] - s[i]/2 < -tol:
                obj += penalty_weight * abs(cx[i] - s[i]/2 + tol)
            if cx[i] + s[i]/2 > 1 + tol:
                obj += penalty_weight * abs(cx[i] + s[i]/2 - 1 - tol)
            if cy[i] - s[i]/2 < -tol:
                obj += penalty_weight * abs(cy[i] - s[i]/2 + tol)
            if cy[i] + s[i]/2 > 1 + tol:
                obj += penalty_weight * abs(cy[i] + s[i]/2 - 1 - tol)
    
    # Overlap violations
    for i in range(n):
        for j in range(i+1, n):
            t = (s[i] + s[j])/2 - tol
            dx = abs(cx[i] - cx[j])
            dy = abs(cy[i] - cy[j])
            if dx < t and dy < t:
                # Both dimensions overlap - violation
                violation = min(t - dx, t - dy)
                obj += penalty_weight * violation
    
    return obj


def _penalty_refine(centers, sides, tol=1e-7, deadline=None, max_iter=500):
    """
    Penalty-based refinement using SLSQP.
    """
    n = len(sides)
    x0 = np.concatenate([centers.flatten(), sides])
    
    bounds = [(1e-8, 1.0) for _ in range(3*n)]
    
    result = minimize(
        _penalty_objective, x0, args=(tol,),
        method='SLSQP', bounds=bounds,
        options={'maxiter': max_iter, 'ftol': 1e-12, 'disp': False}
    )
    
    if np.all(np.isfinite(result.x)):
        new_centers = result.x[:2*n].reshape(n, 2)
        new_sides = result.x[2*n:]
        
        # Clip to valid range
        for i in range(n):
            new_sides[i] = max(1e-6, new_sides[i])
            new_centers[i, 0] = np.clip(new_centers[i, 0], new_sides[i]/2, 1 - new_sides[i]/2)
            new_centers[i, 1] = np.clip(new_centers[i, 1], new_sides[i]/2, 1 - new_sides[i]/2)
        
        if _is_feasible(new_centers, new_sides, tol):
            return new_centers, new_sides
    
    return centers, sides


def _coordinate_ascent(centers, sides, tol=1e-7, deadline=None):
    """
    Coordinate ascent: for each square, find the maximum size it can grow to
    by checking available space in x and y directions.
    """
    n = len(sides)
    best_centers = centers.copy()
    best_sides = sides.copy()
    
    for iteration in range(100):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        
        for i in range(n):
            # Try to grow square i
            current_s = best_sides[i]
            cx, cy = best_centers[i]
            
            # Find max possible size based on boundaries and other squares
            max_s_x = 2 * min(cx, 1 - cx)
            max_s_y = 2 * min(cy, 1 - cy)
            max_s = min(max_s_x, max_s_y)
            
            # Check against other squares
            for j in range(n):
                if i == j:
                    continue
                oj_cx, oj_cy = best_centers[j]
                oj_s = best_sides[j]
                
                # If we grow square i, when would it overlap with j?
                # Overlap in x: |cx_i - cx_j| < (s_i + s_j)/2
                # Overlap in y: |cy_i - cy_j| < (s_i + s_j)/2
                dx = abs(cx - oj_cx)
                dy = abs(cy - oj_cy)
                
                # Current separation
                current_sep_x = dx - (current_s + oj_s)/2
                current_sep_y = dy - (current_s + oj_s)/2
                
                # If both separations are negative, they're already overlapping (shouldn't happen)
                # If one is positive, we can grow until that dimension just touches
                
                # Max growth before x-overlap
                if current_sep_x >= 0:
                    # Can grow in x until sep_x = 0
                    # (s_i + oj_s)/2 = dx
                    # s_i = 2*dx - oj_s
                    max_s_x_from_j = 2*dx - oj_s
                else:
                    max_s_x_from_j = current_s  # Already overlapping in x
                
                # Max growth before y-overlap
                if current_sep_y >= 0:
                    max_s_y_from_j = 2*dy - oj_s
                else:
                    max_s_y_from_j = current_s
                
                max_s = min(max_s, max_s_x_from_j, max_s_y_from_j)
            
            # Try to grow to max_s
            if max_s > current_s + 1e-8:
                new_s = max_s
                # Keep center fixed, just grow
                new_sides = best_sides.copy()
                new_sides[i] = new_s
                new_centers = best_centers.copy()
                
                # Adjust center if needed to stay in bounds
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


def _smooth_refine(centers, sides, tol=1e-7, deadline=None):
    """
    Smooth SLSQP-based refinement using a soft penalty surrogate.
    """
    n = len(sides)
    x0 = np.concatenate([centers.flatten(), sides])
    
    # Soft-abs parameter for smoothness
    k = 50.0  # larger = sharper, more accurate
    
    def soft_abs(a, k=k):
        """Smooth approximation of |a| using log-sum-exp."""
        return (a + np.logaddexp(0, -k*a)) / k
    
    def objective(x):
        """Minimize negative sum of sides."""
        s = x[2*n:]
        return -np.sum(s)
    
    def constraint_overlap(i, j):
        """Return constraint function for pair (i,j): |dx|+|dy| - (s_i+s_j)/2 + tol >= 0"""
        def c(x):
            cx_i, cy_i = x[i], x[n+i]
            cx_j, cy_j = x[j], x[n+j]
            s_i, s_j = x[2*n+i], x[2*n+j]
            dx = cx_i - cx_j
            dy = cy_i - cy_j
            # Use soft-abs for smoothness
            sadx = soft_abs(dx)
            sady = soft_abs(dy)
            return sadx + sady - (s_i + s_j)/2.0 + tol
        return c
    
    def constraint_boundary_x(i):
        def c_lower(x):
            s_i = x[2*n+i]
            return x[i] - s_i/2.0
        def c_upper(x):
            s_i = x[2*n+i]
            return 1.0 - x[i] - s_i/2.0
        return c_lower, c_upper
    
    def constraint_boundary_y(i):
        def c_lower(x):
            s_i = x[2*n+i]
            return x[n+i] - s_i/2.0
        def c_upper(x):
            s_i = x[2*n+i]
            return 1.0 - x[n+i] - s_i/2.0
        return c_lower, c_upper
    
    # Build constraints
    constraints = []
    
    # Boundary constraints
    for i in range(n):
        cx_lo, cx_hi = constraint_boundary_x(i)
        cy_lo, cy_hi = constraint_boundary_y(i)
        constraints.append({'type': 'ineq', 'fun': cx_lo})
        constraints.append({'type': 'ineq', 'fun': cx_hi})
        constraints.append({'type': 'ineq', 'fun': cy_lo})
        constraints.append({'type': 'ineq', 'fun': cy_hi})
    
    # Overlap constraints (soft)
    for i in range(n):
        for j in range(i+1, n):
            constraints.append({'type': 'ineq', 'fun': constraint_overlap(i, j)})
    
    # Side positivity: s_i > 0, enforced via lower bound
    bounds = []
    for i in range(n):
        bounds.append((1e-6, 1.0))  # cx_i
        bounds.append((1e-6, 1.0))  # cy_i
        bounds.append((1e-6, 1.0))  # s_i
    
    # Run SLSQP
    result = minimize(
        objective, x0, method='SLSQP',
        bounds=bounds, constraints=constraints,
        options={'maxiter': 500, 'ftol': 1e-12, 'disp': False}
    )
    
    if result.success and np.all(np.isfinite(result.x)):
        new_centers = result.x[:2*n].reshape(n, 2)
        new_sides = result.x[2*n:]
        # Clip to valid range
        for i in range(n):
            new_sides[i] = max(1e-6, new_sides[i])
            new_centers[i, 0] = np.clip(new_centers[i, 0], new_sides[i]/2, 1 - new_sides[i]/2)
            new_centers[i, 1] = np.clip(new_centers[i, 1], new_sides[i]/2, 1 - new_sides[i]/2)
        
        if _is_feasible(new_centers, new_sides, tol):
            return new_centers, new_sides
    
    return centers, sides


def _high_precision_polish(centers, sides, tol=1e-7, deadline=None):
    """
    Final high-precision polish: run SLSQP with tighter settings from the best found.
    """
    try:
        c, s = _smooth_refine(centers, sides, tol=tol, deadline=deadline)
        return c, s
    except Exception:
        return centers, sides


def _multi_start_optimize(deadline):
    """
    Try multiple initial configurations and optimize each.
    """
    best_centers = None
    best_sides = None
    best_val = -np.inf
    
    # Configuration 1: 2 large + 8 small (current best)
    c1, s1 = _make_2big_8small()
    
    # Configuration 2: 4 medium (0.4) + 6 smaller (0.3)
    # Place 4 medium in corners, 6 smaller in center
    c2 = np.array([
        [0.2, 0.2], [0.8, 0.2], [0.2, 0.8], [0.8, 0.8],  # 4 medium at corners
        [0.5, 0.25], [0.5, 0.75], [0.25, 0.5], [0.75, 0.5],  # 4 in mid-edges
        [0.5, 0.5], [0.5, 0.5]  # 2 in center (will be adjusted)
    ])
    s2 = np.array([0.4, 0.4, 0.4, 0.4, 0.3, 0.3, 0.3, 0.3, 0.2, 0.2])
    # Adjust center squares to not overlap
    c2[8] = [0.35, 0.35]
    c2[9] = [0.65, 0.65]
    
    # Configuration 3: 5 medium (0.4) + 5 small (0.3)
    c3 = np.array([
        [0.2, 0.2], [0.8, 0.2], [0.5, 0.5], [0.2, 0.8], [0.8, 0.8],
        [0.5, 0.2], [0.2, 0.5], [0.8, 0.5], [0.5, 0.8], [0.5, 0.5]
    ])
    s3 = np.array([0.4, 0.4, 0.4, 0.4, 0.4, 0.3, 0.3, 0.3, 0.3, 0.2])
    # Fix overlap in center
    c3[10-1] = [0.6, 0.6]  # Last square
    
    configs = [(c1, s1), (c2, s2), (c3, s3)]
    
    for idx, (c, s) in enumerate(configs):
        if time.time() > deadline - 60:
            break
        
        # Check feasibility
        if not _is_feasible(c, s):
            continue
        
        val = np.sum(s)
        
        # Try coordinate ascent
        try:
            ca_centers, ca_sides = _coordinate_ascent(c, s, deadline=deadline - 30)
            if _is_feasible(ca_centers, ca_sides):
                ca_val = np.sum(ca_sides)
                if ca_val > val:
                    c, s = ca_centers, ca_sides
                    val = ca_val
        except Exception:
            pass
        
        # Try penalty refinement
        try:
            p_centers, p_sides = _penalty_refine(c, s, deadline=deadline - 20, max_iter=200)
            if _is_feasible(p_centers, p_sides):
                p_val = np.sum(p_sides)
                if p_val > val:
                    c, s = p_centers, p_sides
                    val = p_val
        except Exception:
            pass
        
        # Try smooth refinement
        try:
            sm_centers, sm_sides = _smooth_refine(c, s, deadline=deadline - 10)
            if _is_feasible(sm_centers, sm_sides):
                sm_val = np.sum(sm_sides)
                if sm_val > val:
                    c, s = sm_centers, sm_sides
                    val = sm_val
        except Exception:
            pass
        
        if val > best_val:
            best_val = val
            best_centers = c.copy()
            best_sides = s.copy()
    
    return best_centers, best_sides


def squares_in_square10():
    start_time = time.time()
    deadline = start_time + 580
    
    best_centers = None
    best_sides = None
    best_val = -np.inf
    
    # Primary packing: 2 large (0.5) + 8 small (0.25), sum = 3
    c1, s1 = _make_2big_8small()
    if _is_feasible(c1, s1):
        best_centers = c1.copy()
        best_sides = s1.copy()
        best_val = np.sum(s1)
    
    # Coordinate ascent refinement
    if best_centers is not None and time.time() < deadline - 120:
        try:
            ca_centers, ca_sides = _coordinate_ascent(
                best_centers, best_sides, deadline=deadline - 100
            )
            if _is_feasible(ca_centers, ca_sides):
                ca_val = np.sum(ca_sides)
                if ca_val > best_val:
                    best_val = ca_val
                    best_centers = ca_centers
                    best_sides = ca_sides
        except Exception:
            pass
    
    # Local refinement (discrete)
    if best_centers is not None and time.time() < deadline - 90:
        try:
            refined_centers, refined_sides = _local_refine(
                best_centers, best_sides, max_iter=1000
            )
            if _is_feasible(refined_centers, refined_sides):
                refined_val = np.sum(refined_sides)
                if refined_val > best_val:
                    best_val = refined_val
                    best_centers = refined_centers
                    best_sides = refined_sides
        except Exception:
            pass
    
    # Penalty-based refinement
    if best_centers is not None and time.time() < deadline - 60:
        try:
            p_centers, p_sides = _penalty_refine(
                best_centers, best_sides, deadline=deadline - 40, max_iter=300
            )
            if _is_feasible(p_centers, p_sides):
                p_val = np.sum(p_sides)
                if p_val > best_val:
                    best_val = p_val
                    best_centers = p_centers
                    best_sides = p_sides
        except Exception:
            pass
    
    # Multi-start optimization with alternative configurations
    if time.time() < deadline - 30:
        try:
            ms_centers, ms_sides = _multi_start_optimize(deadline - 15)
            if ms_centers is not None and _is_feasible(ms_centers, ms_sides):
                ms_val = np.sum(ms_sides)
                if ms_val > best_val:
                    best_val = ms_val
                    best_centers = ms_centers
                    best_sides = ms_sides
        except Exception:
            pass
    
    # Final polish
    if best_centers is not None and time.time() < deadline - 10:
        try:
            polish_centers, polish_sides = _high_precision_polish(
                best_centers, best_sides, tol=1e-7, deadline=deadline - 5
            )
            if _is_feasible(polish_centers, polish_sides):
                polish_val = np.sum(polish_sides)
                if polish_val > best_val:
                    best_val = polish_val
                    best_centers = polish_centers
                    best_sides = polish_sides
        except Exception:
            pass
    
    if best_centers is None:
        # Fallback: should not happen
        best_centers, best_sides = _make_2big_8small()
    
    return best_centers, best_sides
