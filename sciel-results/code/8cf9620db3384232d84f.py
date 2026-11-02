import numpy as np
import time
import itertools

N = 11
H = np.sqrt(3) / 2
SQRT3 = np.sqrt(3)
TRI_AREA = SQRT3 / 4.0

# Precompute all C(11,3) = 165 triple indices
_indices = list(itertools.combinations(range(N), 3))
_idx_arr = np.array(_indices)  # shape (165, 3)

def _compute_areas(points):
    """Compute all 165 triangle areas from 11 points."""
    i = _idx_arr[:, 0]
    j = _idx_arr[:, 1]
    k = _idx_arr[:, 2]
    xi, yi = points[i, 0], points[i, 1]
    xj, yj = points[j, 0], points[j, 1]
    xk, yk = points[k, 0], points[k, 1]
    det = xi * (yj - yk) + xj * (yk - yi) + xk * (yi - yj)
    return 0.5 * np.abs(det)

def _min_area_and_idx(points):
    """Return (min_area, index_of_min_triangle)."""
    areas = _compute_areas(points)
    idx = int(np.argmin(areas))
    return float(areas[idx]), idx

def _bary_to_cart(bary):
    """Convert barycentric coords (u,v) to Cartesian, ensuring validity."""
    u = np.clip(bary[:, 0], 0.0, None)
    v = np.clip(bary[:, 1], 0.0, None)
    s = u + v
    mask = s > 1.0
    if np.any(mask):
        u = u.copy()
        v = v.copy()
        u[mask] /= s[mask]
        v[mask] /= s[mask]
    pts = np.empty((N, 2))
    pts[:, 0] = v * 1.0 + (1.0 - u - v) * 0.5
    pts[:, 1] = (1.0 - u - v) * H
    return pts

def _cart_to_bary(pts):
    """Convert Cartesian points to barycentric coords (u, v)."""
    bary = np.empty((N, 2))
    for m in range(N):
        x = pts[m, 0]
        y = pts[m, 1]
        bary[m, 0] = 1.0 - x - y / (2.0 * H)
        bary[m, 1] = x - y / (2.0 * H)
    return bary

def _project_to_triangle(pts):
    """Project points to be inside the triangle using barycentric coords."""
    bary = _cart_to_bary(pts)
    return _bary_to_cart(bary)

def _initial_configuration():
    """
    Create a structured initial configuration.
    Use rows of points following the triangle shape.
    Row 0 (bottom, y near 0): 4 points
    Row 1 (middle): 3 points  
    Row 2 (upper): 2 points
    Row 3 (top): 1 point
    Total: 4+3+2+1 = 10... need 11.
    
    Let's use:
    Row 0: 5 points (bottom edge)
    Row 1: 3 points
    Row 2: 2 points
    Row 3: 1 point
    Total: 5+3+2+1 = 11
    """
    pts = np.zeros((N, 2))
    # Row 0: y = 0.05, 5 points spread along x
    n0 = 5
    for i in range(n0):
        t = (i + 0.5) / n0
        pts[i, 0] = t * 1.0  # x from 0 to 1
        pts[i, 1] = 0.05
    
    # Row 1: y = H/3, 3 points
    n1 = 3
    y1 = H / 3.0
    # At height y, the triangle width is 1 - y/H (scaled)
    # Actually at height y, the x range is from y/SQRT3 to 1 - y/SQRT3... 
    # Let me use barycentric for better placement
    for i in range(n1):
        t = (i + 0.5) / n1
        # Barycentric: at level 1/3 from bottom
        # u + v = 1 - y/H = 1 - 1/3 = 2/3
        # x = v + 0.5*(1-u-v) = v + 0.5*(1/3) = v + 1/6
        # So v = x - 1/6, u = 2/3 - v = 2/3 - x + 1/6 = 5/6 - x
        x = t * (1.0 - y1 / H) + y1 / SQRT3  # approximate
        # Better: use barycentric directly
        u = 0.2
        v = t * (1.0 - 2*u)  # so u+v <= 1
        # Let me just place them manually
        pass
    
    # Simpler: use a grid in barycentric coordinates
    bary = np.zeros((N, 2))
    # Place points in a roughly uniform pattern in the triangle
    # Use a 3x3 grid clipped to the triangle, plus 2 extra
    grid_positions = [
        (0.1, 0.1), (0.4, 0.1), (0.7, 0.1),
        (0.2, 0.35), (0.5, 0.35), (0.8, 0.35),
        (0.3, 0.6), (0.6, 0.6),
        (0.4, 0.8),
        (0.5, 0.05),
        (0.15, 0.5),
    ]
    for m, (u, v) in enumerate(grid_positions):
        if u + v > 1.0:
            s = 1.0 / (u + v)
            u *= s
            v *= s
        bary[m, 0] = u
        bary[m, 1] = v
    
    return _bary_to_cart(bary)

def _perturb_points(pts, rng, scale=0.02):
    """Perturb points slightly, keeping them in the triangle."""
    new_pts = pts.copy()
    for i in range(N):
        dx = rng.normal(0, scale)
        dy = rng.normal(0, scale)
        new_pts[i, 0] += dx
        new_pts[i, 1] += dy
    return _project_to_triangle(new_pts)

def _targeted_perturb(pts, min_idx, rng, scale=0.03):
    """Perturb only the points involved in the minimum-area triangle."""
    new_pts = pts.copy()
    tri = _indices[min_idx]
    for i in tri:
        dx = rng.normal(0, scale)
        dy = rng.normal(0, scale)
        new_pts[i, 0] += dx
        new_pts[i, 1] += dy
    return _project_to_triangle(new_pts)

def _nelder_mead_step(pts, rng, max_iter=500, tol=1e-10):
    """
    Perform Nelder-Mead optimization on the direct objective.
    """
    from scipy.optimize import minimize
    
    def objective(params):
        p = _bary_to_cart(params.reshape(N, 2))
        area, _ = _min_area_and_idx(p)
        return -area  # minimize negative = maximize min area
    
    bary = _cart_to_bary(pts)
    bary = np.clip(bary, 0.0, 1.0)
    for m in range(N):
        if bary[m, 0] + bary[m, 1] > 1.0:
            s = 1.0 / (bary[m, 0] + bary[m, 1])
            bary[m, 0] *= s
            bary[m, 1] *= s
    
    bounds = [(0.0, 1.0)] * (2 * N)
    
    try:
        result = minimize(
            objective,
            bary.flatten(),
            method='Nelder-Mead',
            bounds=bounds,
            options={'maxiter': max_iter, 'xatol': tol, 'fatol': tol, 'adaptive': True}
        )
        if result.fun < 0:
            new_pts = _bary_to_cart(result.x.reshape(N, 2))
            return new_pts
    except Exception:
        pass
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    rng = np.random.default_rng(12345)
    
    # Start with a good initial configuration
    best_pts = _initial_configuration()
    best_area, best_idx = _min_area_and_idx(best_pts)
    
    # Phase 1: Nelder-Mead from initial config
    if time.time() < deadline - 30:
        nm_pts = _nelder_mead_step(best_pts, rng, max_iter=300)
        nm_area, _ = _min_area_and_idx(nm_pts)
        if nm_area > best_area:
            best_pts = nm_pts
            best_area = nm_area
    
    # Phase 2: Iterative targeted perturbation + local optimization
    no_improve = 0
    max_no_improve = 30
    
    while time.time() < deadline - 15 and no_improve < max_no_improve:
        # Get current min triangle
        cur_area, cur_idx = _min_area_and_idx(best_pts)
        
        # Try targeted perturbation of the bottleneck triangle
        perturbed = _targeted_perturb(best_pts, cur_idx, rng, scale=0.02)
        pert_area, _ = _min_area_and_idx(perturbed)
        
        if pert_area > best_area:
            best_pts = perturbed
            best_area = pert_area
            no_improve = 0
        else:
            no_improve += 1
        
        # Also try global small perturbation
        if no_improve > 5:
            global_pert = _perturb_points(best_pts, rng, scale=0.01)
            gp_area, _ = _min_area_and_idx(global_pert)
            if gp_area > best_area:
                best_pts = global_pert
                best_area = gp_area
                no_improve = 0
        
        # Periodically run Nelder-Mead to refine
        if no_improve > 10 and time.time() < deadline - 20:
            nm_pts = _nelder_mead_step(best_pts, rng, max_iter=200)
            nm_area, _ = _min_area_and_idx(nm_pts)
            if nm_area > best_area:
                best_pts = nm_pts
                best_area = nm_area
                no_improve = 0
            else:
                no_improve += 5  # penalize
    
    # Phase 3: Final refinement with Nelder-Mead
    if time.time() < deadline - 5:
        nm_pts = _nelder_mead_step(best_pts, rng, max_iter=500, tol=1e-12)
        nm_area, _ = _min_area_and_idx(nm_pts)
        if nm_area > best_area:
            best_pts = nm_pts
            best_area = nm_area
    
    # Final validation
    final_pts = _project_to_triangle(best_pts)
    
    # Verify all points are in the triangle
    x = final_pts[:, 0]
    y = final_pts[:, 1]
    # Check: y >= 0, y <= sqrt(3)*x, y <= -sqrt(3)*x + sqrt(3)
    if not (np.all(y >= -1e-9) and np.all(y <= SQRT3 * x + 1e-9) and 
            np.all(y <= -SQRT3 * x + SQRT3 + 1e-9) and
            np.all(x >= -1e-9) and np.all(x <= 1 + 1e-9)):
        # Fallback to a simple valid configuration
        bary = np.zeros((N, 2))
        for i in range(N):
            bary[i, 0] = (i + 1) / (N + 1) * 0.5
            bary[i, 1] = (i + 1) / (N + 1) * 0.5
        final_pts = _bary_to_cart(bary)
    
    return final_pts
