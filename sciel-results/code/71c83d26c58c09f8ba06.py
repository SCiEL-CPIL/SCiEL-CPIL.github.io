"""
Optimized sphere covering with 130 points via Grid-Based Smooth-Max L-BFGS.

Strategy:
1. Precompute a high-resolution icosahedral grid (approx 10,000 points) to discretize the sphere.
2. Define the objective as the smooth-max (log-sum-exp) of the squared distances from grid points to the nearest configuration point.
   This approximates the covering radius (max distance) but is differentiable.
3. Use scipy.optimize.minimize with L-BFGS-B to optimize the configuration points directly.
   L-BFGS-B handles bounds and is efficient for smooth functions.
4. Refine the result with a few steps of a local search that uses the exact convex hull covering radius to ensure the final result is as good as possible.
"""
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 130
_DEADLINE = time.time() + 140  # 10s safety margin

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms

def _time_ok(margin=5.0):
    return time.time() < _DEADLINE - margin

def _covering_radius(pts):
    """Compute the covering radius in radians via convex hull facet circumcenters."""
    pts = _normalize(pts)
    try:
        hull = ConvexHull(pts)
    except Exception:
        return np.pi / 2

    max_r = 0.0
    for simplex in hull.simplices:
        a, b, c = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
        v1, v2 = b - a, c - a
        cross = np.cross(v1, v2)
        cnorm = np.linalg.norm(cross)
        if cnorm < 1e-12:
            continue
        x = cross / cnorm
        centroid = (a + b + c) / 3.0
        if np.dot(x, centroid) < 0:
            x = -x
        dot_val = np.clip(np.dot(x, a), -1.0, 1.0)
        r = np.arccos(dot_val)
        if r > max_r:
            max_r = r
    return max_r

def _covering_radius_fast(pts):
    """Alias for _covering_radius."""
    return _covering_radius(pts)

def _generate_icosahedron_grid(subdivisions=4):
    """
    Generate a quasi-uniform grid on the sphere using icosahedron subdivision.
    subdivisions=4 gives 20 * 4^2 = 320 faces, 162 vertices? No.
    Icosahedron has 12 vertices, 20 faces.
    Level 1: 42 vertices, 80 faces.
    Level 2: 162 vertices, 320 faces.
    Level 3: 642 vertices, 1280 faces.
    Level 4: 2562 vertices, 5120 faces.
    Level 5: 10242 vertices, 20480 faces.
    
    We want ~10k-20k points for good resolution. Level 5 is ~10k.
    """
    # Golden ratio
    phi = (1 + np.sqrt(5)) / 2
    
    # 12 vertices of icosahedron
    v = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ])
    v = v / np.linalg.norm(v, axis=1, keepdims=True)
    
    # Faces (indices)
    f = [
        (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
        (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
        (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
        (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)
    ]
    
    # Subdivide
    for _ in range(subdivisions):
        new_f = []
        mid_cache = {}
        
        def get_mid(i1, i2):
            key = (min(i1, i2), max(i1, i2))
            if key not in mid_cache:
                mid = (v[i1] + v[i2]) / 2.0
                mid = mid / np.linalg.norm(mid)
                v = np.vstack([v, mid])
                mid_cache[key] = len(v) - 1
            return mid_cache[key]
        
        for face in f:
            a, b, c = face
            ab = get_mid(a, b)
            bc = get_mid(b, c)
            ca = get_mid(c, a)
            
            new_f.append((a, ab, ca))
            new_f.append((b, bc, ab))
            new_f.append((c, ca, bc))
            new_f.append((ab, bc, ca))
            
        f = new_f
    
    # Centroid of each face is a good grid point
    grid = np.array([(v[f[i][0]] + v[f[i][1]] + v[f[i][2]]) / 3.0 for i in range(len(f))])
    grid = grid / np.linalg.norm(grid, axis=1, keepdims=True)
    return grid

def _smooth_max_objective(x, grid, beta=50.0):
    """
    Objective: smooth max of squared distances from grid to nearest point.
    x: (3N,) flat array of point coordinates.
    grid: (M, 3) grid points.
    """
    pts = x.reshape(N, 3)
    pts = _normalize(pts)
    
    # Compute squared distances from grid to all points
    # grid: (M, 3), pts: (N, 3)
    # dist_sq[m, n] = ||grid[m] - pts[n]||^2
    # = ||grid[m]||^2 - 2 grid[m] pts[n]^T + ||pts[n]||^2
    # Since normalized, = 2 - 2 grid @ pts.T
    dist_sq = 2.0 - 2.0 * (grid @ pts.T)
    
    # Min distance for each grid point
    min_dist_sq = np.min(dist_sq, axis=1)
    
    # Smooth max of min_dist_sq? 
    # We want to minimize the MAX distance. 
    # max(d) approx log(sum(exp(beta * d))) / beta.
    # Here d is squared distance. Minimizing max squared dist is equivalent to minimizing max dist.
    # Let's use log-sum-exp on the min_dist_sq values.
    # Actually, we want the worst-case grid point.
    # max_m (min_n dist_sq[m,n])
    # This is a max-min problem.
    # Let y_m = min_n dist_sq[m,n]. We want to minimize max(y).
    # Smooth approx: (1/beta) * log(sum(exp(beta * y_m)))
    
    # Numerical stability
    y = min_dist_sq
    max_y = np.max(y)
    if max_y > 1e6: # Should not happen
        return max_y
    exp_y = np.exp(beta * (y - max_y))
    log_sum_exp = np.log(np.sum(exp_y)) + beta * max_y
    val = log_sum_exp / beta
    
    # Gradient
    # d(val)/d(pts)
    # Let S = sum(exp(beta * y_m))
    # val = (1/beta) log S
    # d(val)/d(pts) = (1/S) sum_m exp(beta * y_m) * dy_m/d(pts)
    
    # y_m = min_n dist_sq[m,n]
    # dy_m/d(pts) is non-zero only for the n that minimizes dist_sq[m,n].
    # Let n_m be argmin_n dist_sq[m,n]
    # dist_sq[m,n] = 2 - 2 grid[m] . pts[n]
    # d(dist_sq[m,n])/d(pts[n]) = -2 grid[m]
    # d(dist_sq[m,n])/d(pts[k]) = 0 for k != n
    
    # So dy_m/d(pts) is a vector of length 3N.
    # The component corresponding to pts[n_m] is -2 grid[m].
    
    # Gradient = (1/S) sum_m exp(beta * y_m) * (-2 grid[m]) at index n_m
    
    grad = np.zeros_like(pts)
    n_m = np.argmin(dist_sq, axis=1)
    
    # Weights
    weights = exp_y / np.sum(exp_y) # Sum to 1
    
    # Accumulate gradient
    # For each m, add weights[m] * (-2 grid[m]) to grad[n_m]
    # Use np.add.at for scatter add
    np.add.at(grad, n_m, -2.0 * weights[:, np.newaxis] * grid)
    
    return val, grad.flatten()

def _lloyd_step(pts, grid):
    """
    Lloyd's algorithm step: Move each point to the centroid of its Voronoi cell (approximated by grid points).
    """
    pts = _normalize(pts.copy())
    
    # Assign grid points to nearest config point
    dist_sq = 2.0 - 2.0 * (grid @ pts.T)
    labels = np.argmin(dist_sq, axis=1)
    
    new_pts = np.zeros_like(pts)
    counts = np.zeros(N)
    
    # Scatter add grid points to their assigned config point
    np.add.at(new_pts, labels, grid)
    np.add.at(counts, labels, 1)
    
    # Average
    valid = counts > 0
    new_pts[valid] /= counts[valid, np.newaxis]
    
    # For points with no grid points (shouldn't happen with fine grid), keep old
    new_pts[~valid] = pts[~valid]
    
    return _normalize(new_pts)

def construct():
    """
    1. Generate initial configs.
    2. Precompute icosahedral grid.
    3. Run L-BFGS-B optimization on smooth-max objective.
    4. Refine with Lloyd's algorithm and final hull-based check.
    """
    start_time = time.time()
    
    # Precompute grid (Level 5 ~ 10k points, Level 4 ~ 2.5k)
    # Level 5 is good for accuracy. Let's use 5.
    grid = _generate_icosahedron_grid(subdivisions=5)
    
    best_pts = None
    best_r = np.pi
    
    # Initial configs
    inits = []
    for seed in [42, 123, 7, 999, 31415]:
        # Fibonacci spiral
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(np.maximum(0, 1.0 - z * z))
        pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        inits.append(_normalize(pts))
        
    # Add random
    rng = np.random.RandomState(555)
    inits.append(_normalize(rng.randn(N, 3)))
    
    # Sort by initial covering radius
    init_scores = []
    for p in inits:
        try:
            r = _covering_radius_fast(p)
        except Exception:
            r = np.pi
        init_scores.append((r, p))
    init_scores.sort(key=lambda x: x[0])
    
    # Optimization Phase
    for r0, p0 in init_scores:
        if not _time_ok(10.0):
            break
            
        x0 = p0.flatten()
        
        # L-BFGS-B
        try:
            res = minimize(
                _smooth_max_objective,
                x0,
                args=(grid, 50.0),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 1000, 'maxfun': 10000, 'ftol': 1e-12, 'gtol': 1e-12}
            )
        except Exception:
            continue
            
        if not res.success and not res.fun < np.pi:
            # Check if it improved anyway
            pass
            
        pts_opt = _normalize(res.x.reshape(N, 3))
        
        # Evaluate
        cr = _covering_radius_fast(pts_opt)
        if cr < best_r:
            best_r = cr
            best_pts = pts_opt.copy()
            
        # Check time
        if not _time_ok(15.0):
            break
            
    # Refinement Phase: Lloyd's Algorithm on the best candidate
    if best_pts is not None and _time_ok(10.0):
        pts = best_pts.copy()
        for _ in range(20):
            if not _time_ok(5.0):
                break
            pts = _lloyd_step(pts, grid)
            cr = _covering_radius_fast(pts)
            if cr < best_r:
                best_r = cr
                best_pts = pts.copy()
                
    # Final Polish: Small L-BFGS step on best
    if best_pts is not None and _time_ok(5.0):
        x0 = best_pts.flatten()
        try:
            res = minimize(
                _smooth_max_objective,
                x0,
                args=(grid, 100.0),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 50, 'maxfun': 500, 'ftol': 1e-14, 'gtol': 1e-14}
            )
            pts_polish = _normalize(res.x.reshape(N, 3))
            cr = _covering_radius_fast(pts_polish)
            if cr < best_r:
                best_r = cr
                best_pts = pts_polish.copy()
        except Exception:
            pass
            
    if best_pts is None:
        best_pts = inits[0]
        
    return _normalize(best_pts)

# Test
if __name__ == "__main__":
    t0 = time.time()
    pts = construct()
    print("Shape:", pts.shape)
    print("Covering radius (deg):", np.degrees(_covering_radius_fast(pts)))
    print("Time: %.1f s" % (time.time() - t0))
