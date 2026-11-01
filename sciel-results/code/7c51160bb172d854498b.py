import numpy as np
import scipy.spatial
import scipy.optimize
import time

N = 14

# Pre-compute triangle indices for speed
def _get_triangle_indices(n):
    """Pre-compute all (i,j,k) with i<j<k"""
    indices = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                indices.append((i, j, k))
    return np.array(indices, dtype=np.int32)

_TRI_INDICES = _get_triangle_indices(N)
_I = _TRI_INDICES[:, 0]
_J = _TRI_INDICES[:, 1]
_K = _TRI_INDICES[:, 2]

def _triangle_areas(points):
    """Compute all triangle areas (unnormalized, 0.5*|cross|)."""
    pi = points[_I]
    pj = points[_J]
    pk = points[_K]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross)

def _min_triangle_area(points):
    """Compute minimum triangle area for given points."""
    return _triangle_areas(points).min()

def _hull_area(points):
    """Compute convex hull area."""
    try:
        hull = scipy.spatial.ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def calc_ratio(points):
    """
    Calculate the score: min triangle area / convex hull area.
    points: (N, 2) array
    """
    try:
        if len(points) < 3:
            return 0.0
        hull_area = _hull_area(points)
        if hull_area < 1e-10:
            return 0.0
        min_area = _min_triangle_area(points)
        return min_area / hull_area
    except Exception:
        return 0.0

def _soft_min(values, tau):
    """
    Soft-min using log-sum-exp trick.
    As tau -> 0, approaches true min.
    """
    values = np.asarray(values)
    # For stability, subtract max before exp
    v_max = values.max()
    if v_max < 1e-15:
        return 0.0
    # soft_min = -tau * log(sum(exp(-values/tau)))
    # = -tau * log(sum(exp(-(values - v_max)/tau))) - v_max
    # = -tau * log(sum(exp(-(values - v_max)/tau))) - v_max
    scaled = -(values - v_max) / tau
    # Clip to avoid overflow
    scaled = np.clip(scaled, -500, 500)
    log_sum = np.log(np.sum(np.exp(scaled)))
    return v_max - tau * log_sum

def _neg_smooth_score(points_flat, tau):
    """
    Smooth surrogate objective: minimize -soft_min(triangle_areas) / hull_area.
    This provides gradient flow for L-BFGS-B.
    """
    points = points_flat.reshape(-1, 2)
    try:
        areas = _triangle_areas(points)
        hull_area = _hull_area(points)
        if hull_area < 1e-10:
            return 1.0
        soft_min_val = _soft_min(areas, tau)
        score = soft_min_val / hull_area
        return -score
    except Exception:
        return 1.0

def _neg_score(points_flat):
    """Exact objective for final evaluation."""
    points = points_flat.reshape(-1, 2)
    score = calc_ratio(points)
    return -score

def normalize_points(points):
    """
    Scale points so that the convex hull area is 1.0.
    """
    try:
        area = _hull_area(points)
        if area < 1e-10:
            return points
        scale_factor = 1.0 / np.sqrt(area)
        return points * scale_factor
    except Exception:
        return points

def _coordinate_descent_polish(points, max_iter=200, tol=1e-10):
    """
    Coordinate descent polish: for the bottleneck triangle, adjust points
    to increase the minimum area.
    """
    points = points.copy()
    n = len(points)
    
    for iteration in range(max_iter):
        areas = _triangle_areas(points)
        min_idx = np.argmin(areas)
        min_area = areas[min_idx]
        
        # Get the three points in the bottleneck triangle
        i, j, k = _TRI_INDICES[min_idx]
        pi = points[i]
        pj = points[j]
        pk = points[k]
        
        # For each of the three points, try small perturbations
        # to increase the area of this triangle
        best_improvement = 0.0
        best_new_points = points
        
        for pt_idx in [i, j, k]:
            # Try moving this point in several directions
            for dx in [-0.01, -0.005, 0.005, 0.01]:
                for dy in [-0.01, -0.005, 0.005, 0.01]:
                    new_points = points.copy()
                    new_points[pt_idx] = points[pt_idx] + np.array([dx, dy])
                    new_areas = _triangle_areas(new_points)
                    new_min = new_areas.min()
                    improvement = new_min - min_area
                    if improvement > best_improvement:
                        best_improvement = improvement
                        best_new_points = new_points
        
        if best_improvement < tol:
            break
        
        points = best_new_points
    
    return points

def _regular_polygon(n, radius=1.0):
    """Generate regular polygon points."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])

def _concentric_circles(n, n_outer=7, n_inner=7, r_inner=0.5):
    """Generate points on two concentric circles."""
    angles_outer = np.linspace(0, 2 * np.pi, n_outer, endpoint=False)
    angles_inner = np.linspace(0, 2 * np.pi, n_inner, endpoint=False) + np.pi / n_inner
    outer = np.column_stack([np.cos(angles_outer), np.sin(angles_outer)])
    inner = r_inner * np.column_stack([np.cos(angles_inner), np.sin(angles_inner)])
    return np.vstack([outer, inner])

def _perturbed_polygon(n, radius=1.0, noise_scale=0.05, seed=42):
    """Generate perturbed regular polygon."""
    rng = np.random.RandomState(seed)
    base = _regular_polygon(n, radius)
    noise = rng.uniform(-noise_scale, noise_scale, (n, 2))
    return base + noise

def _ellipse_config(n, a=1.0, b=0.7, seed=42):
    """Generate points on an ellipse."""
    rng = np.random.RandomState(seed)
    angles = np.sort(rng.uniform(0, 2 * np.pi, n))
    # Add some jitter
    angles += rng.uniform(-0.05, 0.05, n)
    return np.column_stack([a * np.cos(angles), b * np.sin(angles)])

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    best_score = 0.0
    best_points = None
    
    # List of initial configurations to try
    inits = []
    
    # 1. Regular polygon
    inits.append(_regular_polygon(N, 1.0))
    
    # 2. Perturbed regular polygons with different noise scales
    for noise_scale in [0.02, 0.05, 0.1, 0.15]:
        inits.append(_perturbed_polygon(N, 1.0, noise_scale, seed=42+int(noise_scale*100)))
    
    # 3. Concentric circles
    inits.append(_concentric_circles(N, 7, 7, 0.5))
    inits.append(_concentric_circles(N, 7, 7, 0.4))
    inits.append(_concentric_circles(N, 8, 6, 0.5))
    inits.append(_concentric_circles(N, 6, 8, 0.5))
    
    # 4. Random points in unit disk
    rng = np.random.RandomState(123)
    for _ in range(5):
        r = np.sqrt(rng.uniform(0, 1, N))
        theta = rng.uniform(0, 2 * np.pi, N)
        random_pts = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        inits.append(random_pts)
    
    # 5. Grid-like arrangement
    x = np.linspace(-0.5, 0.5, 4)
    y = np.linspace(-0.5, 0.5, 4)
    gx, gy = np.meshgrid(x, y)
    grid_pts = np.column_stack([gx.flatten(), gy.flatten()])
    grid_pts = grid_pts[:N]
    inits.append(grid_pts)
    
    # 6. Ellipse configurations
    inits.append(_ellipse_config(N, 1.0, 0.7, seed=42))
    inits.append(_ellipse_config(N, 1.0, 0.8, seed=43))
    inits.append(_ellipse_config(N, 1.2, 0.6, seed=44))
    
    # Phase 1: Smooth optimization with annealed tau
    # Use soft-min surrogate with decreasing temperature
    tau_schedule = [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01]
    
    for idx_init, current_init in enumerate(inits):
        if time.time() > deadline:
            break
            
        current_init = normalize_points(current_init)
        
        # Start with smooth objective, anneal tau
        for tau in tau_schedule:
            if time.time() > deadline:
                break
            
            time_left = deadline - time.time()
            if time_left < 5:
                break
            
            try:
                # Use L-BFGS-B with smooth objective
                maxiter = int(max(30, time_left * 8))
                options = {'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-10}
                
                res = scipy.optimize.minimize(
                    lambda x: _neg_smooth_score(x, tau),
                    current_init.flatten(),
                    method='L-BFGS-B',
                    options=options
                )
                
                optimized_points = res.x.reshape(-1, 2)
                optimized_points = normalize_points(optimized_points)
                
                score = calc_ratio(optimized_points)
                
                if score > best_score:
                    best_score = score
                    best_points = optimized_points.copy()
                
                # Update current_init for next tau level
                current_init = optimized_points
                
            except Exception:
                pass
    
    # Phase 2: Exact optimization with Powell and Nelder-Mead
    if best_points is not None:
        current_init = best_points
    else:
        current_init = normalize_points(_regular_polygon(N, 1.0))
    
    for method in ['Powell', 'Nelder-Mead']:
        if time.time() > deadline:
            break
            
        time_left = deadline - time.time()
        if time_left < 10:
            break
        
        try:
            if method == 'Powell':
                maxiter = int(max(50, time_left * 5))
                options = {'maxiter': maxiter, 'ftol': 1e-12, 'xtol': 1e-12}
            else:
                maxiter = int(max(100, time_left * 20))
                options = {'maxiter': maxiter, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            
            res = scipy.optimize.minimize(
                _neg_score,
                current_init.flatten(),
                method=method,
                options=options
            )
            
            optimized_points = res.x.reshape(-1, 2)
            optimized_points = normalize_points(optimized_points)
            
            score = calc_ratio(optimized_points)
            
            if score > best_score:
                best_score = score
                best_points = optimized_points.copy()
                current_init = optimized_points
                
        except Exception:
            pass
    
    # Phase 3: Coordinate descent polish
    if best_points is not None and time.time() < deadline - 5:
        polished = _coordinate_descent_polish(best_points, max_iter=100, tol=1e-12)
        polished = normalize_points(polished)
        score = calc_ratio(polished)
        if score > best_score:
            best_score = score
            best_points = polished
    
    # Final normalization
    if best_points is None:
        best_points = normalize_points(_regular_polygon(N, 1.0))
    else:
        best_points = normalize_points(best_points)
        
    return best_points
