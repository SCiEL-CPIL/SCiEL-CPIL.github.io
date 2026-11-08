import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull

N = 14
DEADLINE = 145.0  # Safety margin before 150s kill

def _hull_area(pts):
    """Compute convex hull area."""
    try:
        if len(pts) < 3:
            return 0.0
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0

def _min_triangle_area_vec(points):
    """Vectorized minimum triangle area for N points."""
    n = len(points)
    min_area = float('inf')
    x = points[:, 0]
    y = points[:, 1]
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            dx_ki = x[j+1:] - x[i]
            dy_ki = y[j+1:] - y[i]
            cross = (x[j] - x[i]) * dy_ki - dx_ki * (y[j] - y[i])
            local_min = 0.5 * np.min(np.abs(cross))
            if local_min < min_area:
                min_area = local_min
                if min_area < 1e-14:
                    return min_area
    return min_area

def _score(pts):
    """Compute min triangle area / hull area."""
    hull_a = _hull_area(pts)
    if hull_a <= 0:
        return 0.0
    min_area = _min_triangle_area_vec(pts)
    return min_area / hull_a

def _neg_score(pts_flat):
    """Negative score for minimization."""
    pts = pts_flat.reshape(-1, 2)
    s = _score(pts)
    return -s

def _polar_to_cartesian(params):
    """
    Convert polar parameters to Cartesian points.
    params: [r_0, ..., r_{N-1}, theta_0, d_theta_0, ..., d_theta_{N-2}]
    Total: 2N - 1 parameters.
    theta_k = theta_0 + cumsum(d_theta_0..d_theta_{k-1})
    We enforce d_theta_k > 0 so angles are strictly increasing.
    We also normalize the points to have unit hull area at the end? 
    No, the score normalizes by hull area. 
    However, scaling affects the absolute areas but the ratio is scale-invariant.
    So we can fix the scale, e.g., mean radius = 1, or just let DE find the scale.
    To make DE more efficient, let's fix the scale by normalizing the output.
    """
    n = N
    radii = np.abs(params[:n]) + 0.1  # ensure positive
    theta_start = params[n]
    d_thetas = np.abs(params[n+1:n+n-1]) + 0.05  # ensure positive gaps
    
    thetas = np.zeros(n)
    thetas[0] = theta_start
    if n > 1:
        thetas[1:] = theta_start + np.cumsum(d_thetas)
    
    # Normalize angles to be within [0, 2*pi) for numerical stability? 
    # Not strictly necessary for cos/sin, but helps if values get huge.
    # Let's just center them.
    thetas = thetas - np.mean(thetas)
    
    x = radii * np.cos(thetas)
    y = radii * np.sin(thetas)
    pts = np.column_stack([x, y])
    
    # Normalize to unit hull area to keep values in a reasonable range for DE
    hull_a = _hull_area(pts)
    if hull_a > 0:
        pts = pts / np.sqrt(hull_a)
        
    return pts

def _de_obj(params):
    """Objective for Differential Evolution."""
    pts = _polar_to_cartesian(params)
    if np.any(~np.isfinite(pts)):
        return 1e10
    s = _score(pts)
    if s < 0:
        return 1e10
    return -s

def _local_refine(points, time_budget):
    """Local refinement using Nelder-Mead on full point coordinates."""
    start_t = time.time()
    x0 = points.flatten()
    
    def neg_score_local(x):
        if time.time() - start_t > time_budget:
            return 1e10
        pts = x.reshape(-1, 2)
        return -_score(pts)
    
    try:
        res = minimize(neg_score_local, x0, method='Nelder-Mead',
                      options={'maxiter': 2000, 'xatol': 1e-9, 'fatol': 1e-11,
                              'adaptive': True, 'maxfev': 5000})
        if res.x is not None:
            return res.x.reshape(-1, 2)
    except Exception:
        pass
    
    return points

def _initial_regular_polygon(n):
    """Points on a regular n-gon inscribed in unit circle, scaled to unit area."""
    angles = 2 * np.pi * np.arange(n) / n
    pts = np.column_stack([np.cos(angles), np.sin(angles)])
    return _polar_to_cartesian(np.concatenate([np.ones(n), [0.0] + np.full(n-1, 2*np.pi/n)]))

def _initial_elliptical(n, ex=1.2, ey=0.8):
    """Points on an ellipse."""
    angles = 2 * np.pi * np.arange(n) / n
    pts = np.column_stack([ex * np.cos(angles), ey * np.sin(angles)])
    # Convert to polar params for consistency? 
    # Or just normalize area.
    hull_a = _hull_area(pts)
    if hull_a > 0:
        pts = pts / np.sqrt(hull_a)
    return pts

def construct():
    start_time = time.time()
    
    # Generate diverse initial configurations
    inits = [
        _initial_regular_polygon(N),
        _initial_elliptical(N, 1.2, 0.8),
        _initial_elliptical(N, 0.8, 1.2),
        _initial_elliptical(N, 1.5, 0.6),
        _initial_elliptical(N, 0.6, 1.5),
    ]
    
    # Add some random convex perturbations
    rng = np.random.RandomState(42)
    for _ in range(3):
        angles = 2 * np.pi * np.arange(N) / N + rng.normal(0, 0.1, N)
        radii = 1.0 + rng.normal(0, 0.1, N)
        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        hull_a = _hull_area(pts)
        if hull_a > 0:
            pts = pts / np.sqrt(hull_a)
        inits.append(pts)
        
    # Evaluate initial scores
    init_scores = [_score(p) for p in inits]
    best_init_idx = np.argmax(init_scores)
    best_pts = inits[best_init_idx]
    best_score = init_scores[best_init_idx]
    
    # Time allocation
    total_time = DEADLINE - (time.time() - start_time)
    
    # Phase 1: Global Search via Differential Evolution in polar space
    # We have 2N-1 = 27 parameters.
    # Bounds:
    # Radii: [0.1, 3.0]
    # Theta start: [0, 2*pi]
    # D_thetas: [0.05, 2*pi] (must sum to ~2*pi, but DE doesn't know that constraint directly.
    # However, the normalization in _polar_to_cartesian handles scaling.
    # If d_thetas sum to much more or less than 2*pi, the points will still be on a circle/ellipse-like shape,
    # just with different angular spacing. The normalization keeps the size fixed.
    # So we can let d_thetas vary freely in a reasonable range.
    
    n_params = 2 * N - 1
    bounds = [(0.1, 3.0)] * N + [(0, 2*np.pi)] + [(0.05, 2*np.pi)] * (N - 1)
    
    de_time_budget = total_time * 0.5
    local_time_budget = total_time * 0.4
    reserve = total_time * 0.1
    
    if time.time() - start_time < DEADLINE - local_time_budget - reserve:
        try:
            # Use Sobol initialization for better coverage
            res_de = differential_evolution(
                _de_obj, bounds,
                seed=42, maxiter=100, popsize=15,
                tol=1e-6, mutation=(0.5, 1.0), recombination=0.7,
                polish=False, init='sobol',
                updating='immediate',
                callback=None
            )
            if res_de.x is not None:
                de_pts = _polar_to_cartesian(res_de.x)
                s_de = _score(de_pts)
                if s_de > best_score:
                    best_score = s_de
                    best_pts = de_pts
        except Exception:
            pass

    # Phase 2: Local Refinement
    # Refine the best found so far
    remaining = DEADLINE - 5 - (time.time() - start_time)
    if remaining > 5:
        try:
            refined_pts = _local_refine(best_pts, remaining * 0.7)
            s_ref = _score(refined_pts)
            if s_ref > best_score:
                best_score = s_ref
                best_pts = refined_pts
        except Exception:
            pass
            
    # Phase 3: Refine top initializations if time permits
    remaining = DEADLINE - 5 - (time.time() - start_time)
    if remaining > 10:
        # Sort inits by score
        sorted_inits = [inits[i] for i in np.argsort(init_scores)[::-1]]
        for init in sorted_inits[:3]:
            if time.time() - start_time > DEADLINE - 5:
                break
            time_left = DEADLINE - 5 - (time.time() - start_time)
            if time_left < 5:
                break
            try:
                refined_init = _local_refine(init, time_left * 0.8)
                s_init_ref = _score(refined_init)
                if s_init_ref > best_score:
                    best_score = s_init_ref
                    best_pts = refined_init
            except Exception:
                pass
                
    return best_pts
