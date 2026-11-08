"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist
import itertools

DEADLINE = 135.0  # Safety margin before 150s kill


def _hull_area(pts):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0


def _all_triangle_areas(points):
    """Return array of all triangle areas (N choose 3)."""
    n = len(points)
    x = points[:, 0]
    y = points[:, 1]
    areas_list = []
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            dx_ki = x[j+1:] - x[i]
            dy_ki = y[j+1:] - y[i]
            cross = (x[j] - x[i]) * dy_ki - dx_ki * (y[j] - y[i])
            areas_list.append(0.5 * np.abs(cross))
    if not areas_list:
        return np.array([0.0])
    return np.concatenate(areas_list)


def _min_triangle_area(points):
    """Compute minimum triangle area."""
    areas = _all_triangle_areas(points)
    return float(np.min(areas)) if len(areas) > 0 else 0.0


def _score(pts):
    """Compute min triangle area / hull area."""
    hull_a = _hull_area(pts)
    if hull_a <= 1e-15:
        return 0.0
    min_area = _min_triangle_area(pts)
    return min_area / hull_a


def _soft_min(areas, beta):
    """Soft minimum via log-sum-exp."""
    if len(areas) == 0:
        return 0.0
    z = -beta * areas
    max_z = np.max(z)
    if max_z < -500:
        log_sum = max_z
    else:
        log_sum = max_z + np.log(np.sum(np.exp(z - max_z)))
    return -log_sum / beta


def _polar_to_cartesian(radii, angles):
    """Convert polar coordinates to Cartesian, ensuring sorted angles."""
    # Sort by angle to maintain convex ordering
    order = np.argsort(angles)
    r_sorted = radii[order]
    a_sorted = angles[order]
    x = r_sorted * np.cos(a_sorted)
    y = r_sorted * np.sin(a_sorted)
    return np.column_stack([x, y])


def _soft_obj_polar(params, beta=100.0):
    """
    Soft objective in polar coordinates.
    params: [r1, r2, ..., rN, a1, a2, ..., aN]
    Returns: hull_area / soft_min(triangle_areas)
    """
    n = N
    radii = params[:n]
    angles = params[n:]
    
    # Ensure radii are positive
    radii = np.maximum(radii, 0.01)
    
    pts = _polar_to_cartesian(radii, angles)
    
    if np.any(~np.isfinite(pts)):
        return 1e10
    
    areas = _all_triangle_areas(pts)
    if len(areas) == 0 or np.min(areas) < 1e-14:
        return 1e10
    
    hull_area = _hull_area(pts)
    if hull_area < 1e-10:
        return 1e10
    
    sm = _soft_min(areas, beta)
    if sm < 1e-14:
        return 1e10
    
    return hull_area / sm


def _neg_score_polar(params, beta=100.0):
    """Negative soft score for minimization in polar coords."""
    return _soft_obj_polar(params, beta)


def _hard_obj_polar(params):
    """Hard objective: negative of score."""
    n = N
    radii = params[:n]
    angles = params[n:]
    radii = np.maximum(radii, 0.01)
    pts = _polar_to_cartesian(radii, angles)
    if np.any(~np.isfinite(pts)):
        return 1e10
    s = _score(pts)
    return -s


def _epigraph_obj_polar(params, beta=50.0):
    """
    Epigraph-style smooth objective.
    Maximizes t subject to t <= min(areas) / hull_area.
    We use: minimize hull_area / soft_min(areas, beta)
    with a penalty if points are too close.
    """
    n = N
    radii = params[:n]
    angles = params[n:]
    radii = np.maximum(radii, 0.01)
    pts = _polar_to_cartesian(radii, angles)
    
    if np.any(~np.isfinite(pts)):
        return 1e10
    
    # Penalty for very close points
    dmin = np.min(pdist(pts))
    if dmin < 0.005:
        return 1e10 + (0.005 - dmin) * 10000
    
    areas = _all_triangle_areas(pts)
    if len(areas) == 0 or np.min(areas) < 1e-14:
        return 1e10
    
    hull_area = _hull_area(pts)
    if hull_area < 1e-10:
        return 1e10
    
    sm = _soft_min(areas, beta)
    if sm < 1e-14:
        return 1e10
    
    return hull_area / sm


def _generate_initial_polar(rng, n=N):
    """Generate a random initial polar configuration."""
    # Radii: mix of inner and outer
    radii = np.ones(n) * rng.uniform(0.5, 1.5, size=n)
    # Angles: roughly uniform with some perturbation
    base_angles = 2 * np.pi * np.arange(n) / n
    angles = base_angles + rng.normal(0, 0.15, size=n)
    angles = np.sort(angles)
    return np.concatenate([radii, angles])


def _regular_polar(n=N):
    """Regular polygon in polar coordinates."""
    radii = np.ones(n)
    angles = 2 * np.pi * np.arange(n) / n
    return np.concatenate([radii, angles])


def _two_ring_polar(n1, n2, r1, r2, offset=0.0):
    """Two-ring configuration in polar coordinates."""
    n = n1 + n2
    radii = np.concatenate([np.ones(n1) * r1, np.ones(n2) * r2])
    angles1 = 2 * np.pi * np.arange(n1) / n1
    angles2 = 2 * np.pi * np.arange(n2) / n2 + np.pi / n2 + offset
    angles = np.sort(np.concatenate([angles1, angles2]))
    return np.concatenate([radii, angles])


def _optimize_polar(x0, method='L-BFGS-B', beta=100.0, maxiter=2000, ftol=1e-12):
    """Optimize in polar space."""
    n = N
    # Bounds: radii in [0.1, 3.0], angles in [-pi, pi] (will be sorted)
    bounds = [(0.1, 3.0)] * n + [(-np.pi, np.pi)] * n
    
    if method == 'L-BFGS-B':
        res = minimize(_soft_obj_polar, x0, args=(beta,), method='L-BFGS-B',
                      bounds=bounds, options={'maxiter': maxiter, 'ftol': ftol, 'gtol': 1e-8})
    elif method == 'Nelder-Mead':
        res = minimize(_soft_obj_polar, x0, args=(beta,), method='Nelder-Mead',
                      options={'maxiter': maxiter, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
    else:
        return x0
    
    if res.x is not None:
        return res.x
    return x0


def _polar_params_to_pts(params):
    """Convert polar params to point array."""
    n = N
    radii = np.maximum(params[:n], 0.01)
    angles = params[n:]
    return _polar_to_cartesian(radii, angles)


def construct():
    start_time = time.time()
    
    best_pts = None
    best_score = 0.0
    best_polar = None
    
    rng = np.random.RandomState(42)
    
    # Phase 1: Differential Evolution in polar space
    if time.time() - start_time < DEADLINE - 40:
        try:
            n = N
            bounds = [(0.1, 3.0)] * n + [(-np.pi, np.pi)] * n
            
            de_res = differential_evolution(
                _soft_obj_polar,
                bounds,
                args=(80.0,),
                seed=42,
                maxiter=60,
                popsize=20,
                tol=1e-5,
                mutation=(0.4, 1.0),
                recombination=0.7,
                polish=False,
                init='sobol',
                update_interval=1
            )
            
            if de_res.x is not None:
                best_polar = de_res.x.copy()
                cand = _polar_params_to_pts(de_res.x)
                s = _score(cand)
                if s > best_score:
                    best_score = s
                    best_pts = cand
        except Exception:
            pass
    
    # Phase 2: Local refinement from multiple polar starting points
    polar_inits = []
    
    # Regular polygon
    polar_inits.append(_regular_polar(N))
    
    # Two-ring configurations
    for n1, n2, r1, r2 in [(7, 7, 0.6, 1.2), (8, 6, 0.5, 1.3), (6, 8, 0.7, 1.1), (9, 5, 0.4, 1.4)]:
        polar_inits.append(_two_ring_polar(n1, n2, r1, r2))
    
    # Random initializations
    for _ in range(8):
        polar_inits.append(_generate_initial_polar(rng))
    
    # Add DE result if available
    if best_polar is not None:
        polar_inits.insert(0, best_polar.copy())
    
    for pi, x0 in enumerate(polar_inits):
        if time.time() - start_time > DEADLINE - 25:
            break
        
        # L-BFGS-B with moderate beta
        try:
            x_opt = _optimize_polar(x0, method='L-BFGS-B', beta=80.0, maxiter=1500, ftol=1e-12)
            cand = _polar_params_to_pts(x_opt)
            s = _score(cand)
            if s > best_score:
                best_score = s
                best_pts = cand
                best_polar = x_opt.copy()
        except Exception:
            pass
        
        # Nelder-Mead with higher beta for sharper min
        if time.time() - start_time > DEADLINE - 25:
            break
        try:
            x_opt = _optimize_polar(x0, method='Nelder-Mead', beta=150.0, maxiter=3000, ftol=1e-10)
            cand = _polar_params_to_pts(x_opt)
            s = _score(cand)
            if s > best_score:
                best_score = s
                best_pts = cand
                best_polar = x_opt.copy()
        except Exception:
            pass
    
    # Phase 3: High-precision polish in Cartesian space from best polar result
    if best_pts is not None and time.time() - start_time < DEADLINE - 15:
        try:
            x0_cart = best_pts.flatten()
            
            # L-BFGS-B with high beta
            def _soft_obj_cart(pts_flat, beta):
                points = pts_flat.reshape(N, 2)
                if np.any(~np.isfinite(points)):
                    return 1e10
                areas = _all_triangle_areas(points)
                if len(areas) == 0 or np.min(areas) < 1e-14:
                    return 1e10
                hull_area = _hull_area(points)
                if hull_area < 1e-10:
                    return 1e10
                sm = _soft_min(areas, beta)
                if sm < 1e-14:
                    return 1e10
                return hull_area / sm
            
            res = minimize(_soft_obj_cart, x0_cart, args=(200.0,), method='L-BFGS-B',
                          options={'maxiter': 3000, 'ftol': 1e-14, 'gtol': 1e-10})
            if res.x is not None:
                cand = res.x.reshape(-1, 2)
                s = _score(cand)
                if s > best_score:
                    best_score = s
                    best_pts = cand
        except Exception:
            pass
    
    # Phase 4: Final Nelder-Mead on hard objective
    if best_pts is not None and time.time() - start_time < DEADLINE - 8:
        try:
            x0 = best_pts.flatten()
            res = minimize(_hard_obj_polar, x0, method='Nelder-Mead',
                          options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
            if res.x is not None:
                # This was in polar space, convert back
                cand = _polar_params_to_pts(res.x)
                s = _score(cand)
                if s > best_score:
                    best_score = s
                    best_pts = cand
        except Exception:
            pass
    
    # Phase 5: Direct Cartesian Nelder-Mead polish
    if best_pts is not None and time.time() - start_time < DEADLINE - 5:
        try:
            def _neg_score_cart(pts_flat):
                points = pts_flat.reshape(N, 2)
                if np.any(~np.isfinite(points)):
                    return 1e10
                s = _score(points)
                return -s
            
            x0 = best_pts.flatten()
            res = minimize(_neg_score_cart, x0, method='Nelder-Mead',
                          options={'maxiter': 8000, 'xatol': 1e-11, 'fatol': 1e-13, 'adaptive': True})
            if res.x is not None:
                cand = res.x.reshape(-1, 2)
                s = _score(cand)
                if s > best_score:
                    best_score = s
                    best_pts = cand
        except Exception:
            pass
    
    # Phase 6: Multi-start local search with small perturbations
    if best_pts is not None and time.time() - start_time < DEADLINE - 3:
        try:
            for _ in range(3):
                if time.time() - start_time > DEADLINE - 3:
                    break
                perturbation = rng.normal(0, 0.05, size=best_pts.shape)
                perturbed = best_pts + perturbation
                # Normalize
                hull_a = _hull_area(perturbed)
                if hull_a > 0:
                    perturbed = perturbed / np.sqrt(hull_a)
                
                x0 = perturbed.flatten()
                res = minimize(_neg_score_cart, x0, method='Nelder-Mead',
                              options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
                if res.x is not None:
                    cand = res.x.reshape(-1, 2)
                    s = _score(cand)
                    if s > best_score:
                        best_score = s
                        best_pts = cand
        except Exception:
            pass
    
    if best_pts is None:
        # Fallback: regular polygon
        angles = 2 * np.pi * np.arange(N) / N
        best_pts = np.column_stack([np.cos(angles), np.sin(angles)])
        hull_a = _hull_area(best_pts)
        if hull_a > 0:
            best_pts = best_pts / np.sqrt(hull_a)
    
    return best_pts
# EVOLVE-BLOCK-END
