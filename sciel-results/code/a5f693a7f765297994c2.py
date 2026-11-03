"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

# Fixed angles for support function representation
# Uniform base angles with slight perturbation to break perfect symmetry
_BASE_ANGLES = np.linspace(0, 2 * np.pi, N, endpoint=False)

def _decode_support(radii):
    """
    Decode support function coefficients (radii at fixed angles) to point coordinates.
    The points are p_i = r_i * (cos(theta_i), sin(theta_i)).
    This parameterization does NOT strictly enforce convexity, but with radii > 0
    and angles in [0, 2*pi), the points will be in convex position if the polygon
    formed is convex. We verify convexity via cross products.
    """
    radii = np.abs(radii)  # Ensure positive radii
    cos_a = np.cos(_BASE_ANGLES)
    sin_a = np.sin(_BASE_ANGLES)
    points = np.column_stack([radii * cos_a, radii * sin_a])
    return points

def _is_convex(points):
    """Check if points form a convex polygon (all cross products same sign)."""
    n = len(points)
    signs = []
    for i in range(n):
        p0 = points[i]
        p1 = points[(i+1) % n]
        p2 = points[(i+2) % n]
        cross = (p1[0]-p0[0])*(p2[1]-p1[1]) - (p1[1]-p0[1])*(p2[0]-p1[0])
        if abs(cross) < 1e-15:
            return False
        signs.append(1 if cross > 0 else -1)
    # All must be same sign
    if len(set(signs)) != 1:
        return False
    return True

def _min_triangle_area(points):
    """Compute the minimum area of any triangle formed by 3 points."""
    min_area = float('inf')
    x = points[:, 0]
    y = points[:, 1]
    
    for i in range(N):
        for j in range(i + 1, N):
            if j + 1 >= N:
                continue
            dx_ji = x[j] - x[i]
            dy_ji = y[j] - y[i]
            dx_ki = x[j+1:] - x[i]
            dy_ki = y[j+1:] - y[i]
            cross = dx_ji * dy_ki - dx_ki * dy_ji
            areas = 0.5 * np.abs(cross)
            if len(areas) > 0:
                local_min = np.min(areas)
                if local_min < min_area:
                    min_area = local_min
    return min_area

def _score(points):
    """Compute the minimum triangle area divided by the convex hull area."""
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
    except Exception:
        return 0.0
    if hull_area < 1e-10:
        return 0.0
    min_area = _min_triangle_area(points)
    return min_area / hull_area

def _score_from_radii(radii):
    """Compute score directly from radii parameterization."""
    points = _decode_support(radii)
    if not _is_convex(points):
        return 0.0
    return _score(points)

def _objective_support(radii_flat):
    """
    Objective for support function optimization.
    We want to maximize min_triangle_area / hull_area, so minimize -score.
    """
    radii = radii_flat
    if np.any(np.isnan(radii)) or np.any(np.isinf(radii)):
        return 1e10
    
    points = _decode_support(radii)
    
    # Check convexity
    if not _is_convex(points):
        return 1e10
    
    score = _score(points)
    if score < 1e-6:
        return 1e10
    
    return -score

def _objective_support_soft(radii_flat, beta=50.0):
    """
    Soft-min objective for smoother optimization landscape.
    """
    radii = radii_flat
    if np.any(np.isnan(radii)) or np.any(np.isinf(radii)):
        return 1e10
    
    points = _decode_support(radii)
    
    if not _is_convex(points):
        return 1e10
    
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
    except Exception:
        return 1e10
    
    if hull_area < 1e-10:
        return 1e10
    
    x = points[:, 0]
    y = points[:, 1]
    
    # Compute all triangle areas
    areas_list = []
    for i in range(N):
        for j in range(i + 1, N):
            if j + 1 >= N:
                continue
            dx_ji = x[j] - x[i]
            dy_ji = y[j] - y[i]
            dx_ki = x[j+1:] - x[i]
            dy_ki = y[j+1:] - y[i]
            cross = dx_ji * dy_ki - dx_ki * dy_ji
            tri_areas = 0.5 * np.abs(cross)
            if len(tri_areas) > 0:
                areas_list.append(tri_areas)
    
    if not areas_list:
        return 1e10
    
    all_areas = np.concatenate(areas_list)
    
    if len(all_areas) == 0 or np.min(all_areas) < 1e-10:
        return 1e10
    
    # Log-sum-exp for soft min
    z = -beta * all_areas
    max_z = np.max(z)
    if max_z < -1000:
        log_sum = max_z
    else:
        log_sum = max_z + np.log(np.sum(np.exp(z - max_z)))
    
    soft_min_val = -log_sum / beta
    score = soft_min_val / hull_area
    
    if score < 1e-6:
        return 1e10
    
    return -score

def _generate_support_inits(rng):
    """Generate diverse initializations in radii space."""
    inits = []
    
    # 1. Uniform radii (regular polygon)
    inits.append(np.ones(N))
    
    # 2. Slightly perturbed uniform
    inits.append(np.ones(N) + rng.normal(0, 0.05, N))
    
    # 3. Larger perturbation
    inits.append(np.ones(N) + rng.normal(0, 0.15, N))
    
    # 4. Alternating radii (breaks symmetry)
    alt = np.ones(N)
    alt[::2] = 1.1
    alt[1::2] = 0.9
    inits.append(alt)
    
    # 5. Random positive radii
    inits.append(rng.uniform(0.5, 1.5, N))
    
    # 6. Another random set
    inits.append(rng.uniform(0.7, 1.3, N))
    
    # Ensure all radii are positive
    inits = [np.abs(r) for r in inits]
    
    return inits

def construct():
    """
    Generate 14 points using support function representation.
    Optimize 14 radial values at fixed angles using multi-start Nelder-Mead.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    rng = np.random.default_rng(12345)
    
    # Generate diverse initializations in radii space
    inits = _generate_support_inits(rng)
    
    best_radii = None
    best_score_val = -np.inf
    
    n_inits = len(inits)
    time_per_init = (deadline - start_time - 10) / n_inits
    
    for idx, r0 in enumerate(inits):
        if time.time() >= deadline - 10:
            break
        
        current_best_r = r0.copy()
        current_best_s = _score_from_radii(r0)
        
        # Nelder-Mead with soft objective for smoother landscape
        t_check = time.time()
        if t_check < deadline - max(5, time_per_init * 0.5):
            try:
                res = minimize(
                    _objective_support_soft, current_best_r,
                    method='Nelder-Mead',
                    options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
                )
                s = -res.fun
                if s > current_best_s:
                    current_best_s = s
                    current_best_r = res.x.copy()
            except Exception:
                pass
        
        # Second NM with hard objective for precise convergence
        t_check = time.time()
        if t_check < deadline - 5:
            try:
                res = minimize(
                    _objective_support, current_best_r,
                    method='Nelder-Mead',
                    options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
                )
                s = -res.fun
                if s > current_best_s:
                    current_best_s = s
                    current_best_r = res.x.copy()
            except Exception:
                pass
        
        # Powell for final polish
        t_check = time.time()
        if t_check < deadline - 3:
            try:
                res = minimize(
                    _objective_support, current_best_r,
                    method='Powell',
                    options={'maxiter': 1000, 'xtol': 1e-10, 'ftol': 1e-12}
                )
                s = -res.fun
                if s > current_best_s:
                    current_best_s = s
                    current_best_r = res.x.copy()
            except Exception:
                pass
        
        if current_best_s > best_score_val:
            best_score_val = current_best_s
            best_radii = current_best_r.copy()
        
        # Check time
        if time.time() >= deadline - 10:
            break
    
    # Final polish on best result
    if best_radii is not None and time.time() < deadline - 5:
        try:
            res_final = minimize(
                _objective_support, best_radii,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-11, 'fatol': 1e-13, 'adaptive': True}
            )
            s = -res_final.fun
            if s > best_score_val:
                best_score_val = s
                best_radii = res_final.x.copy()
        except Exception:
            pass
    
    if best_radii is not None and time.time() < deadline - 3:
        try:
            res_final2 = minimize(
                _objective_support, best_radii,
                method='Powell',
                options={'maxiter': 2000, 'xtol': 1e-11, 'ftol': 1e-13}
            )
            s = -res_final2.fun
            if s > best_score_val:
                best_score_val = s
                best_radii = res_final2.x.copy()
        except Exception:
            pass
    
    # Decode final radii to points
    if best_radii is not None:
        final_points = _decode_support(best_radii)
    else:
        # Fallback: regular 14-gon
        final_points = np.column_stack([np.cos(_BASE_ANGLES), np.sin(_BASE_ANGLES)])
    
    # Normalize to unit area
    try:
        hull = ConvexHull(final_points)
        hull_area = hull.volume
    except Exception:
        final_points = np.column_stack([np.cos(_BASE_ANGLES), np.sin(_BASE_ANGLES)])
        hull = ConvexHull(final_points)
        hull_area = hull.volume
    
    if hull_area > 1e-10:
        scale = np.sqrt(1.0 / hull_area)
        final_points = final_points * scale
    
    return final_points
# EVOLVE-BLOCK-END
