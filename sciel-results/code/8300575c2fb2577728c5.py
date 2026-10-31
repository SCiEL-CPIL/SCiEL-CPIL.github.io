import numpy as np
import time
import warnings
from scipy.spatial import ConvexHull
from scipy.optimize import minimize, differential_evolution
import sys

warnings.filterwarnings('ignore')

N = 13

# Precompute all triangle index triples
_TRIPLE_IDX = []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            _TRIPLE_IDX.append((i, j, k))
_TRIPLE_IDX = np.array(_TRIPLE_IDX, dtype=np.int32)
_NUM_TRIPLES = len(_TRIPLE_IDX)


def _triangle_areas_vectorized(points):
    """Compute all triangle areas using precomputed indices."""
    pi = points[_TRIPLE_IDX[:, 0]]
    pj = points[_TRIPLE_IDX[:, 1]]
    pk = points[_TRIPLE_IDX[:, 2]]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross)


def _hull_area(points):
    """Compute convex hull area."""
    try:
        if len(points) < 3:
            return 0.0
        if np.ptp(points, axis=0).max() < 1e-10:
            return 0.0
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0


def _score(points):
    """Compute the score: min triangle area / hull area."""
    if len(points) < 3:
        return 0.0
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    areas = _triangle_areas_vectorized(points)
    min_area = np.min(areas)
    if min_area < 1e-15:
        return 0.0
    return min_area / hull_area


def _support_line_polygon(normals, offsets):
    """
    Build a convex polygon from support lines.
    normals: (N, 2) array of unit normal vectors
    offsets: (N,) array of offsets (distance from origin along normal)
    Returns: (N, 2) array of vertex coordinates
    """
    # Each edge i is defined by: normals[i] . x = offsets[i]
    # Vertex i is the intersection of lines i and (i+1) % N
    vertices = np.zeros((N, 2))
    for i in range(N):
        j = (i + 1) % N
        n1 = normals[i]
        n2 = normals[j]
        o1 = offsets[i]
        o2 = offsets[j]
        
        # Solve: n1.x = o1, n2.x = o2
        # Using Cramer's rule for 2x2 system
        det = n1[0] * n2[1] - n1[1] * n2[0]
        if abs(det) < 1e-15:
            # Degenerate case - lines are parallel
            vertices[i] = [o1 * n1[0], o1 * n1[1]]
            continue
        x = (o1 * n2[1] - o2 * n1[1]) / det
        y = (n1[0] * o2 - n2[0] * o1) / det
        vertices[i] = [x, y]
    
    return vertices


def _support_line_objective(params):
    """
    Objective for support-line representation.
    params: (2N,) array = [normal_angles (N,), offsets (N,)]
    Uses penalty for non-convexity or degenerate polygons.
    """
    angles = params[:N]
    offsets = params[N:]
    
    # Convert angles to unit normals
    normals = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Ensure offsets are positive (for a convex polygon containing origin)
    # Add penalty if any offset is too small
    min_offset = np.min(offsets)
    if min_offset < 0.05:
        return 1e8 + (0.05 - min_offset) * 1e6
    
    points = _support_line_polygon(normals, offsets)
    
    # Check if polygon is valid (non-degenerate)
    if np.any(~np.isfinite(points)):
        return 1e8
    
    score = _score(points)
    if score <= 0:
        return 1e8
    
    return -score


def _smooth_objective(params, beta=50.0):
    """
    Smooth surrogate objective using log-sum-exp.
    params: (2N,) array of point coordinates
    """
    points = params.reshape(N, 2)
    
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 1e10
    
    areas = _triangle_areas_vectorized(points)
    a_min_val = np.min(areas)
    if a_min_val < 1e-15:
        return 1e10
    
    ratios = areas / hull_area
    r_min = np.min(ratios)
    if r_min < 1e-15:
        return 1e10
    
    shifted = -beta * (ratios - r_min)
    shifted = np.clip(shifted, -500, 500)
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_min_val = r_min + log_sum / beta
    return -soft_min_val


def _coordinate_descent_polish(points, deadline, max_iter=3000):
    """
    Coordinate descent on the hard objective.
    Tries small moves in each coordinate direction.
    """
    best_pts = points.copy()
    best_s = _score(best_pts)
    
    step_sizes = [0.02, 0.01, 0.005, 0.002, 0.001, 0.0005]
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        improved = False
        for i in range(N):
            for coord in range(2):
                for delta in step_sizes:
                    for sign in [1, -1]:
                        trial = best_pts.copy()
                        trial[i, coord] += sign * delta
                        trial_s = _score(trial)
                        if trial_s > best_s:
                            best_pts = trial
                            best_s = trial_s
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        
        if not improved:
            # Try smaller steps
            step_sizes = [s * 0.5 for s in step_sizes]
            if step_sizes[0] < 1e-6:
                break
    
    return best_pts, best_s


def _local_polish(points, deadline, max_iter=2000):
    """
    Local search with adaptive step size.
    """
    best_pts = points.copy()
    best_s = _score(best_pts)
    current_pts = points.copy()
    current_s = best_s
    step = 0.03
    rng = np.random.default_rng()
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        if step < 1e-7:
            break
        
        # Generate random perturbation
        perturbation = rng.normal(0, step, size=(N, 2))
        new_pts = current_pts + perturbation
        new_s = _score(new_pts)
        
        if new_s > current_s:
            current_pts = new_pts
            current_s = new_s
            if new_s > best_s:
                best_s = new_s
                best_pts = new_pts.copy()
            step = min(step * 1.02, 0.08)
        else:
            step *= 0.998
    
    return best_pts, best_s


def _regular_polygon(n, radius=1.0):
    """Generate a regular n-gon centered at origin."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin against 150s limit
    
    best_points = None
    best_score = 0.0
    
    def _update_best(points):
        nonlocal best_points, best_score
        if points is None:
            return
        if len(points) != N:
            return
        s = _score(points)
        if s > best_score:
            best_score = s
            best_points = points.copy()
    
    # Phase 1: Initialize with regular polygon and support-line configs
    _update_best(_regular_polygon(N, 1.0))
    
    # Try support-line representation with regular angles
    regular_angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    # For a regular polygon, offsets = r * cos(pi/N) where r is circumradius
    # Area of regular n-gon with circumradius r: (n/2) * r^2 * sin(2*pi/n)
    # We want area ~ 1, so r^2 = 2 / (n * sin(2*pi/n))
    r_circum = np.sqrt(2.0 / (N * np.sin(2 * np.pi / N)))
    offsets_reg = r_circum * np.cos(np.pi / N)
    sl_params = np.concatenate([regular_angles, np.full(N, offsets_reg)])
    sl_points = _support_line_polygon(
        np.column_stack([np.cos(regular_angles), np.sin(regular_angles)]),
        np.full(N, offsets_reg)
    )
    _update_best(sl_points)
    
    # Phase 2: Optimize support-line representation
    if time.time() < deadline - 60:
        # Try multiple starting points for support-line optimization
        for trial in range(5):
            if time.time() > deadline - 55:
                break
            
            rng = np.random.default_rng(trial * 7919)
            
            if trial == 0:
                # Start from regular polygon
                angles_init = regular_angles.copy()
                offsets_init = np.full(N, offsets_reg)
            else:
                # Perturbed angles and offsets
                angles_init = regular_angles + rng.normal(0, 0.1, N)
                offsets_init = offsets_reg * (1 + rng.normal(0, 0.05, N))
            
            params_init = np.concatenate([angles_init, offsets_init])
            
            try:
                result = minimize(
                    _support_line_objective,
                    params_init,
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'xatol': 1e-5, 'fatol': 1e-6}
                )
                opt_angles = result.x[:N]
                opt_offsets = result.x[N:]
                opt_normals = np.column_stack([np.cos(opt_angles), np.sin(opt_angles)])
                opt_points = _support_line_polygon(opt_normals, opt_offsets)
                _update_best(opt_points)
            except Exception:
                pass
    
    # Phase 3: Smooth objective optimization from best points
    if best_points is not None and time.time() < deadline - 40:
        for trial in range(3):
            if time.time() > deadline - 35:
                break
            
            rng = np.random.default_rng(trial * 104729)
            if trial == 0:
                init = best_points.flatten()
            else:
                init = best_points.flatten() + rng.normal(0, 0.02, N * 2)
            
            try:
                result = minimize(
                    _smooth_objective,
                    init,
                    method='Nelder-Mead',
                    options={'maxiter': 300, 'xatol': 1e-5, 'fatol': 1e-6}
                )
                opt_points = result.x.reshape(N, 2)
                _update_best(opt_points)
            except Exception:
                pass
    
    # Phase 4: Coordinate descent polish
    if best_points is not None and time.time() < deadline - 25:
        cd_points, cd_score = _coordinate_descent_polish(
            best_points, deadline - 20, max_iter=2000
        )
        _update_best(cd_points)
    
    # Phase 5: Local search polish
    if best_points is not None and time.time() < deadline - 15:
        for trial in range(4):
            if time.time() > deadline - 10:
                break
            
            rng = np.random.default_rng(trial * 31337)
            if trial == 0:
                start_pts = best_points.copy()
            else:
                start_pts = best_points + rng.normal(0, 0.015, (N, 2))
            
            ls_points, ls_score = _local_polish(
                start_pts, deadline - 8, max_iter=1500
            )
            _update_best(ls_points)
    
    # Phase 6: Final differential evolution if time permits
    if time.time() < deadline - 30:
        try:
            # Define bounds for differential evolution
            bounds = [(-3, 3)] * (N * 2)
            
            de_result = differential_evolution(
                _smooth_objective,
                bounds,
                maxiter=50,
                popsize=15,
                tol=1e-4,
                seed=42,
                polish=False,
                workers=1
            )
            de_points = de_result.x.reshape(N, 2)
            _update_best(de_points)
            
            # Polish the DE result
            if time.time() < deadline - 10:
                cd_points, cd_score = _coordinate_descent_polish(
                    de_points, deadline - 5, max_iter=1000
                )
                _update_best(cd_points)
        except Exception:
            pass
    
    # Final fallback
    if best_points is None:
        best_points = _regular_polygon(N, 1.0)
    
    return best_points
