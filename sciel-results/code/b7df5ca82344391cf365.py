import numpy as np
import time
import warnings
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull

warnings.filterwarnings('ignore')

N = 13

_TRIPLE_IDX = np.array([
    (i, j, k)
    for i in range(N)
    for j in range(i + 1, N)
    for k in range(j + 1, N)
], dtype=np.int32)

def _triangle_areas_vectorized(points):
    pi = points[_TRIPLE_IDX[:, 0]]
    pj = points[_TRIPLE_IDX[:, 1]]
    pk = points[_TRIPLE_IDX[:, 2]]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross)

def _hull_area(points):
    try:
        if len(points) < 3:
            return 0.0
        if np.allclose(points, points[0]):
            return 0.0
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
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

def _soft_min_score(points, p=100.0):
    """
    Calculate a smooth approximation of the score using soft-min.
    Returns the score (higher is better).
    """
    if len(points) < 3:
        return 0.0
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    areas = _triangle_areas_vectorized(points)
    areas = np.maximum(areas, 1e-15)
    
    # Log-sum-exp for soft-min
    neg_p_areas = -p * areas
    max_neg = np.max(neg_p_areas)
    log_sum = max_neg + np.log(np.sum(np.exp(neg_p_areas - max_neg)))
    soft_min = -log_sum / p
    
    return soft_min / hull_area

def _smooth_objective(x_flat, p=100.0):
    """
    Objective for minimization: minimize -soft_min_score.
    """
    points = x_flat.reshape(N, 2)
    # Center the points to remove translational invariance
    points = points - np.mean(points, axis=0)
    s = _soft_min_score(points, p)
    if s <= 0:
        return 1e10
    return -s

def _hard_objective(x_flat):
    """
    Objective for minimization: minimize -score.
    """
    points = x_flat.reshape(N, 2)
    points = points - np.mean(points, axis=0)
    s = _score(points)
    if s <= 0:
        return 1e10
    return -s

def _generate_regular_polygon(n):
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([np.cos(angles), np.sin(angles)])

def _generate_hexagonal_pattern():
    # 1 center, 6 ring 1, 6 ring 2
    center = np.array([[0.0, 0.0]])
    
    # Ring 1: 6 points at radius 1
    angles1 = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    ring1 = np.column_stack([np.cos(angles1), np.sin(angles1)])
    
    # Ring 2: 6 points at radius sqrt(3), rotated by pi/6
    # In hexagonal packing, next ring is at sqrt(3) and offset
    r2 = np.sqrt(3)
    angles2 = np.linspace(np.pi/6, 2 * np.pi + np.pi/6, 6, endpoint=False)
    ring2 = np.column_stack([r2 * np.cos(angles2), r2 * np.sin(angles2)])
    
    pts = np.vstack([center, ring1, ring2])
    return pts

def _generate_ring_config(n_outer, n_inner, r_inner, offset):
    """Create a two-ring configuration."""
    angles_outer = np.linspace(0, 2 * np.pi, n_outer, endpoint=False)
    angles_inner = np.linspace(offset, 2 * np.pi + offset, n_inner, endpoint=False)
    
    outer_pts = np.column_stack([np.cos(angles_outer), np.sin(angles_outer)])
    inner_pts = np.column_stack([r_inner * np.cos(angles_inner), r_inner * np.sin(angles_inner)])
    
    return np.vstack([outer_pts, inner_pts])

def _local_optimize(x0, deadline, p_smooth=100.0):
    """
    Perform local optimization starting from x0.
    1. Powell on smooth objective.
    2. Nelder-Mead on hard objective.
    3. Coordinate descent on hard objective.
    """
    best_x = x0.copy()
    best_s = _score(best_x.reshape(N, 2))
    
    # Phase 1: Powell on smooth objective
    if time.time() < deadline - 10:
        try:
            res = minimize(
                lambda x: _smooth_objective(x, p=p_smooth),
                best_x,
                method='Powell',
                options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-10}
            )
            s = _score(res.x.reshape(N, 2))
            if s > best_s:
                best_s = s
                best_x = res.x.copy()
        except Exception:
            pass
    
    # Phase 2: Nelder-Mead on hard objective
    if time.time() < deadline - 5:
        try:
            res_nm = minimize(
                _hard_objective,
                best_x,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
            )
            s_nm = _score(res_nm.x.reshape(N, 2))
            if s_nm > best_s:
                best_s = s_nm
                best_x = res_nm.x.copy()
        except Exception:
            pass
            
    # Phase 3: Coordinate Descent Polish
    if time.time() < deadline - 2:
        pts = best_x.reshape(N, 2)
        pts = pts - np.mean(pts, axis=0)
        curr_s = _score(pts)
        improved = True
        while improved and time.time() < deadline - 1:
            improved = False
            for i in range(N):
                for coord in range(2):
                    # Try small steps
                    for delta in [0.01, -0.01, 0.005, -0.005, 0.001, -0.001]:
                        trial = pts.copy()
                        trial[i, coord] += delta
                        trial = trial - np.mean(trial, axis=0) # Re-center
                        trial_s = _score(trial)
                        if trial_s > curr_s + 1e-12:
                            pts = trial
                            curr_s = trial_s
                            improved = True
                            break
            best_x = pts.flatten()
            best_s = curr_s
            
    return best_x, best_s

def construct():
    start_time = time.time()
    # Use 145s deadline to be safe against 150s limit
    deadline = start_time + 145.0
    
    best_points = None
    best_score = 0.0
    
    def _update_best(points):
        nonlocal best_points, best_score
        if points is None or len(points) != N:
            return
        pts = points - np.mean(points, axis=0)
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_points = pts.copy()
            
    # List of initial configurations
    inits = []
    
    # 1. Regular 13-gon (Highly likely to be near optimal or optimal)
    inits.append(_generate_regular_polygon(N))
    
    # 2. Hexagonal pattern (1+6+6)
    inits.append(_generate_hexagonal_pattern())
    
    # 3. Two-ring configurations
    # Try different splits of 13: 6+7, 5+8, 4+9
    for n_out in [6, 7, 5]:
        n_in = N - n_out
        if n_in < 3: continue
        for r_in in [0.4, 0.5, 0.6, 0.7, 0.8]:
            for offset in [0, np.pi / n_out, np.pi / (2 * n_out)]:
                inits.append(_generate_ring_config(n_out, n_in, r_in, offset))
                
    # 4. Random points on circle (with noise)
    rng = np.random.default_rng(42)
    for _ in range(10):
        angles = np.sort(rng.random(N) * 2 * np.pi)
        radii = 1.0 + rng.normal(0, 0.1, N)
        radii = np.abs(radii)
        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        inits.append(pts)
        
    # 5. Random points in disk
    for _ in range(10):
        r = np.sqrt(rng.random(N))
        theta = rng.random(N) * 2 * np.pi
        pts = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        inits.append(pts)

    # Shuffle inits to ensure we try diverse ones early
    rng2 = np.random.default_rng(123)
    indices = rng2.permutation(len(inits))
    inits = [inits[i] for i in indices]
    
    # Optimize each initial configuration
    for i, x0 in enumerate(inits):
        if time.time() > deadline - 10:
            break
            
        x_opt, s_opt = _local_optimize(x0.flatten(), deadline)
        _update_best(x_opt)
        
        if i % 5 == 0:
            elapsed = time.time() - start_time
            print(f"Init {i}/{len(inits)}: best_score={best_score:.6f}, elapsed={elapsed:.1f}s", flush=True)

    # If we have time, run a global search (Differential Evolution) on polar coordinates
    # to find potentially better topologies
    if best_points is not None and time.time() < deadline - 30:
        def _polar_objective(polar):
            angles = polar[:N]
            radii = polar[N:]
            x = radii * np.cos(angles)
            y = radii * np.sin(angles)
            points = np.column_stack([x, y])
            points = points - np.mean(points, axis=0)
            s = _soft_min_score(points, p=50.0)
            if s <= 0: return 1e10
            return -s
            
        bounds = [(0, 2 * np.pi)] * N + [(0.1, 2.0)] * N
        try:
            de_result = differential_evolution(
                _polar_objective,
                bounds=bounds,
                seed=42,
                maxiter=50,
                popsize=20,
                tol=1e-5,
                polish=False
            )
            if de_result.fun < 1e9:
                de_points = _polar_to_cartesian(de_result.x)
                de_points = de_points - np.mean(de_points, axis=0)
                _update_best(de_points)
                
                # Polish the DE result
                if time.time() < deadline - 10:
                    x_opt, s_opt = _local_optimize(de_points.flatten(), deadline)
                    _update_best(x_opt)
        except Exception:
            pass
            
    # Final high-precision polish
    if best_points is not None and time.time() < deadline - 5:
        x_final = best_points.flatten()
        try:
            res_final = minimize(
                _hard_objective,
                x_final,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-14, 'fatol': 1e-14, 'adaptive': True}
            )
            _update_best(res_final.x.reshape(N, 2))
        except Exception:
            pass

    # Fallback
    if best_points is None:
        best_points = _generate_regular_polygon(N)
        
    return best_points

def _polar_to_cartesian(polar):
    angles = polar[:N]
    radii = polar[N:]
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    return np.column_stack([x, y])
