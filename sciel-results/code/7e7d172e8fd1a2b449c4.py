import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull
import warnings
warnings.filterwarnings('ignore')

N = 13

# Precompute all triangle index triples
_TRIPLE_IDX = np.array([
    (i, j, k)
    for i in range(N)
    for j in range(i + 1, N)
    for k in range(j + 1, N)
], dtype=np.int32)
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
        if np.allclose(points, points[0]):
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

def _objective(x_flat):
    """
    Objective function for optimization.
    1. Reshape to N x 2
    2. Center (remove translation)
    3. Rotate (remove rotation)
    4. Compute score and return negative
    """
    points = x_flat.reshape(N, 2)
    
    # Center
    centroid = np.mean(points, axis=0)
    points = points - centroid
    
    # Check if points are all the same (degenerate)
    if np.allclose(points, 0):
        return 1e10
    
    # Rotate such that point 0 is on the positive x-axis
    p0 = points[0]
    norm_p0 = np.linalg.norm(p0)
    if norm_p0 > 1e-9:
        theta = np.arctan2(p0[1], p0[0])
        c = np.cos(-theta)
        s = np.sin(-theta)
        R = np.array([[c, -s], [s, c]])
        points = points @ R.T
    else:
        p1 = points[1]
        norm_p1 = np.linalg.norm(p1)
        if norm_p1 > 1e-9:
            theta = np.arctan2(p1[1], p1[0])
            c = np.cos(-theta)
            s = np.sin(-theta)
            R = np.array([[c, -s], [s, c]])
            points = points @ R.T
        else:
            return 1e10

    score = _score(points)
    if score <= 0:
        return 1e10
    return -score

def _smooth_objective_adaptive(x_flat, p_start=20, p_end=80, beta=30.0):
    """Adaptive smooth surrogate that increases sharpness over iterations."""
    points = x_flat.reshape(N, 2)
    centroid = np.mean(points, axis=0)
    points = points - centroid
    
    if np.allclose(points, 0):
        return 1e10
    
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 1e10
    
    areas = _triangle_areas_vectorized(points)
    areas = np.maximum(areas, 1e-15)
    
    # Adaptive power mean: start smooth, get sharper
    p = p_start + (p_end - p_start) * min(1.0, _smooth_iter_counter[0] / 100.0)
    _smooth_iter_counter[0] += 1
    
    inv_areas = 1.0 / areas
    soft_min = (np.mean(inv_areas ** p)) ** (-1.0 / p)
    
    score = soft_min / hull_area
    if score <= 0:
        return 1e10
    return -score

_smooth_iter_counter = [0]

def _coordinate_descent_polish(points, max_iters=200, deadline=None):
    """Coordinate descent: optimize each point's x and y separately."""
    pts = points.copy()
    best_s = _score(pts)
    step = 0.01
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline - 2:
            break
        
        improved = False
        for i in range(N):
            # Optimize x-coordinate of point i
            for dx in [step, -step, step*0.5, -step*0.5]:
                trial = pts.copy()
                trial[i, 0] += dx
                s = _score(trial)
                if s > best_s + 1e-12:
                    pts = trial
                    best_s = s
                    improved = True
                    break
            
            # Optimize y-coordinate of point i
            for dy in [step, -step, step*0.5, -step*0.5]:
                trial = pts.copy()
                trial[i, 1] += dy
                s = _score(trial)
                if s > best_s + 1e-12:
                    pts = trial
                    best_s = s
                    improved = True
                    break
        
        if not improved:
            step *= 0.5
            if step < 1e-10:
                break
    
    return pts, best_s

def _local_polish(points, deadline):
    """Multi-method local polish: Nelder-Mead + coordinate descent."""
    best_pts = points.copy()
    best_s = _score(best_pts)
    
    # Nelder-Mead with tight tolerances
    if time.time() < deadline - 5:
        try:
            res = minimize(
                lambda x: -_score(x.reshape(N, 2)),
                best_pts.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-11, 'fatol': 1e-11, 'adaptive': True}
            )
            s = _score(res.x.reshape(N, 2))
            if s > best_s:
                best_s = s
                best_pts = res.x.reshape(N, 2).copy()
        except Exception:
            pass
    
    # Coordinate descent polish
    if time.time() < deadline - 3:
        cd_pts, cd_s = _coordinate_descent_polish(best_pts, max_iters=100, deadline=deadline)
        if cd_s > best_s:
            best_s = cd_s
            best_pts = cd_pts
    
    return best_pts, best_s

def _fourier_to_points(params):
    """
    Convert Fourier parameters to 13 points.
    params: [a1, b1, a2, b2, a3, b3, a4, b4]
    """
    k = np.arange(N)
    theta = 2 * np.pi * k / N
    a1, b1, a2, b2, a3, b3, a4, b4 = params
    r = (1.0 
         + a1 * np.cos(theta) + b1 * np.sin(theta) 
         + a2 * np.cos(2 * theta) + b2 * np.sin(2 * theta)
         + a3 * np.cos(3 * theta) + b3 * np.sin(3 * theta)
         + a4 * np.cos(4 * theta) + b4 * np.sin(4 * theta))
    r = np.maximum(r, 0.1)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.column_stack([x, y])

def _fourier_objective(params):
    """Objective for Fourier parameter optimization."""
    points = _fourier_to_points(params)
    if np.allclose(points, points[0]):
        return 1e10
    score = _score(points)
    if score <= 0:
        return 1e10
    return -score

def _grid_base_4x4_minus_corner():
    """Return a centered 4x4 grid minus one corner (13 points)."""
    xs = np.array([-1.5, -0.5, 0.5, 1.5])
    ys = np.array([-1.5, -0.5, 0.5, 1.5])
    gx, gy = np.meshgrid(xs, ys)
    pts = np.column_stack([gx.ravel(), gy.ravel()])
    pts = pts[:-1]
    pts = pts - np.mean(pts, axis=0)
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
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

    # Initial guesses
    inits = []
    
    # 1. Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    reg_pts = np.column_stack([np.cos(angles), np.sin(angles)])
    inits.append(reg_pts.flatten())
    
    # 2. Two rings: 7 outer, 6 inner
    r_out, r_in = 1.0, 0.5
    ang_out = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    ang_in = np.linspace(0, 2 * np.pi, 6, endpoint=False) + np.pi/6
    pts_7_6 = np.vstack([
        np.column_stack([r_out * np.cos(ang_out), r_out * np.sin(ang_out)]),
        np.column_stack([r_in * np.cos(ang_in), r_in * np.sin(ang_in)])
    ])
    inits.append(pts_7_6.flatten())
    
    # 3. Three rings: 5-4-4
    r1, r2, r3 = 1.0, 0.6, 0.2
    a1 = np.linspace(0, 2 * np.pi, 5, endpoint=False)
    a2 = np.linspace(0, 2 * np.pi, 4, endpoint=False) + np.pi/4
    a3 = np.linspace(0, 2 * np.pi, 4, endpoint=False)
    pts_5_4_4 = np.vstack([
        np.column_stack([r1 * np.cos(a1), r1 * np.sin(a1)]),
        np.column_stack([r2 * np.cos(a2), r2 * np.sin(a2)]),
        np.column_stack([r3 * np.cos(a3), r3 * np.sin(a3)])
    ])
    inits.append(pts_5_4_4.flatten())
    
    # 4. Grid base
    grid_pts = _grid_base_4x4_minus_corner()
    inits.append(grid_pts.flatten())
    
    # 5. Random starts
    rng = np.random.default_rng(42)
    for _ in range(20):
        r = np.sqrt(rng.random(N))
        theta = rng.random(N) * 2 * np.pi
        pts = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        inits.append(pts.flatten())
        
    # Optimization Loop
    for i, x0 in enumerate(inits):
        if time.time() > deadline - 15:
            break
            
        # Phase 1: Smooth optimization to find a good basin
        _smooth_iter_counter[0] = 0
        try:
            res = minimize(
                lambda x: _smooth_objective_adaptive(x), 
                x0, 
                method='Powell',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            opt_pts = res.x.reshape(N, 2)
            _update_best(opt_pts)
        except Exception:
            pass
            
        # Phase 2: Hard objective polish with Nelder-Mead
        if best_score > 0 and time.time() < deadline - 5:
            try:
                start_x = res.x if 'res' in locals() and res.success else x0
                res_nm = minimize(
                    _objective,
                    start_x,
                    method='Nelder-Mead',
                    options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
                )
                opt_pts_nm = res_nm.x.reshape(N, 2)
                _update_best(opt_pts_nm)
            except Exception:
                pass

    # Fourier series optimization
    if time.time() < deadline - 10:
        try:
            bounds = [(-0.3, 0.3)] * 8
            
            de_res = differential_evolution(
                _fourier_objective,
                bounds,
                seed=42,
                maxiter=80,
                tol=1e-6,
                popsize=20,
                update_interval=1,
                polish=True
            )
            
            if de_res.success or de_res.fun < 1e9:
                best_fourier_params = de_res.x
                fourier_points = _fourier_to_points(best_fourier_params)
                _update_best(fourier_points)
                
                if best_points is not None and time.time() < deadline - 3:
                    res_polish = minimize(
                        _objective,
                        fourier_points.flatten(),
                        method='Nelder-Mead',
                        options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
                    )
                    opt_pts_polish = res_polish.x.reshape(N, 2)
                    _update_best(opt_pts_polish)
                    
        except Exception:
            pass

    # Final high-precision polish on the best candidate found
    if best_points is not None and time.time() < deadline - 2:
        try:
            res_final = minimize(
                _objective,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
            )
            opt_pts_final = res_final.x.reshape(N, 2)
            _update_best(opt_pts_final)
        except Exception:
            pass
    
    # Final coordinate descent polish
    if best_points is not None and time.time() < deadline - 1:
        try:
            cd_pts, cd_s = _coordinate_descent_polish(best_points, max_iters=50, deadline=deadline)
            if cd_s > best_score:
                best_score = cd_s
                best_points = cd_pts
        except Exception:
            pass

    if best_points is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
        
    return best_points
