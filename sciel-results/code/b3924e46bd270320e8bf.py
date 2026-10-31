import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull
import warnings
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

def _polar_to_cartesian(params):
    """Convert 26-dim polar params (13 angles, 13 radii) to 13x2 Cartesian points.
    Angles are sorted to enforce convex polygon vertex ordering."""
    angles = params[:N]
    radii = params[N:]
    angles_sorted = np.sort(angles)
    angles_norm = angles_sorted % (2 * np.pi)
    radii = np.maximum(radii, 1e-6)
    xs = radii * np.cos(angles_norm)
    ys = radii * np.sin(angles_norm)
    return np.column_stack([xs, ys])

def _cartesian_to_polar(points):
    """Convert 13x2 Cartesian points to 26-dim polar params (sorted by angle)."""
    xs = points[:, 0]
    ys = points[:, 1]
    radii = np.sqrt(xs**2 + ys**2)
    angles = np.arctan2(ys, xs)
    angles = angles % (2 * np.pi)
    idx = np.argsort(angles)
    angles_sorted = angles[idx]
    radii_sorted = radii[idx]
    return np.concatenate([angles_sorted, radii_sorted])

def _polar_score(params):
    """Score for polar parameterization."""
    points = _polar_to_cartesian(params)
    return _score(points)

def _polar_de_objective(params):
    """Objective for DE in polar space: minimize negative score."""
    score = _polar_score(params)
    if score <= 0:
        return 1e10
    return -score

def _logsumexp_smooth_obj(params, beta=40.0):
    """Smooth surrogate: -soft_min(areas) / hull_area using power mean."""
    points = _polar_to_cartesian(params)
    
    if np.allclose(points, 0):
        return 1e10
    
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 1e10
    
    areas = _triangle_areas_vectorized(points)
    areas = np.maximum(areas, 1e-15)
    
    p = beta
    inv_areas = 1.0 / areas
    inv_areas_p = inv_areas ** p
    mean_inv_p = np.mean(inv_areas_p)
    soft_min = mean_inv_p ** (-1.0 / p)
    
    score = soft_min / hull_area
    if score <= 0:
        return 1e10
    return -score

def _run_de_polar(deadline):
    """Run DE in polar space (26 dims: 13 sorted angles + 13 radii)."""
    t_start = time.time()
    time_remaining = deadline - t_start
    
    if time_remaining < 20:
        return None
    
    bounds = [(0, 2 * np.pi)] * N + [(0.1, 3.0)] * N
    
    max_iter = max(10, min(80, int(time_remaining / 4.0)))
    
    de_params = {
        'maxiter': max_iter,
        'popsize': 15,
        'mutation': (0.4, 0.8),
        'recombination': 0.7,
        'tol': 1e-6,
        'seed': 42,
        'polish': False,
        'init': 'sobol',
        'updating': 'immediate',
    }
    
    try:
        result = differential_evolution(
            _polar_de_objective,
            bounds,
            **de_params
        )
        if result.fun < 1e9:
            return result.x
    except Exception:
        pass
    
    return None

def _local_polish_polar(params, deadline):
    """Local polish in polar space: L-BFGS-B on smooth surrogate, then Nelder-Mead on exact."""
    best_params = params.copy()
    best_score = _polar_score(params)
    
    if time.time() < deadline - 10:
        try:
            bounds = [(0, 2 * np.pi)] * N + [(0.1, 3.0)] * N
            res = minimize(
                lambda p: _logsumexp_smooth_obj(p, beta=40.0),
                best_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            s = _polar_score(res.x)
            if s > best_score:
                best_score = s
                best_params = res.x.copy()
        except Exception:
            pass
    
    if time.time() < deadline - 5:
        try:
            res = minimize(
                _polar_de_objective,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
            )
            s = _polar_score(res.x)
            if s > best_score:
                best_score = s
                best_params = res.x.copy()
        except Exception:
            pass
    
    return best_params, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    best_points = None
    best_score = 0.0
    
    def _update_best(points):
        nonlocal best_points, best_score
        if points is None or len(points) != N:
            return
        s = _score(points)
        if s > best_score:
            best_score = s
            best_points = points.copy()
    
    # Phase 1: DE in polar space
    de_result = _run_de_polar(deadline)
    if de_result is not None:
        de_points = _polar_to_cartesian(de_result)
        _update_best(de_points)
        print(f"DE polar: score={best_score:.6f}", flush=True)
    
    # Phase 2: Local polish from DE result
    if de_result is not None and time.time() < deadline - 15:
        polished_params, polished_score = _local_polish_polar(de_result, deadline)
        polished_points = _polar_to_cartesian(polished_params)
        _update_best(polished_points)
        print(f"Polished: score={best_score:.6f}", flush=True)
    
    # Phase 3: Diverse polar starts with local polish
    rng = np.random.default_rng(12345)
    
    structured_inits = []
    
    # Regular 13-gon
    angles_reg = np.linspace(0, 2 * np.pi, N, endpoint=False)
    radii_reg = np.ones(N)
    structured_inits.append(np.concatenate([angles_reg, radii_reg]))
    
    # Two-ring configs
    for n_out in [6, 7, 8]:
        n_in = N - n_out
        for r_in in [0.3, 0.5, 0.7]:
            angles_out = np.linspace(0, 2 * np.pi, n_out, endpoint=False)
            angles_in = np.linspace(0, 2 * np.pi, n_in, endpoint=False) + np.pi / n_out
            radii = np.concatenate([np.ones(n_out), np.full(n_in, r_in)])
            angles_all = np.concatenate([angles_out, angles_in])
            structured_inits.append(np.concatenate([angles_all, radii]))
    
    # Random starts
    for i in range(5):
        angles = np.sort(rng.random(N) * 2 * np.pi)
        radii = 0.5 + rng.random(N) * 1.5
        structured_inits.append(np.concatenate([angles, radii]))
    
    for i, params in enumerate(structured_inits):
        if time.time() > deadline - 20:
            break
        
        t_start = time.time()
        polished_params, polished_score = _local_polish_polar(params, deadline)
        polished_points = _polar_to_cartesian(polished_params)
        _update_best(polished_points)
        
        elapsed = time.time() - t_start
        if i % 3 == 0:
            print(f"Start {i}: score={best_score:.6f}, elapsed={elapsed:.1f}s", flush=True)
    
    # Phase 4: Cartesian Nelder-Mead polish
    if best_points is not None and time.time() < deadline - 10:
        try:
            def cart_obj(x):
                s = _score(x.reshape(N, 2))
                return -s if s > 0 else 1e10
            res_cart = minimize(
                cart_obj,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-11, 'fatol': 1e-11, 'adaptive': True}
            )
            _update_best(res_cart.x.reshape(N, 2))
        except Exception:
            pass
    
    # Phase 5: Final high-precision polish
    if best_points is not None and time.time() < deadline - 5:
        best_polar = _cartesian_to_polar(best_points)
        try:
            res_final = minimize(
                _polar_de_objective,
                best_polar,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
            )
            final_points = _polar_to_cartesian(res_final.x)
            _update_best(final_points)
        except Exception:
            pass
        
        if time.time() < deadline - 3:
            try:
                def cart_obj2(x):
                    s = _score(x.reshape(N, 2))
                    return -s if s > 0 else 1e10
                res_final_cart = minimize(
                    cart_obj2,
                    best_points.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 1000, 'xatol': 1e-13, 'fatol': 1e-13, 'adaptive': True}
                )
                _update_best(res_final_cart.x.reshape(N, 2))
            except Exception:
                pass
    
    if best_points is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    return best_points
