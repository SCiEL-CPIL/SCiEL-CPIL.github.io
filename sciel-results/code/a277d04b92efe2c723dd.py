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
        if np.allclose(points, points[0], atol=1e-12):
            return 0.0
        hull = ConvexHull(points)
        return float(hull.volume)
    except Exception:
        return 0.0


def _score(points):
    if len(points) < 3:
        return 0.0
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    areas = _triangle_areas_vectorized(points)
    min_area = float(np.min(areas))
    if min_area < 1e-15:
        return 0.0
    return min_area / hull_area


def _neg_score(points):
    s = _score(points)
    if s <= 0:
        return 1e10
    return -s


def _soft_min_value(values, temperature):
    if temperature <= 1e-12:
        return float(np.min(values))
    min_val = np.min(values)
    shifted = values - min_val
    shifted = np.clip(shifted, 0, 700 * temperature)
    log_sum = np.log(np.sum(np.exp(-shifted / temperature)))
    return min_val - temperature * log_sum


def _soft_score(points, temperature):
    if len(points) < 3:
        return 0.0
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    areas = _triangle_areas_vectorized(points)
    if np.any(areas < 1e-15):
        return 0.0
    soft_min = _soft_min_value(areas, temperature)
    if soft_min < 1e-15:
        return 0.0
    return soft_min / hull_area


def _neg_soft_score(points, temperature):
    s = _soft_score(points, temperature)
    if s <= 0:
        return 1e10
    return -s


# ---------- Radial / Angular representation ----------

def _radial_to_cartesian(params):
    """params: (2N,) -> [angles(N), radii(N)] -> (N,2) points centered at origin."""
    angles = params[:N]
    radii = params[N:]
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    pts = np.column_stack([x, y])
    pts = pts - np.mean(pts, axis=0)
    return pts


def _cartesian_to_radial(points):
    """Convert (N,2) centered points to (2N,) [angles, radii]."""
    points = points - np.mean(points, axis=0)
    radii = np.linalg.norm(points, axis=1)
    angles = np.arctan2(points[:, 1], points[:, 0])
    return np.concatenate([angles, radii])


def _neg_score_radial(params):
    pts = _radial_to_cartesian(params)
    return _neg_score(pts)


def _neg_soft_score_radial(params, temperature):
    pts = _radial_to_cartesian(params)
    return _neg_soft_score(pts, temperature)


# ---------- Penalty-based optimization (t-variable formulation) ----------

def _penalty_objective(params, t):
    """
    params: (2N,) radial/angular
    t: penalty weight for convexity violations
    """
    pts = _radial_to_cartesian(params)
    s = _score(pts)
    if s <= 0:
        return 1e10
    
    # Check convexity: all points should be on the hull
    # A point is interior if it's not a vertex of the convex hull
    try:
        hull = ConvexHull(pts)
        hull_vertices = set(hull.vertices)
        n_interior = N - len(hull_vertices)
    except Exception:
        n_interior = N  # degenerate
    
    # Penalty for interior points
    penalty = t * n_interior
    return -s + penalty


def _penalty_optimize(params0, deadline, t_start=10.0, t_end=1000.0, n_steps=8):
    """Annealed penalty optimization in radial space."""
    best_params = params0.copy()
    best_s = _score(_radial_to_cartesian(best_params))
    
    cur = params0.copy()
    ts = np.exp(np.linspace(np.log(t_start), np.log(t_end), n_steps))
    
    for t in ts:
        if time.time() > deadline - 5:
            break
        try:
            res = minimize(
                lambda x, tt=t: _penalty_objective(x, tt),
                cur,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': 800},
            )
            cand_params = res.x
            cand_pts = _radial_to_cartesian(cand_params)
            s = _score(cand_pts)
            if s > best_s:
                best_s = s
                best_params = cand_params.copy()
            cur = cand_params
        except Exception:
            pass
    
    # Final polish
    pts = _radial_to_cartesian(best_params)
    if time.time() < deadline - 3:
        try:
            res = minimize(
                lambda x: _neg_score(x.reshape(N, 2)),
                pts.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 400, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True},
            )
            cand = res.x.reshape(N, 2)
            s = _score(cand)
            if s > best_s:
                best_s = s
                pts = cand
        except Exception:
            pass
    
    return pts, best_s


# ---------- Multi-representation annealed soft-min ----------

def _annealed_soft_minimize_cartesian(points, deadline, temp_start=0.05, temp_end=1e-5,
                                       max_total_iters=1000, stages=4):
    best = points.copy()
    best_s = _score(best)
    temps = np.exp(np.linspace(np.log(temp_start), np.log(temp_end), stages))
    iters_per_stage = max(50, max_total_iters // stages)

    cur = points.flatten()
    for temp in temps:
        if time.time() > deadline - 3:
            break
        try:
            res = minimize(
                lambda x, t=temp: _neg_soft_score(x.reshape(N, 2), t),
                cur,
                method='L-BFGS-B',
                options={'maxiter': iters_per_stage, 'ftol': 1e-12,
                         'gtol': 1e-10, 'maxfun': iters_per_stage * 4},
            )
            cand = res.x.reshape(N, 2)
            s = _score(cand)
            if s > best_s:
                best_s, best = s, cand.copy()
            cur = res.x
        except Exception:
            pass

    if time.time() > deadline - 4:
        return best, best_s
    try:
        res = minimize(
            lambda x: _neg_score(x.reshape(N, 2)),
            best.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 600, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True},
        )
        cand = res.x.reshape(N, 2)
        s = _score(cand)
        if s > best_s:
            best_s, best = s, cand
    except Exception:
        pass
    return best, best_s


def _annealed_soft_minimize_radial(params0, deadline, temp_start=0.05, temp_end=1e-5,
                                    max_total_iters=1000, stages=4):
    best_params = params0.copy()
    best_s = _score(_radial_to_cartesian(best_params))
    temps = np.exp(np.linspace(np.log(temp_start), np.log(temp_end), stages))
    iters_per_stage = max(50, max_total_iters // stages)

    cur = params0.copy()
    for temp in temps:
        if time.time() > deadline - 3:
            break
        try:
            res = minimize(
                lambda x, t=temp: _neg_soft_score_radial(x, t),
                cur,
                method='L-BFGS-B',
                options={'maxiter': iters_per_stage, 'ftol': 1e-12,
                         'gtol': 1e-10, 'maxfun': iters_per_stage * 4},
            )
            cand_params = res.x
            cand_pts = _radial_to_cartesian(cand_params)
            s = _score(cand_pts)
            if s > best_s:
                best_s = s
                best_params = cand_params.copy()
            cur = cand_params
        except Exception:
            pass

    pts = _radial_to_cartesian(best_params)
    if time.time() > deadline - 4:
        return pts, best_s
    try:
        res = minimize(
            lambda x: _neg_score(x.reshape(N, 2)),
            pts.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True},
        )
        cand = res.x.reshape(N, 2)
        s = _score(cand)
        if s > best_s:
            best_s = s
            pts = cand
    except Exception:
        pass
    return pts, best_s


# ---------- Seed generators ----------

def _seed_regular(radius=1.0):
    ang = np.linspace(0, 2 * np.pi, N, endpoint=False)
    return np.column_stack([radius * np.cos(ang), radius * np.sin(ang)])


def _seed_two_ring(n_out, r_in, angle_offset):
    n_in = N - n_out
    a_out = np.linspace(0, 2 * np.pi, n_out, endpoint=False)
    a_in = np.linspace(angle_offset, 2 * np.pi + angle_offset, n_in, endpoint=False)
    outer = np.column_stack([np.cos(a_out), np.sin(a_out)])
    inner = np.column_stack([r_in * np.cos(a_in), r_in * np.sin(a_in)])
    return np.vstack([outer, inner])


def _seed_quasi_regular(radial_pert, angular_pert, radius=1.0):
    ang = np.linspace(0, 2 * np.pi, N, endpoint=False) + angular_pert
    rad = radius * (1.0 + radial_pert)
    return np.column_stack([rad * np.cos(ang), rad * np.sin(ang)])


# ---------- Simulated Annealing ----------

def _simulated_annealing(points, deadline, max_iters=4000, T_start=0.1, T_end=0.001):
    rng = np.random.default_rng(42)
    best = points.copy()
    best_s = _score(best)
    cur = points.copy()
    cur_s = best_s

    for it in range(max_iters):
        if time.time() > deadline - 5:
            break
        T = T_start * (T_end / T_start) ** (it / max_iters)

        n_perturb = rng.integers(1, 4)
        idxs = rng.choice(N, size=n_perturb, replace=False)
        step_size = T * 0.4
        new_pts = cur.copy()
        for idx in idxs:
            new_pts[idx] += rng.normal(0, step_size, size=2)

        new_s = _score(new_pts)
        if new_s >= cur_s or rng.random() < np.exp((new_s - cur_s) / max(T, 1e-10)):
            cur = new_pts
            cur_s = new_s
            if cur_s > best_s:
                best_s = cur_s
                best = cur.copy()
    return best, best_s


# ---------- CMA-ES-like via differential evolution in radial space ----------

def _de_objective(params):
    pts = _radial_to_cartesian(params)
    return _neg_score(pts)


def _differential_evolution_radial(params0, deadline, max_iter=50, popsize=15):
    """Use scipy's differential_evolution in radial parameter space."""
    # Bounds for angles: [-pi, pi], radii: [0.1, 2.0]
    bounds = [(-np.pi, np.pi)] * N + [(0.1, 2.0)] * N
    
    # Seed the DE with our best point
    x0 = params0.copy()
    
    def callback(xk, convergence):
        if time.time() > deadline - 3:
            raise RuntimeError("timeout")
    
    try:
        result = differential_evolution(
            _de_objective,
            bounds,
            seed=42,
            maxiter=max_iter,
            popsize=popsize,
            tol=1e-10,
            polish=True,
            init='sobol',
        )
        if time.time() > deadline - 3:
            return _radial_to_cartesian(params0), _score(_radial_to_cartesian(params0))
        pts = _radial_to_cartesian(result.x)
        s = _score(pts)
        return pts, s
    except Exception:
        return _radial_to_cartesian(params0), _score(_radial_to_cartesian(params0))


# ---------- Main construct ----------

def construct():
    start_time = time.time()
    deadline = start_time + 143.0

    best_points = _seed_regular()
    best_score = _score(best_points)

    def _update(pts):
        nonlocal best_points, best_score
        if pts is None or len(pts) != N:
            return
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_points = pts.copy()

    # Phase 1: Penalty-based radial optimization from multiple seeds
    rng = np.random.default_rng(12345)
    seeds = []
    seeds.append(_seed_regular())
    seeds.append(_seed_regular(0.95))
    seeds.append(_seed_regular(1.05))

    for r_in in [0.4, 0.5, 0.6, 0.7, 0.8]:
        for n_out in [6, 7, 8]:
            for off in [0.0, np.pi / 7, np.pi / 6, np.pi / 5]:
                seeds.append(_seed_two_ring(n_out, r_in, off))

    for _ in range(15):
        rp = rng.uniform(-0.15, 0.15, N)
        ap = rng.uniform(-0.2, 0.2, N)
        seeds.append(_seed_quasi_regular(rp, ap))

    for _ in range(10):
        r = rng.uniform(0.3, 1.0, N)
        a = rng.uniform(0, 2 * np.pi, N)
        seeds.append(np.column_stack([r * np.cos(a), r * np.sin(a)]))

    # Score and sort seeds
    scored = []
    for s in seeds:
        sc = _score(s)
        scored.append((sc, s))
    scored.sort(key=lambda x: -x[0])
    top_seeds = [s for sc, s in scored[:10]]
    if _score(best_points) > scored[0][0]:
        top_seeds.insert(0, best_points.copy())

    # Run penalty optimization on top seeds
    for idx, seed in enumerate(top_seeds[:6]):
        if time.time() > deadline - 80:
            break
        params0 = _cartesian_to_radial(seed)
        try:
            pts, s = _penalty_optimize(params0, deadline, t_start=5.0, t_end=500.0, n_steps=6)
            _update(pts)
        except Exception:
            pass

    # Phase 2: Radial soft-min annealing
    for idx, seed in enumerate(top_seeds[:5]):
        if time.time() > deadline - 50:
            break
        params0 = _cartesian_to_radial(seed)
        try:
            pts, s = _annealed_soft_minimize_radial(params0, deadline,
                                                     temp_start=0.05, temp_end=1e-5,
                                                     max_total_iters=600, stages=3)
            _update(pts)
        except Exception:
            pass

    # Phase 3: Cartesian soft-min annealing
    for idx, seed in enumerate(top_seeds[:4]):
        if time.time() > deadline - 30:
            break
        try:
            pts, s = _annealed_soft_minimize_cartesian(seed, deadline,
                                                        temp_start=0.05, temp_end=1e-5,
                                                        max_total_iters=600, stages=3)
            _update(pts)
        except Exception:
            pass

    # Phase 4: Differential evolution in radial space
    if time.time() < deadline - 25:
        params0 = _cartesian_to_radial(best_points)
        try:
            pts, s = _differential_evolution_radial(params0, deadline, max_iter=30, popsize=12)
            _update(pts)
        except Exception:
            pass

    # Phase 5: Simulated annealing
    if time.time() < deadline - 20:
        try:
            pts, s = _simulated_annealing(best_points, deadline, max_iters=2500)
            _update(pts)
            if time.time() < deadline - 10:
                pts2, s2 = _annealed_soft_minimize_cartesian(pts, deadline,
                                                              temp_start=0.03, temp_end=1e-5,
                                                              max_total_iters=400, stages=3)
                _update(pts2)
        except Exception:
            pass

    # Phase 6: Final Nelder-Mead polish
    if time.time() < deadline - 5:
        try:
            res = minimize(
                lambda x: _neg_score(x.reshape(N, 2)),
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-14, 'fatol': 1e-14, 'adaptive': True},
            )
            _update(res.x.reshape(N, 2))
        except Exception:
            pass

    # Phase 7: High-precision soft pass
    if time.time() < deadline - 2:
        try:
            res = minimize(
                lambda x: _neg_soft_score(x.reshape(N, 2), 1e-5),
                best_points.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': 800},
            )
            cand = res.x.reshape(N, 2)
            s = _score(cand)
            if s > best_score:
                best_score = s
                best_points = cand.copy()
        except Exception:
            pass

    if best_points is None:
        best_points = _seed_regular()

    return best_points
