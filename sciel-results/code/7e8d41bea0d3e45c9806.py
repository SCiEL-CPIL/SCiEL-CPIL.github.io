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


def _soft_min_value(areas, beta):
    a = np.clip(areas, 1e-15, None)
    x = -beta * a
    m = np.max(x)
    return -(m + np.log(np.sum(np.exp(x - m)))) / beta


def _soft_score(points, beta):
    if len(points) < 3:
        return 0.0
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    areas = _triangle_areas_vectorized(points)
    sm = _soft_min_value(areas, beta)
    if sm < 1e-15:
        return 0.0
    return sm / hull_area


def _neg_soft_score(points, beta):
    s = _soft_score(points, beta)
    if s <= 0:
        return 1e10
    return -s


def _polar_to_cartesian(polar):
    """Convert (angles, radii) to Cartesian points.
    polar shape: (2N,) -> angles[0:N], radii[N:2N]
    """
    angles = polar[:N]
    radii = polar[N:]
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    return np.column_stack([x, y])


def _fourier_to_polar(params):
    """Convert Fourier parameters to (angles, radii).
    params shape: (7,) -> [base_angle, a1, b1, a2, b2, base_radius, r1]
    angles[i] = base_angle + 2*pi*i/N + a1*cos(2*pi*i/N) + b1*sin(2*pi*i/N)
                + a2*cos(4*pi*i/N) + b2*sin(4*pi*i/N)
    radii[i]  = base_radius + r1*cos(2*pi*i/N)
    """
    base_angle = params[0]
    a1, b1, a2, b2 = params[1], params[2], params[3], params[4]
    base_radius = params[5]
    r1 = params[6]
    
    i = np.arange(N)
    base_ang = 2 * np.pi * i / N
    angles = base_angle + base_ang + a1 * np.cos(base_ang) + b1 * np.sin(base_ang) + \
             a2 * np.cos(2 * base_ang) + b2 * np.sin(2 * base_ang)
    radii = base_radius + r1 * np.cos(base_ang)
    
    return np.concatenate([angles, radii])


def _fourier_objective(params):
    """Objective for Fourier-parameterized optimization."""
    polar = _fourier_to_polar(params)
    points = _polar_to_cartesian(polar)
    points = points - np.mean(points, axis=0)
    s = _score(points)
    if s <= 0:
        return 1e10
    return -s


def _fourier_soft_objective(params, beta):
    """Soft objective for Fourier-parameterized optimization."""
    polar = _fourier_to_polar(params)
    points = _polar_to_cartesian(polar)
    points = points - np.mean(points, axis=0)
    s = _soft_score(points, beta)
    if s <= 0:
        return 1e10
    return -s


def _annealed_soft_minimize(x0, deadline, beta_start=8.0, beta_end=120.0,
                            max_total_iters=1200, stages=4):
    best = x0.copy()
    best_s = _score(best)
    betas = np.geomspace(beta_start, beta_end, stages)
    iters_per_stage = max(50, max_total_iters // stages)

    cur = x0.copy()
    for beta in betas:
        if time.time() > deadline - 3:
            break
        try:
            res = minimize(
                lambda x, b=beta: _neg_soft_score(x.reshape(N, 2), b),
                cur,
                method='L-BFGS-B',
                options={'maxiter': iters_per_stage, 'ftol': 1e-12,
                         'gtol': 1e-10, 'maxfun': iters_per_stage * 4},
            )
            cand = res.x.reshape(N, 2)
            s = _score(cand)
            if s > best_s:
                best_s, best = s, cand.copy()
            cur = cand
        except Exception:
            pass

    if time.time() > deadline - 4:
        return best, best_s
    try:
        res = minimize(
            lambda x: -_score(x.reshape(N, 2)),
            best.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 600, 'xatol': 1e-12, 'fatol': 1e-12,
                     'adaptive': True},
        )
        cand = res.x.reshape(N, 2)
        s = _score(cand)
        if s > best_s:
            best_s, best = s, cand
    except Exception:
        pass
    return best, best_s


def _optimize_fourier(params0, deadline, beta_start=10.0, beta_end=150.0,
                      max_total_iters=800, stages=4):
    """Optimize in Fourier parameter space using annealed soft-min L-BFGS-B."""
    best_params = params0.copy()
    best_s = _fourier_objective(best_params)
    
    betas = np.geomspace(beta_start, beta_end, stages)
    iters_per_stage = max(30, max_total_iters // stages)
    
    cur = params0.copy()
    for beta in betas:
        if time.time() > deadline - 3:
            break
        try:
            res = minimize(
                lambda x, b=beta: _fourier_soft_objective(x, b),
                cur,
                method='L-BFGS-B',
                options={'maxiter': iters_per_stage, 'ftol': 1e-12,
                         'gtol': 1e-10, 'maxfun': iters_per_stage * 4},
            )
            cand_params = res.x
            s = -_fourier_objective(cand_params)
            if s > best_s:
                best_s, best_params = s, cand_params.copy()
            cur = cand_params
        except Exception:
            pass
    
    # Convert to Cartesian and do final polish
    polar = _fourier_to_polar(best_params)
    points = _polar_to_cartesian(polar)
    points = points - np.mean(points, axis=0)
    s = _score(points)
    if s > best_s:
        best_s = s
    
    # Polish in Cartesian space
    if time.time() > deadline - 4:
        return points, best_s
    try:
        res = minimize(
            lambda x: -_score(x.reshape(N, 2)),
            points.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 400, 'xatol': 1e-12, 'fatol': 1e-12,
                     'adaptive': True},
        )
        cand = res.x.reshape(N, 2)
        s = _score(cand)
        if s > best_s:
            best_s = s
            points = cand
    except Exception:
        pass
    return points, best_s


def _fourier_de_search(deadline, n_starts=8):
    """Run differential evolution in Fourier parameter space with multiple starts."""
    best_points = None
    best_score = -np.inf
    
    # Fourier parameter bounds
    # [base_angle, a1, b1, a2, b2, base_radius, r1]
    bounds = [
        (-np.pi, np.pi),    # base_angle
        (-0.3, 0.3),        # a1
        (-0.3, 0.3),        # b1
        (-0.15, 0.15),      # a2
        (-0.15, 0.15),      # b2
        (0.5, 1.5),         # base_radius
        (-0.2, 0.2),        # r1
    ]
    
    rng = np.random.default_rng(12345)
    for start_idx in range(n_starts):
        if time.time() > deadline - 10:
            break
        
        # Generate initial params
        if start_idx == 0:
            # Regular polygon
            params0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        else:
            params0 = np.array([
                rng.uniform(-np.pi, np.pi),
                rng.uniform(-0.2, 0.2),
                rng.uniform(-0.2, 0.2),
                rng.uniform(-0.1, 0.1),
                rng.uniform(-0.1, 0.1),
                rng.uniform(0.7, 1.3),
                rng.uniform(-0.15, 0.15),
            ])
        
        per_seed_budget = deadline - 10 - time.time()
        max_iters = min(300, max(50, int(per_seed_budget * 15)))
        
        try:
            pts, s = _optimize_fourier(params0, deadline,
                                       beta_start=10.0, beta_end=120.0,
                                       max_total_iters=max_iters, stages=3)
            if s > best_score:
                best_score = s
                best_points = pts.copy()
        except Exception:
            pass
    
    return best_points, best_score


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


def _simulated_annealing(points, deadline, max_iters=5000, T_start=0.1, T_end=0.001):
    """
    Global Simulated Annealing to escape local minima.
    """
    rng = np.random.default_rng(42)
    best = points.copy()
    best_s = _score(best)
    cur = points.copy()
    cur_s = best_s
    
    for it in range(max_iters):
        if time.time() > deadline - 5:
            break
        T = T_start * (T_end / T_start) ** (it / max_iters)
        
        # Propose move: perturb one or two points
        n_perturb = rng.integers(1, 3)
        idxs = rng.choice(N, size=n_perturb, replace=False)
        step_size = T * 0.5
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

    # Phase 1: Fourier-space optimization (primary strategy)
    # This enforces smooth convexity and reduces dimensionality to 7 parameters
    pts, s = _fourier_de_search(deadline, n_starts=10)
    _update(pts)

    # Phase 2: Cartesian-space optimization from multiple seeds
    seeds = []
    seeds.append(_seed_regular())
    seeds.append(_seed_regular(0.95))
    seeds.append(_seed_regular(1.05))

    for r_in in [0.4, 0.5, 0.6, 0.7, 0.8]:
        for n_out in [6, 7, 8]:
            for off in [0.0, np.pi / 7, np.pi / 6, np.pi / 5]:
                seeds.append(_seed_two_ring(n_out, r_in, off))

    rng = np.random.default_rng(7)
    for _ in range(30):
        rp = rng.uniform(-0.15, 0.15, N)
        ap = rng.uniform(-0.2, 0.2, N)
        seeds.append(_seed_quasi_regular(rp, ap))

    for _ in range(20):
        r = rng.uniform(0.3, 1.0, N)
        a = rng.uniform(0, 2 * np.pi, N)
        seeds.append(np.column_stack([r * np.cos(a), r * np.sin(a)]))

    scored = []
    for s in seeds:
        sc = _score(s)
        scored.append((sc, s))
    scored.sort(key=lambda x: -x[0])
    top_seeds = [s for sc, s in scored[:15]]
    if _score(best_points) > scored[0][0]:
        top_seeds.insert(0, best_points.copy())

    # Phase 3: Annealed soft-min L-BFGS-B in Cartesian space
    for idx, seed in enumerate(top_seeds):
        if time.time() > deadline - 30:
            break
        per_seed_budget = deadline - 30 - time.time()
        max_iters = min(1000, max(200, int(per_seed_budget * 20)))
        try:
            pts, s = _annealed_soft_minimize(seed, deadline,
                                             beta_start=8.0, beta_end=150.0,
                                             max_total_iters=max_iters, stages=4)
            _update(pts)
        except Exception:
            pass

    # Phase 4: Simulated Annealing for global exploration
    if time.time() < deadline - 25:
        try:
            pts, s = _simulated_annealing(best_points, deadline, max_iters=3000)
            _update(pts)
            # Re-optimize from SA result
            if time.time() < deadline - 15:
                pts2, s2 = _annealed_soft_minimize(pts, deadline,
                                                   beta_start=15.0, beta_end=120.0,
                                                   max_total_iters=800, stages=3)
                _update(pts2)
        except Exception:
            pass

    # Phase 5: Second round of Fourier optimization from best Cartesian points
    if time.time() < deadline - 15:
        # Try to fit Fourier params to best points
        try:
            # Simple approach: use the best points to initialize Fourier params
            # by computing their angles and radii
            centroid = np.mean(best_points, axis=0)
            centered = best_points - centroid
            radii = np.linalg.norm(centered, axis=1)
            angles = np.arctan2(centered[:, 1], centered[:, 0])
            
            # Sort by angle to match the Fourier parameterization
            sort_idx = np.argsort(angles)
            sorted_angles = angles[sort_idx]
            sorted_radii = radii[sort_idx]
            
            # Fit base parameters
            base_angle = sorted_angles[0]
            base_radius = np.mean(sorted_radii)
            r1 = np.mean(sorted_radii - base_radius)
            
            # Fit first harmonic
            i = np.arange(N)
            base_ang = 2 * np.pi * i / N
            ang_diff = sorted_angles - base_angle - base_ang
            a1 = np.mean(ang_diff * np.cos(base_ang))
            b1 = np.mean(ang_diff * np.sin(base_ang))
            a2 = 0.0
            b2 = 0.0
            
            params0 = np.array([base_angle, a1, b1, a2, b2, base_radius, r1])
            
            pts, s = _optimize_fourier(params0, deadline,
                                       beta_start=15.0, beta_end=150.0,
                                       max_total_iters=600, stages=3)
            _update(pts)
        except Exception:
            pass

    # Phase 6: Final Nelder-Mead polish
    if time.time() < deadline - 5:
        try:
            res = minimize(
                lambda x: -_score(x.reshape(N, 2)),
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-14, 'fatol': 1e-14,
                         'adaptive': True},
            )
            _update(res.x.reshape(N, 2))
        except Exception:
            pass

    # Phase 7: High-beta soft pass
    if time.time() < deadline - 2:
        try:
            res = minimize(
                lambda x: _neg_soft_score(x.reshape(N, 2), 250.0),
                best_points.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12,
                         'maxfun': 800},
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
