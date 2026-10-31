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


def _soft_min_value(areas, beta):
    """Smooth min of areas via log-sum-exp."""
    areas_clipped = np.clip(areas, 1e-15, None)
    neg_beta_areas = -beta * areas_clipped
    max_val = np.max(neg_beta_areas)
    log_sum = max_val + np.log(np.sum(np.exp(neg_beta_areas - max_val)))
    return -log_sum / beta


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


def _seed_regular():
    ang = np.linspace(0, 2 * np.pi, N, endpoint=False)
    return np.column_stack([np.cos(ang), np.sin(ang)])


def _seed_two_ring(n_out, r_inner, angle_offset):
    n_in = N - n_out
    a_out = np.linspace(0, 2 * np.pi, n_out, endpoint=False)
    a_in = np.linspace(angle_offset, 2 * np.pi + angle_offset, n_in, endpoint=False)
    outer = np.column_stack([np.cos(a_out), np.sin(a_out)])
    inner = np.column_stack([r_inner * np.cos(a_in), r_inner * np.sin(a_in)])
    return np.vstack([outer, inner])


def _seed_three_ring(n1, n2, n3, r2, r3, off2, off3):
    a1 = np.linspace(0, 2 * np.pi, n1, endpoint=False)
    a2 = np.linspace(off2, 2 * np.pi + off2, n2, endpoint=False)
    a3 = np.linspace(off3, 2 * np.pi + off3, n3, endpoint=False)
    pts = np.vstack([
        np.column_stack([np.cos(a1), np.sin(a1)]),
        np.column_stack([r2 * np.cos(a2), r2 * np.sin(a2)]),
        np.column_stack([r3 * np.cos(a3), r3 * np.sin(a3)]),
    ])
    return pts


def _seed_quasi_regular(angular_pert_scale=0.05, radial_pert_scale=0.05, seed=0):
    rng = np.random.RandomState(seed)
    ang = np.linspace(0, 2 * np.pi, N, endpoint=False)
    ang += rng.normal(0, angular_pert_scale, N)
    rad = 1.0 + rng.normal(0, radial_pert_scale, N)
    return np.column_stack([rad * np.cos(ang), rad * np.sin(ang)])


def _annealed_soft_minimize(points, deadline, beta_start=5.0, beta_end=500.0,
                            n_anneal_steps=8, max_iter_per_step=200):
    """Anneal beta from low to high, using L-BFGS-B at each step.
    Returns best point found."""
    best = points.copy()
    best_s = _score(best)

    # Log-spaced beta schedule
    betas = np.geomspace(beta_start, beta_end, n_anneal_steps)

    for beta in betas:
        if time.time() > deadline - 10:
            break
        try:
            res = minimize(
                lambda x: _neg_soft_score(x.reshape(N, 2), beta),
                best.flatten(),
                method='L-BFGS-B',
                options={'maxiter': max_iter_per_step, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': 500},
            )
            cand = res.x.reshape(N, 2)
            s = _score(cand)
            if s > best_s:
                best_s = s
                best = cand.copy()
        except Exception:
            pass

    return best, best_s


def _coordinate_wise_local_search(points, deadline, step=0.01, max_passes=5):
    """Coordinate-wise line search to escape local minima in the true objective."""
    best = points.copy()
    best_s = _score(best)

    for _pass in range(max_passes):
        if time.time() > deadline - 8:
            break
        improved = False
        for i in range(N):
            for dim in range(2):
                if time.time() > deadline - 8:
                    break
                # Try small perturbations
                for delta in [step, -step, step * 0.5, -step * 0.5, step * 2.0, -step * 2.0]:
                    cand = best.copy()
                    cand[i, dim] += delta
                    s = _score(cand)
                    if s > best_s + 1e-12:
                        best = cand
                        best_s = s
                        improved = True
                        break
        if not improved:
            # Shrink step
            step *= 0.5
            if step < 1e-10:
                break

    return best, best_s


def _ultra_fine_polish(points, deadline):
    """Final Nelder-Mead polish with very tight tolerances."""
    if time.time() > deadline - 3:
        return points, _score(points)
    try:
        res = minimize(
            lambda x: _neg_score(x.reshape(N, 2)),
            points.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-14, 'adaptive': True},
        )
        cand = res.x.reshape(N, 2)
        s = _score(cand)
        if s > 0:
            return cand, s
    except Exception:
        pass
    return points, _score(points)


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin under 150s kill

    best_points = None
    best_score = 0.0

    def _update(pts):
        nonlocal best_points, best_score
        if pts is None or len(pts) != N:
            return
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_points = pts.copy()

    # ---------- Generate diverse seeds ----------
    seeds = []

    # Regular 13-gon
    seeds.append(_seed_regular())

    # Two-ring configurations
    for r_in in [0.4, 0.5, 0.6, 0.7, 0.8]:
        for off in [0.0, np.pi / 6, np.pi / 3, np.pi / 12]:
            seeds.append(_seed_two_ring(7, r_in, off))
    for r_in in [0.4, 0.5, 0.6, 0.7]:
        for off in [0.0, np.pi / 8, np.pi / 4]:
            seeds.append(_seed_two_ring(8, r_in, off))
    for r_in in [0.5, 0.6, 0.7]:
        for off in [0.0, np.pi / 5]:
            seeds.append(_seed_two_ring(6, r_in, off))

    # Three-ring configurations
    for r2 in [0.5, 0.6, 0.7]:
        for r3 in [0.2, 0.3, 0.4]:
            for o2 in [0.0, np.pi / 6]:
                for o3 in [0.0, np.pi / 4]:
                    seeds.append(_seed_three_ring(6, 4, 3, r2, r3, o2, o3))
    for r2 in [0.5, 0.6]:
        for r3 in [0.25, 0.35]:
            seeds.append(_seed_three_ring(5, 5, 3, r2, r3, np.pi / 10, np.pi / 6))

    # Quasi-regular with different random perturbations
    for s in range(6):
        seeds.append(_seed_quasi_regular(angular_pert_scale=0.05, radial_pert_scale=0.05, seed=s))
        seeds.append(_seed_quasi_regular(angular_pert_scale=0.1, radial_pert_scale=0.1, seed=s))

    # Score all seeds and keep top-5
    seed_scores = []
    for i, pts in enumerate(seeds):
        s = _score(pts)
        seed_scores.append((s, i, pts))
    seed_scores.sort(key=lambda x: -x[0])

    top_seeds = [pts for _, _, pts in seed_scores[:5]]

    # Update best from seeds
    for pts in top_seeds:
        _update(pts)

    # ---------- Phase 1: Annealed soft-min from each top seed ----------
    for idx, seed_pts in enumerate(top_seeds):
        if time.time() > deadline - 30:
            break
        try:
            pts, s = _annealed_soft_minimize(seed_pts, deadline,
                                              beta_start=5.0, beta_end=300.0,
                                              n_anneal_steps=7, max_iter_per_step=150)
            _update(pts)
        except Exception:
            pass

    # ---------- Phase 2: Coordinate-wise local search from best ----------
    if best_points is not None and time.time() < deadline - 20:
        try:
            pts, s = _coordinate_wise_local_search(best_points, deadline, step=0.02, max_passes=4)
            _update(pts)
        except Exception:
            pass

    # ---------- Phase 3: Annealed soft-min from coordinate-search result ----------
    if best_points is not None and time.time() < deadline - 25:
        try:
            pts, s = _annealed_soft_minimize(best_points, deadline,
                                              beta_start=20.0, beta_end=800.0,
                                              n_anneal_steps=6, max_iter_per_step=100)
            _update(pts)
        except Exception:
            pass

    # ---------- Phase 4: Differential Evolution global search ----------
    if time.time() < deadline - 40:
        try:
            def de_obj(x):
                return _neg_score(x.reshape(N, 2))
            res = differential_evolution(
                de_obj,
                bounds=[(-1.5, 1.5)] * (2 * N),
                maxiter=60,
                popsize=20,
                tol=1e-7,
                seed=42,
                polish=True,
                init='sobol',
                disp=False,
            )
            _update(res.x.reshape(N, 2))
        except Exception:
            pass

    # ---------- Phase 5: Annealed soft-min from DE result ----------
    if best_points is not None and time.time() < deadline - 20:
        try:
            pts, s = _annealed_soft_minimize(best_points, deadline,
                                              beta_start=10.0, beta_end=500.0,
                                              n_anneal_steps=6, max_iter_per_step=120)
            _update(pts)
        except Exception:
            pass

    # ---------- Phase 6: Second round of coordinate-wise search ----------
    if best_points is not None and time.time() < deadline - 15:
        try:
            pts, s = _coordinate_wise_local_search(best_points, deadline, step=0.01, max_passes=3)
            _update(pts)
        except Exception:
            pass

    # ---------- Phase 7: Final annealed soft-min with high beta ----------
    if best_points is not None and time.time() < deadline - 12:
        try:
            pts, s = _annealed_soft_minimize(best_points, deadline,
                                              beta_start=100.0, beta_end=2000.0,
                                              n_anneal_steps=5, max_iter_per_step=80)
            _update(pts)
        except Exception:
            pass

    # ---------- Phase 8: Ultra-fine Nelder-Mead polish ----------
    if best_points is not None and time.time() < deadline - 5:
        try:
            pts, s = _ultra_fine_polish(best_points, deadline)
            _update(pts)
        except Exception:
            pass

    # ---------- Phase 9: One more coordinate pass + polish ----------
    if best_points is not None and time.time() < deadline - 3:
        try:
            pts, s = _coordinate_wise_local_search(best_points, deadline, step=0.005, max_passes=2)
            _update(pts)
        except Exception:
            pass

    if best_points is None:
        best_points = _seed_regular()

    return best_points
