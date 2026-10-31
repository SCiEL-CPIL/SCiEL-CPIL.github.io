import numpy as np
import time
from scipy.optimize import minimize
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


def _soft_min_value(values, beta):
    """Soft minimum using log-sum-exp trick."""
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return 0.0
    vmin = np.min(values)
    return vmin - np.log(np.sum(np.exp(-(values - vmin) * beta))) / beta


def _soft_score(points, beta):
    """Soft version of the score for gradient-based optimization."""
    if len(points) < 3:
        return 0.0
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    areas = _triangle_areas_vectorized(points)
    soft_min = _soft_min_value(areas, beta)
    if soft_min < 1e-15:
        return 0.0
    return soft_min / hull_area


def _neg_soft_score(points, beta):
    s = _soft_score(points, beta)
    if s <= 0:
        return 1e10
    return -s


def _build_regular_perturbed(radial_pert, angular_pert):
    """
    Build 13 points as a regular 13-gon with radial and angular perturbations.
    radial_pert: array of length N, perturbation to radius (base 1.0)
    angular_pert: array of length N, perturbation to angle (base 2*pi*i/N)
    """
    base_angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    angles = base_angles + angular_pert
    radii = 1.0 + radial_pert
    return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])


def _seed_regular():
    """Regular 13-gon on unit circle."""
    ang = np.linspace(0, 2 * np.pi, N, endpoint=False)
    return np.column_stack([np.cos(ang), np.sin(ang)])


def _seed_two_ring(n_out, r_inner, angle_offset):
    """Two-ring configuration."""
    n_in = N - n_out
    a_out = np.linspace(0, 2 * np.pi, n_out, endpoint=False)
    a_in = np.linspace(angle_offset, 2 * np.pi + angle_offset, n_in, endpoint=False)
    outer = np.column_stack([np.cos(a_out), np.sin(a_out)])
    inner = np.column_stack([r_inner * np.cos(a_in), r_inner * np.sin(a_in)])
    return np.vstack([outer, inner])


def _seed_three_ring(n1, n2, n3, r2, r3, off2, off3):
    """Three-ring configuration."""
    a1 = np.linspace(0, 2 * np.pi, n1, endpoint=False)
    a2 = np.linspace(off2, 2 * np.pi + off2, n2, endpoint=False)
    a3 = np.linspace(off3, 2 * np.pi + off3, n3, endpoint=False)
    pts = np.vstack([
        np.column_stack([np.cos(a1), np.sin(a1)]),
        np.column_stack([r2 * np.cos(a2), r2 * np.sin(a2)]),
        np.column_stack([r3 * np.cos(a3), r3 * np.sin(a3)]),
    ])
    return pts


def _seed_quasi_regular():
    """Quasi-regular: regular 13-gon with small random perturbations."""
    base = _seed_regular()
    rng = np.random.RandomState(42)
    radial_pert = rng.normal(0, 0.05, N)
    angular_pert = rng.normal(0, 0.05, N)
    return _build_regular_perturbed(radial_pert, angular_pert)


def _annealed_soft_minimize(points, deadline, initial_beta=50.0, final_beta=500.0, 
                            n_anneal_steps=8, max_iters_per_step=200):
    """
    Annealed soft-min optimization:
    Start with low beta (smooth approximation) and increase to high beta (sharp min).
    Optimize the 26-dimensional coordinate space directly.
    """
    best = points.copy()
    best_s = _score(best)
    
    # Generate annealing schedule
    betas = np.geomspace(initial_beta, final_beta, n_anneal_steps)
    
    for beta in betas:
        if time.time() >= deadline - 5:
            break
        
        x0 = best.flatten()
        
        def obj(x):
            pts = x.reshape(N, 2)
            return _neg_soft_score(pts, beta)
        
        try:
            res = minimize(
                obj, x0, method='L-BFGS-B',
                options={'maxiter': max_iters_per_step, 'gtol': 1e-8},
            )
            cand = res.x.reshape(N, 2)
            s = _score(cand)
            if s > best_s:
                best_s = s
                best = cand
        except Exception:
            pass
        
        # Also try Powell for non-smooth refinement
        if time.time() >= deadline - 5:
            break
        
        try:
            res = minimize(
                lambda x: _neg_score(x.reshape(N, 2)),
                best.flatten(),
                method='Powell',
                options={'maxiter': max_iters_per_step // 2, 'xatol': 1e-8, 'fatol': 1e-8},
            )
            cand = res.x.reshape(N, 2)
            s = _score(cand)
            if s > best_s:
                best_s = s
                best = cand
        except Exception:
            pass
    
    return best, best_s


def _optimize_from_points(points, deadline, max_iters_powell=300):
    """Polish a point configuration using Powell then Nelder-Mead on the true objective."""
    best = points.copy()
    best_s = _score(best)

    x0 = best.flatten()

    # Powell on the true (non-smooth) objective
    if time.time() < deadline - 6:
        try:
            res = minimize(
                lambda x: _neg_score(x.reshape(N, 2)),
                x0,
                method='Powell',
                options={'maxiter': max_iters_powell, 'xatol': 1e-8, 'fatol': 1e-8},
            )
            cand = res.x.reshape(N, 2)
            s = _score(cand)
            if s > best_s:
                best_s, best = s, cand
        except Exception:
            pass

    # Nelder-Mead fine polish
    if time.time() < deadline - 3:
        try:
            res = minimize(
                lambda x: _neg_score(x.reshape(N, 2)),
                best.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 800, 'xatol': 1e-11, 'fatol': 1e-11, 'adaptive': True},
            )
            cand = res.x.reshape(N, 2)
            s = _score(cand)
            if s > best_s:
                best_s, best = s, cand
        except Exception:
            pass

    return best, best_s


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin under 150s kill

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

    # ---------- Seed candidates ----------
    # Regular 13-gon
    _update(_seed_regular())
    
    # Two-ring seeds
    for n_out in [5, 6, 7, 8]:
        for r_in in [0.3, 0.5, 0.7]:
            for a_off in [0.0, np.pi / n_out, np.pi / (2.0 * n_out)]:
                _update(_seed_two_ring(n_out, r_in, a_off))
    
    # Three-ring seeds
    for (n1, n2, n3, r2, r3, o2, o3) in [
        (5, 4, 4, 0.6, 0.3, 0.0, 0.0),
        (6, 4, 3, 0.6, 0.3, 0.0, 0.0),
        (7, 4, 2, 0.6, 0.3, 0.0, 0.0),
    ]:
        _update(_seed_three_ring(n1, n2, n3, r2, r3, o2, o3))
    
    # Quasi-regular seed
    _update(_seed_quasi_regular())

    # ---------- Annealed soft-min optimization from multiple seeds ----------
    seeds = [
        _seed_regular(),
        _seed_two_ring(6, 0.5, np.pi / 6),
        _seed_two_ring(7, 0.5, np.pi / 7),
        _seed_three_ring(6, 4, 3, 0.6, 0.3, 0.0, 0.0),
        _seed_quasi_regular(),
    ]
    
    for seed in seeds:
        if time.time() >= deadline - 20:
            break
        try:
            pts, s = _annealed_soft_minimize(seed, deadline, 
                                             initial_beta=30.0, final_beta=300.0,
                                             n_anneal_steps=6, max_iters_per_step=150)
            _update(pts)
        except Exception:
            pass

    # ---------- Full-coordinate polish from the best found so far ----------
    if best_points is not None and time.time() < deadline - 10:
        try:
            pts, s = _optimize_from_points(best_points, deadline, max_iters_powell=300)
            _update(pts)
        except Exception:
            pass

    # ---------- Final ultra-fine Nelder-Mead on best ----------
    if best_points is not None and time.time() < deadline - 3:
        try:
            res = minimize(
                lambda x: _neg_score(x.reshape(N, 2)),
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 1500, 'xatol': 1e-13, 'fatol': 1e-13, 'adaptive': True},
            )
            _update(res.x.reshape(N, 2))
        except Exception:
            pass

    if best_points is None:
        best_points = _seed_regular()

    return best_points
