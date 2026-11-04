"""Rewritten program: Fourier-radius perturbation of regular 14-gon + aggressive polish."""
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize, differential_evolution

N = 14

# Precompute triangle indices
_i_idx, _j_idx, _k_idx = [], [], []
for i in range(N - 2):
    for j in range(i + 1, N - 1):
        for k in range(j + 1, N):
            _i_idx.append(i)
            _j_idx.append(j)
            _k_idx.append(k)
_I = np.array(_i_idx)
_J = np.array(_j_idx)
_K = np.array(_k_idx)

# Precompute base angles for regular 14-gon
_BASE_ANGLES = np.linspace(0, 2 * np.pi, N, endpoint=False)
_COS_A = np.cos(_BASE_ANGLES)
_SIN_A = np.sin(_BASE_ANGLES)


def _all_triangle_areas(pts):
    """Return array of all C(14,3)=364 triangle areas."""
    pi = pts[_I]
    pj = pts[_J]
    pk = pts[_K]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross)


def _hull_area(pts):
    try:
        return ConvexHull(pts).volume
    except Exception:
        return 1e-10


def _score(pts):
    ha = _hull_area(pts)
    if ha < 1e-9:
        return 0.0
    return np.min(_all_triangle_areas(pts)) / ha


def _soft_min(areas, beta):
    m = np.min(areas)
    shifted = areas - m
    return m - (1.0 / beta) * np.log(np.sum(np.exp(-beta * shifted)))


def _fourier_to_points(c):
    """
    c = [a1, b1, a2, b2, a3, b3, a4, b4, ...] up to mode 4.
    radius_i = 1 + sum_{k=1}^{4} (a_k cos(k*theta_i) + b_k sin(k*theta_i))
    """
    radii = np.ones(N)
    for k in range(1, 5):
        a = c[2 * (k - 1)]
        b = c[2 * (k - 1) + 1]
        radii += a * np.cos(k * _BASE_ANGLES) + b * np.sin(k * _BASE_ANGLES)
    # Clamp radii to avoid negative or zero
    radii = np.clip(radii, 0.1, 2.0)
    return radii[:, None] * np.column_stack([_COS_A, _SIN_A])


def _neg_score_fourier(c):
    pts = _fourier_to_points(c)
    return -_score(pts)


def _neg_soft_fourier(c, beta):
    pts = _fourier_to_points(c)
    ha = _hull_area(pts)
    if ha < 1e-9:
        return 1e10
    areas = _all_triangle_areas(pts)
    return -(_soft_min(areas, beta) / ha)


def _neg_score_full(x):
    pts = x.reshape(-1, 2)
    return -_score(pts)


def _neg_soft_full(x, beta):
    pts = x.reshape(-1, 2)
    ha = _hull_area(pts)
    if ha < 1e-9:
        return 1e10
    areas = _all_triangle_areas(pts)
    return -(_soft_min(areas, beta) / ha)


def _polish(pts, deadline, label=""):
    """Multi-stage local polish: continuation L-BFGS-B then Powell."""
    best_pts = pts.copy()
    best_s = _score(best_pts)
    x = best_pts.flatten()

    # Continuation on soft-min
    betas = [20.0, 80.0, 300.0]
    for beta in betas:
        if time.time() > deadline - 8:
            break
        try:
            res = minimize(_neg_soft_full, x, args=(beta,), method='L-BFGS-B',
                           options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10})
            s = -res.fun
            if s > best_s:
                best_s = s
                best_pts = res.x.reshape(-1, 2)
                x = best_pts.flatten()
        except Exception:
            pass

    # Final Powell on hard objective
    if time.time() < deadline - 4:
        try:
            res = minimize(_neg_score_full, x, method='Powell',
                           options={'maxiter': 300, 'xtol': 1e-10, 'ftol': 1e-12})
            s = -res.fun
            if s > best_s:
                best_s = s
                best_pts = res.x.reshape(-1, 2)
        except Exception:
            pass

    return best_pts, best_s


def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # safety margin before 150s kill

    best_pts = None
    best_score = -np.inf

    def update(pts):
        nonlocal best_pts, best_score
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_pts = pts.copy()
        return s

    # 1. Regular 14-gon baseline
    reg_pts = np.column_stack([_COS_A, _SIN_A])
    update(reg_pts)

    # 2. Fourier-radius DE search (8-dim: modes 1-4)
    if time.time() < deadline - 50:
        bounds = [(-0.3, 0.3)] * 8
        try:
            res = differential_evolution(_neg_score_fourier, bounds, seed=7,
                                         maxiter=300, tol=1e-8, polish=True,
                                         popsize=15)
            if res.fun < 0:
                pts = _fourier_to_points(res.x)
                update(pts)
        except Exception:
            pass

    # 3. Second DE with different seed / more modes (12-dim: modes 1-6)
    if time.time() < deadline - 40:
        # Extend: add modes 5,6
        def _fourier6_to_points(c):
            radii = np.ones(N)
            for k in range(1, 7):
                a = c[2 * (k - 1)]
                b = c[2 * (k - 1) + 1]
                radii += a * np.cos(k * _BASE_ANGLES) + b * np.sin(k * _BASE_ANGLES)
            radii = np.clip(radii, 0.1, 2.0)
            return radii[:, None] * np.column_stack([_COS_A, _SIN_A])

        def _neg_score_f6(c):
            return -_score(_fourier6_to_points(c))

        bounds6 = [(-0.2, 0.2)] * 12
        try:
            res = differential_evolution(_neg_score_f6, bounds6, seed=13,
                                         maxiter=200, tol=1e-7, polish=True,
                                         popsize=12)
            if res.fun < 0:
                pts = _fourier6_to_points(res.x)
                update(pts)
        except Exception:
            pass

    # 4. Two-ring DE (inner 7 + outer 7) as alternative family
    if time.time() < deadline - 35:
        def _two_ring_pts(x):
            r = x[0]
            thi = x[1:8]
            tho = x[8:15]
            ip = np.column_stack([r * np.cos(thi), r * np.sin(thi)])
            op = np.column_stack([np.cos(tho), np.sin(tho)])
            return np.vstack([ip, op])

        def _neg_two_ring(x):
            return -_score(_two_ring_pts(x))

        b2 = [(0.2, 0.8)] + [(0, 2 * np.pi)] * 14
        try:
            res = differential_evolution(_neg_two_ring, b2, seed=21,
                                         maxiter=150, tol=1e-6, polish=True,
                                         popsize=12)
            if res.fun < 0:
                update(_two_ring_pts(res.x))
        except Exception:
            pass

    # 5. Polish the best found
    if best_pts is not None and time.time() < deadline - 15:
        best_pts, best_score = _polish(best_pts, deadline, "main")
        update(best_pts)

    # 6. Random restarts + polish if time remains
    rng = np.random.RandomState(42)
    for restart in range(5):
        if time.time() > deadline - 20:
            break
        # Perturb regular polygon
        noise = rng.normal(0, 0.15, size=(N, 2))
        init = reg_pts + noise
        s_init = update(init)
        if time.time() < deadline - 12:
            polished, s_pol = _polish(init, deadline - 8, f"r{restart}")
            update(polished)

    # 7. Final global polish on the very best
    if best_pts is not None and time.time() < deadline - 5:
        best_pts, best_score = _polish(best_pts, deadline, "final")
        update(best_pts)

    if best_pts is None:
        return reg_pts
    return best_pts
