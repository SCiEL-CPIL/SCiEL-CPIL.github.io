"""
Heilbronn convex regions n=14: maximize min triangle area / hull area.
Strategy: symmetric parameterizations (two-ring, Fourier) + aggressive polish.
"""
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

# Heptagon angles for two-ring
_HEPT_ANGLES = np.linspace(0, 2 * np.pi, 7, endpoint=False)
_HEPT_COS = np.cos(_HEPT_ANGLES)
_HEPT_SIN = np.sin(_HEPT_ANGLES)


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


def _two_ring_pts(x):
    """
    x = [r, phi0, phi1, ..., phi6, theta0, theta1, ..., theta6]
    Inner ring: radius r, angles phi_i
    Outer ring: radius 1, angles theta_i
    Total: 15 params
    """
    r = x[0]
    phi = x[1:8]
    theta = x[8:15]
    ip = np.column_stack([r * np.cos(phi), r * np.sin(phi)])
    op = np.column_stack([np.cos(theta), np.sin(theta)])
    return np.vstack([ip, op])


def _two_ring_symmetric_pts(x):
    """
    Symmetric two-ring: 7-fold rotational symmetry.
    x = [r_inner, offset_inner, offset_outer]
    Inner ring at radius r, angles = heptagon angles + offset_inner
    Outer ring at radius 1, angles = heptagon angles + offset_outer
    """
    r = x[0]
    off_i = x[1]
    off_o = x[2]
    phi = _HEPT_ANGLES + off_i
    theta = _HEPT_ANGLES + off_o
    ip = np.column_stack([r * np.cos(phi), r * np.sin(phi)])
    op = np.column_stack([np.cos(theta), np.sin(theta)])
    return np.vstack([ip, op])


def _neg_two_ring_sym(x):
    pts = _two_ring_symmetric_pts(x)
    return -_score(pts)


def _neg_two_ring(x):
    pts = _two_ring_pts(x)
    return -_score(pts)


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
    radii = np.clip(radii, 0.1, 2.0)
    return radii[:, None] * np.column_stack([_COS_A, _SIN_A])


def _neg_score_fourier(c):
    pts = _fourier_to_points(c)
    return -_score(pts)


def _neg_soft_full(x, beta):
    pts = x.reshape(-1, 2)
    ha = _hull_area(pts)
    if ha < 1e-9:
        return 1e10
    areas = _all_triangle_areas(pts)
    return -(_soft_min(areas, beta) / ha)


def _neg_score_full(x):
    pts = x.reshape(-1, 2)
    return -_score(pts)


def _polish(pts, deadline, label=""):
    """Multi-stage local polish: continuation L-BFGS-B then Powell."""
    best_pts = pts.copy()
    best_s = _score(best_pts)
    x = best_pts.flatten()

    # Continuation on soft-min
    betas = [30.0, 100.0, 350.0]
    for beta in betas:
        if time.time() > deadline - 8:
            break
        try:
            res = minimize(_neg_soft_full, x, args=(beta,), method='L-BFGS-B',
                           options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10})
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
                           options={'maxiter': 400, 'xtol': 1e-10, 'ftol': 1e-12})
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

    # 2. Symmetric two-ring search (3 params: r, offset_inner, offset_outer)
    if time.time() < deadline - 60:
        bounds_sym = [(0.2, 0.9), (-0.5, 0.5), (-0.5, 0.5)]
        try:
            res = differential_evolution(_neg_two_ring_sym, bounds_sym, seed=7,
                                         maxiter=400, tol=1e-10, polish=True,
                                         popsize=20)
            if res.fun < 0:
                pts = _two_ring_symmetric_pts(res.x)
                update(pts)
        except Exception:
            pass

    # 3. Full two-ring search (15 params)
    if time.time() < deadline - 50:
        b2 = [(0.2, 0.8)] + [(0, 2 * np.pi)] * 14
        try:
            res = differential_evolution(_neg_two_ring, b2, seed=21,
                                         maxiter=200, tol=1e-6, polish=True,
                                         popsize=15)
            if res.fun < 0:
                update(_two_ring_pts(res.x))
        except Exception:
            pass

    # 4. Fourier-radius DE search (8-dim: modes 1-4)
    if time.time() < deadline - 40:
        bounds = [(-0.3, 0.3)] * 8
        try:
            res = differential_evolution(_neg_score_fourier, bounds, seed=11,
                                         maxiter=300, tol=1e-8, polish=True,
                                         popsize=15)
            if res.fun < 0:
                pts = _fourier_to_points(res.x)
                update(pts)
        except Exception:
            pass

    # 5. Polish the best found
    if best_pts is not None and time.time() < deadline - 20:
        best_pts, best_score = _polish(best_pts, deadline - 5, "main")
        update(best_pts)

    # 6. Random restarts + polish if time remains
    rng = np.random.RandomState(42)
    for restart in range(4):
        if time.time() > deadline - 25:
            break
        # Perturb regular polygon
        noise = rng.normal(0, 0.12, size=(N, 2))
        init = reg_pts + noise
        s_init = update(init)
        if time.time() < deadline - 15:
            polished, s_pol = _polish(init, deadline - 10, f"r{restart}")
            update(polished)

    # 7. Final global polish on the very best
    if best_pts is not None and time.time() < deadline - 5:
        best_pts, best_score = _polish(best_pts, deadline, "final")
        update(best_pts)

    if best_pts is None:
        return reg_pts
    return best_pts
