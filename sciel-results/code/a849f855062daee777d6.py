"""Heilbronn convex regions: maximize min triangle area / hull area for 13 points."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import itertools

N = 13

# EVOLVE-BLOCK-START

# Precompute triangle index triplets once (C(13,3)=286)
_TRI_IDX = np.array(list(itertools.combinations(range(N), 3)))
_TRI_I = _TRI_IDX[:, 0]
_TRI_J = _TRI_IDX[:, 1]
_TRI_K = _TRI_IDX[:, 2]

# Base angles for the regular 13-gon representation
_BASE_ANGLES = np.linspace(0, 2 * np.pi, N, endpoint=False)
_BASE_COS = np.cos(_BASE_ANGLES)
_BASE_SIN = np.sin(_BASE_ANGLES)

def _hull_area(pts):
    """Calculate the area of the convex hull of the points."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0

def _min_tri_area(pts):
    """Calculate the minimum area of any triangle formed by three points."""
    p_i = pts[_TRI_I]
    p_j = pts[_TRI_J]
    p_k = pts[_TRI_K]
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def _score(pts):
    """Calculate the score: min triangle area / hull area."""
    ha = _hull_area(pts)
    if ha < 1e-10:
        return 0.0
    mta = _min_tri_area(pts)
    return mta / ha

def _all_tri_areas(pts):
    """Vectorized areas of all C(N,3) triangles."""
    p_i = pts[_TRI_I]
    p_j = pts[_TRI_J]
    p_k = pts[_TRI_K]
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross)

def _soft_min_ratio(pts, temperature):
    """Smooth surrogate: soft-min of triangle areas / hull area."""
    areas = _all_tri_areas(pts)
    ha = _hull_area(pts)
    if ha < 1e-8:
        return -1e6
    neg = -areas / temperature
    m = np.max(neg)
    soft_min = -(np.log(np.sum(np.exp(neg - m))) + m) * temperature
    return soft_min / ha

def _radial_to_pts(radii):
    """Map 13 radial values (around unit circle) to 2D points."""
    return np.column_stack((radii * _BASE_COS, radii * _BASE_SIN))

def _neg_soft_radial(radii, temperature):
    """Objective for radial-parameter optimization (minimize)."""
    pts = _radial_to_pts(radii)
    return -_soft_min_ratio(pts, temperature)

def _optimize_radial(radii0, deadline, max_iter=300):
    """Anneal temperature and optimize 13 radial scalars with L-BFGS-B."""
    temps = [0.15, 0.08, 0.04, 0.02, 0.01, 0.005]
    best_radii = radii0.copy()
    best_pts = _radial_to_pts(best_radii)
    best_score = _score(best_pts)
    bounds = [(0.3, 1.7)] * N
    for temp in temps:
        if time.time() > deadline - 5:
            break
        try:
            res = minimize(
                _neg_soft_radial,
                best_radii,
                args=(temp,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            cand_radii = res.x
            cand_pts = _radial_to_pts(cand_radii)
            s = _score(cand_pts)
            if s > best_score:
                best_score = s
                best_pts = cand_pts
                best_radii = cand_radii
        except Exception:
            continue
    return best_pts, best_score

def _optimize_2d(pts0, deadline, max_iter=200):
    """Full 2D L-BFGS-B polish with annealed soft-min surrogate."""
    temps = [0.05, 0.02, 0.01, 0.005, 0.002]
    best_pts = pts0.copy()
    best_score = _score(best_pts)
    for temp in temps:
        if time.time() > deadline - 5:
            break
        try:
            res = minimize(
                lambda x: -_soft_min_ratio(x.reshape(N, 2), temp),
                best_pts.flatten(),
                method='L-BFGS-B',
                options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            cand_pts = res.x.reshape(N, 2)
            s = _score(cand_pts)
            if s > best_score:
                best_score = s
                best_pts = cand_pts
        except Exception:
            continue
    return best_pts, best_score

def _worst_tri_polish(pts, deadline, n_rounds=8):
    """Targeted polish: nudge the 3 vertices of the current worst triangle."""
    best_pts = pts.copy()
    best_score = _score(best_pts)
    rng = np.random.default_rng(7)
    for _ in range(n_rounds):
        if time.time() > deadline - 5:
            break
        areas = _all_tri_areas(best_pts)
        worst = int(np.argmin(areas))
        i, j, k = _TRI_I[worst], _TRI_J[worst], _TRI_K[worst]
        # Small random nudge on those three vertices
        delta = rng.normal(0, 0.05, size=(3, 2))
        cand = best_pts.copy()
        cand[[i, j, k]] += delta
        s = _score(cand)
        if s > best_score:
            best_score = s
            best_pts = cand
    return best_pts, best_score

def _local_search(pts, deadline, n_steps=40):
    """Random-perturbation local search to escape flat minima."""
    best_pts = pts.copy()
    best_score = _score(pts)
    rng = np.random.default_rng(42)
    step = 0.1
    for _ in range(n_steps):
        if time.time() > deadline - 5:
            break
        cand = best_pts + rng.normal(0, step, size=(N, 2))
        s = _score(cand)
        if s > best_score:
            best_score = s
            best_pts = cand
            step *= 0.95
        else:
            step *= 0.97
    return best_pts, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s wall-clock limit

    best_score = -np.inf
    best_pts = None

    # --- Phase 1: radial-parameter optimization (13 scalars, structurally convex) ---
    # Multiple diverse radial initializations
    rng0 = np.random.default_rng(0)
    radial_starts = [
        np.ones(N),                                   # regular polygon
        1.0 + rng0.normal(0, 0.1, N),                 # small jitter
        1.0 + rng0.normal(0, 0.2, N),                 # medium jitter
        1.0 + rng0.normal(0, 0.3, N),                 # large jitter
        1.0 + 0.15 * np.sin(2 * _BASE_ANGLES),        # sinusoidal deformation
        1.0 + 0.15 * np.cos(3 * _BASE_ANGLES),        # higher-mode deformation
    ]

    for rs in radial_starts:
        if time.time() > deadline - 25:
            break
        # clip to bounds before optimization
        rs = np.clip(rs, 0.3, 1.7)
        opt_pts, opt_score = _optimize_radial(rs, deadline)
        if opt_score > best_score:
            best_score = opt_score
            best_pts = opt_pts

    # --- Phase 2: full 2D polish from the best radial solution ---
    if best_pts is not None:
        if time.time() > deadline - 20:
            pass
        else:
            p2_pts, p2_score = _optimize_2d(best_pts, deadline)
            if p2_score > best_score:
                best_score = p2_score
                best_pts = p2_pts

    # --- Phase 3: targeted worst-triangle polish ---
    if best_pts is not None and time.time() <= deadline - 10:
        p3_pts, p3_score = _worst_tri_polish(best_pts, deadline)
        if p3_score > best_score:
            best_score = p3_score
            best_pts = p3_pts

    # --- Phase 4: local search to escape flat minima ---
    if best_pts is not None and time.time() <= deadline - 5:
        p4_pts, p4_score = _local_search(best_pts, deadline, n_steps=30)
        if p4_score > best_score:
            best_score = p4_score
            best_pts = p4_pts

    # Fallback: regular polygon
    if best_pts is None:
        best_pts = np.column_stack((_BASE_COS, _BASE_SIN))

    return best_pts

# EVOLVE-BLOCK-END
