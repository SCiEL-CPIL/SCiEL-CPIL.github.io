"""Packing 50 equal circles in a unit circle:
ring/hex init + multi-stage L-BFGS on a smooth soft-min objective +
SA escape + constraint-push / binary-search polish.
"""
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 50
DEADLINE = 140.0  # safety margin before the 150 s wall-clock kill

# ---------------------------- scoring ----------------------------

def _pairwise_distances(pts):
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=-1))


def _score(pts):
    """Largest common radius for the given centres (0 if invalid)."""
    if pts is None or len(pts) != N:
        return 0.0
    pts = np.asarray(pts, dtype=np.float64)
    if not np.all(np.isfinite(pts)):
        return 0.0
    d = _pairwise_distances(pts)
    np.fill_diagonal(d, np.inf)
    min_pair = np.min(d) / 2.0
    min_boundary = np.min(1.0 - np.sqrt(np.sum(pts ** 2, axis=1)))
    val = min(min_pair, min_boundary)
    if val <= 0.0:
        return 0.0
    return val


# ------------------- smooth objective (soft-min) -------------------

def _softmin_logsumexp(values, beta):
    if len(values) == 0:
        return 0.0
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return 0.0
    neg = -values * beta
    neg_max = np.max(neg)
    if not np.isfinite(neg_max):
        return 0.0
    exp_sum = np.sum(np.exp(neg - neg_max))
    if exp_sum <= 0.0:
        return 0.0
    return -(neg_max + np.log(exp_sum)) / beta


def _smooth_objective(x, beta):
    pts = x.reshape(N, 2)
    if not np.all(np.isfinite(pts)):
        return 1e10
    d = _pairwise_distances(pts)
    np.fill_diagonal(d, np.inf)
    boundary = 1.0 - np.sqrt(np.sum(pts ** 2, axis=1))
    pair_half = d / 2.0
    all_vals = np.concatenate([pair_half.ravel(), boundary])
    all_vals = all_vals[np.isfinite(all_vals)]
    if len(all_vals) == 0:
        return 1e10
    return -_softmin_logsumexp(all_vals, beta)


# ---------------------------- initialisation ----------------------------

def _ring_to_points(radii, angles_list):
    pts = []
    for i in range(len(radii)):
        r = radii[i]
        a = angles_list[i]
        if len(a) == 0:
            continue
        pts.append(np.column_stack([r * np.cos(a), r * np.sin(a)]))
    return np.vstack(pts)


def _ring_init(counts):
    n_rings = len(counts)
    radii0 = np.linspace(0.0, 0.85, n_rings)
    radii0[0] = 0.0
    angles_list = []
    for i, n_pts in enumerate(counts):
        if n_pts == 1:
            angles_list.append(np.array([0.0]))
        else:
            angles_list.append(np.linspace(0, 2 * np.pi, n_pts, endpoint=False))
    pts = _ring_to_points(radii0, angles_list)
    return pts[:N]


def _hexagonal_init(scale=0.12):
    points = []
    for i in range(-12, 13):
        for j in range(-12, 13):
            x = scale * (i + 0.5 * j)
            y = scale * (np.sqrt(3) / 2 * j)
            if x * x + y * y <= 0.81:
                points.append([x, y])
    points = np.array(points)
    if len(points) == 0:
        return None
    dists = np.sqrt(np.sum(points ** 2, axis=1))
    idx = np.argsort(dists)
    points = points[idx[:N]]
    return points


def _random_init():
    return np.random.uniform(-0.8, 0.8, (N, 2))


# --------------------------- optimisation stages ---------------------------

def _anneal_points(pts, deadline):
    """Multi-stage L-BFGS on the smooth soft-min objective."""
    current = pts.copy()
    current_score = _score(current)
    betas = [3.0, 8.0, 20.0, 50.0, 120.0, 300.0, 700.0, 1500.0, 3000.0, 6000.0, 12000.0, 25000.0]
    for beta in betas:
        if time.time() > deadline - 8:
            break
        try:
            res = minimize(
                _smooth_objective,
                current.ravel(),
                args=(beta,),
                method='L-BFGS-B',
                options={'maxiter': 800, 'ftol': 1e-16, 'gtol': 1e-14, 'maxfun': 20000},
            )
            new_pts = res.x.reshape(N, 2)
            if not np.all(np.isfinite(new_pts)):
                continue
            s = _score(new_pts)
            if s > current_score:
                current = new_pts
                current_score = s
        except Exception:
            pass
    return current, current_score


def _sa_polish(pts, deadline):
    """Simulated annealing on raw coordinates to escape local minima."""
    current = pts.copy()
    current_score = _score(current)
    best_points = current.copy()
    best_score = current_score

    T = 0.05
    T_min = 1e-6
    alpha = 0.995
    step_size = 0.05

    while time.time() < deadline - 10:
        new_points = current + np.random.normal(0, step_size, (N, 2))
        new_score = _score(new_points)
        if new_score > current_score:
            current = new_points
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_points = current.copy()
        else:
            if T > 0:
                delta = new_score - current_score
                if random.random() < np.exp(delta / T):
                    current = new_points
                    current_score = new_score
        T *= alpha
        if T < T_min:
            T = T_min
            if random.random() < 0.01:
                kick_idx = np.random.randint(N)
                current[kick_idx] += np.random.normal(0, 0.1, 2)
                current_score = _score(current)
        step_size *= 0.999
        if step_size < 1e-4:
            step_size = 1e-4

    return best_points, best_score


def _constraint_push(pts, r_target, deadline, max_iters=400):
    """Iteratively push violating pair/boundary constraints apart."""
    pts = pts.copy()
    for _ in range(max_iters):
        if time.time() > deadline - 4:
            break
        d = _pairwise_distances(pts)
        np.fill_diagonal(d, np.inf)
        pair_margins = d / 2.0
        boundary_margins = 1.0 - np.sqrt(np.sum(pts ** 2, axis=1))
        min_pair = np.min(pair_margins)
        min_bound = np.min(boundary_margins)
        if min_pair < min_bound - 1e-14:
            i, j = np.unravel_index(np.argmin(pair_margins), pair_margins.shape)
            if i == j:
                break
            di = pts[i] - pts[j]
            dist = np.linalg.norm(di)
            if dist < 1e-15:
                break
            di /= dist
            push = (2.0 * r_target - dist) / 2.0
            if push > 1e-15:
                pts[i] += push * di
                pts[j] -= push * di
        else:
            i = int(np.argmin(boundary_margins))
            pi = pts[i]
            dist = np.linalg.norm(pi)
            if dist < 1e-15:
                pts[i] += 1e-4 * np.array([1.0, 0.0])
            else:
                di = pi / dist
                push = (1.0 - r_target - dist)
                if push > 1e-15:
                    pts[i] += push * di
        d = _pairwise_distances(pts)
        np.fill_diagonal(d, np.inf)
        r = min(np.min(d) / 2.0, np.min(1.0 - np.sqrt(np.sum(pts ** 2, axis=1))))
        if r >= r_target - 1e-12:
            break
    return pts


def _binary_search_polish(pts, deadline):
    """Binary search the maximum feasible radius via constraint push."""
    current = pts.copy()
    low = _score(current) - 1e-8
    high = _score(current) + 1e-4
    for _ in range(28):
        if time.time() > deadline - 3:
            break
        mid = (low + high) / 2.0
        polished = _constraint_push(current, mid, deadline, max_iters=150)
        s = _score(polished)
        if s >= mid - 1e-10:
            low = mid
            current = polished
        else:
            high = mid
    return current, _score(current)


# ------------------------------- driver -------------------------------

def construct():
    t0 = time.time()
    deadline = t0 + DEADLINE

    best_points = None
    best_score = -1.0

    ring_structures = [
        [1, 6, 12, 18, 13],
        [1, 6, 12, 14, 17],
        [1, 6, 12, 16, 15],
        [1, 6, 12, 15, 16],
        [1, 6, 10, 14, 19],
        [1, 6, 13, 13, 17],
        [1, 6, 12, 12, 19],
        [1, 6, 11, 13, 19],
    ]

    candidates = []
    for counts in ring_structures:
        pts = _ring_init(counts)
        s = _score(pts)
        if s > 0:
            candidates.append((pts.copy(), s))

    hex_pts = _hexagonal_init(scale=0.12)
    if hex_pts is not None and len(hex_pts) == N:
        hs = _score(hex_pts)
        if hs > 0:
            candidates.append((hex_pts.copy(), hs))

    rnd_pts = _random_init()
    rs = _score(rnd_pts)
    if rs > 0:
        candidates.append((rnd_pts.copy(), rs))

    candidates.sort(key=lambda x: x[1], reverse=True)

    # Anneal top candidates (most of the budget).
    max_candidates = min(5, len(candidates))
    for idx in range(max_candidates):
        if time.time() > deadline - 40:
            break
        pts, s = candidates[idx]
        polished_pts, polished_score = _anneal_points(pts, deadline)
        if polished_score > best_score:
            best_score = polished_score
            best_points = polished_pts.copy()

    if best_points is None:
        best_points = candidates[0][0].copy()
        best_score = _score(best_points)

    # SA escape on the best.
    if time.time() < deadline - 25:
        sa_pts, sa_score = _sa_polish(best_points, deadline)
        if sa_score > best_score:
            best_score = sa_score
            best_points = sa_pts.copy()

    # Final binary-search / constraint polish.
    if time.time() < deadline - 12:
        best_points, best_score = _binary_search_polish(best_points, deadline)

    # Last-ditch SA polish if time remains.
    if time.time() < deadline - 5:
        sa_pts, sa_score = _sa_polish(best_points, deadline)
        if sa_score > best_score:
            best_score = sa_score
            best_points = sa_pts.copy()

    return best_points
