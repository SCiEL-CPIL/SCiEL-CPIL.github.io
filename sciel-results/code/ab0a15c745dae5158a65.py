import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

N = 50
DEADLINE = 132.0  # wall-clock safety margin before the 150s kill


# ---------------------------------------------------------------------------
# Scoring / smooth objective
# ---------------------------------------------------------------------------
def _pairwise_distances(pts):
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=-1))


def _score(pts):
    """Compute the packing score: largest common radius."""
    if pts is None or len(pts) != N:
        return -1.0
    pts = np.asarray(pts, dtype=float)
    if not np.all(np.isfinite(pts)):
        return -1.0

    d = _pairwise_distances(pts)
    np.fill_diagonal(d, np.inf)
    min_pair = np.min(d) / 2.0
    min_boundary = np.min(1.0 - np.sqrt(np.sum(pts ** 2, axis=1)))
    val = min(min_pair, min_boundary)
    if val <= 0:
        return -1.0
    return val


def _softmin_logsumexp(values, beta):
    """Numerically-stable soft-min via log-sum-exp."""
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
    """Smooth objective for optimization: minimize negative soft-min of margins."""
    pts = x.reshape(N, 2)
    if not np.all(np.isfinite(pts)):
        return 1e10
    d = _pairwise_distances(pts)
    np.fill_diagonal(d, np.inf)
    boundary = 1.0 - np.sqrt(np.sum(pts ** 2, axis=1))
    pair_half = d / 2.0
    # Combine all constraints: pairwise distances (half) and boundary distances
    all_vals = np.concatenate([pair_half.ravel(), boundary])
    all_vals = all_vals[np.isfinite(all_vals)]
    if len(all_vals) == 0:
        return 1e10
    return -_softmin_logsumexp(all_vals, beta)


# ---------------------------------------------------------------------------
# Ring parameter helpers
# ---------------------------------------------------------------------------
def _ring_to_points(radii, angles_list):
    """Convert ring parameters (radii, list of angle arrays) to point coordinates."""
    if len(radii) == 0:
        return np.empty((0, 2))
    pts = [np.array([0.0, 0.0])]  # Center
    for i in range(1, len(radii)):
        r = radii[i]
        a = angles_list[i]
        pts.append(np.column_stack([r * np.cos(a), r * np.sin(a)]))
    return np.vstack(pts)


def _flatten_ring(radii, angles_list):
    """Flatten ring parameters into a single vector for optimization."""
    vec = list(radii)
    for i in range(1, len(angles_list)):
        vec.extend(list(angles_list[i]))
    return np.array(vec)


def _unflatten_ring(vec, ring_sizes):
    """Unflatten optimization vector back to radii and angles."""
    radii = vec[:len(ring_sizes)].copy()
    angles_list = [np.array([0.0])]
    idx = len(ring_sizes)
    for i in range(1, len(ring_sizes)):
        n = ring_sizes[i]
        angles_list.append(vec[idx:idx + n].copy())
        idx += n
    return radii, angles_list


# ---------------------------------------------------------------------------
# Constraint push polish
# ---------------------------------------------------------------------------
def _constraint_push(pts, r_target, deadline, max_iters=400):
    """Polish solution by iteratively pushing violating constraints with relaxation."""
    pts = pts.copy()
    relaxation = 1.0
    prev_score = _score(pts)

    for _ in range(max_iters):
        if time.time() > deadline - 3:
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
            push = (2.0 * r_target - dist) / 2.0 * relaxation
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
                push = (1.0 - r_target - dist) * relaxation
                if push > 1e-15:
                    pts[i] += push * di

        current_score = _score(pts)
        if current_score < prev_score - 1e-10:
            relaxation *= 0.7
        elif current_score > prev_score + 1e-12:
            relaxation = min(relaxation * 1.1, 1.0)
        prev_score = current_score

        if current_score >= r_target - 1e-12:
            break

    return pts


# ---------------------------------------------------------------------------
# Multi-stage annealing on full point coordinates
# ---------------------------------------------------------------------------
def _anneal_points(pts, deadline):
    """Multi-stage annealing optimization on full point coordinates."""
    current = pts.copy()
    current_score = _score(current)

    betas = [3.0, 8.0, 20.0, 50.0, 100.0, 200.0, 400.0, 800.0, 1600.0, 3200.0, 6400.0]

    for beta in betas:
        if time.time() > deadline - 4:
            break
        try:
            res = minimize(
                _smooth_objective,
                current.ravel(),
                args=(beta,),
                method='L-BFGS-B',
                options={
                    'maxiter': 1500,
                    'ftol': 1e-16,
                    'gtol': 1e-14,
                    'maxfun': 30000
                }
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


# ---------------------------------------------------------------------------
# DE ring search for a given topology
# ---------------------------------------------------------------------------
def _de_ring_search(ring_sizes, deadline, seed=42):
    """Run DE search for a given ring topology."""
    n_rings = len(ring_sizes)
    if n_rings <= 1:
        return None, -1.0

    cum_counts = np.cumsum(ring_sizes)
    total = cum_counts[-1]
    init_radii = np.zeros(n_rings)
    for i in range(n_rings):
        if i == 0:
            init_radii[i] = 0.0
        else:
            frac = cum_counts[i] / total
            init_radii[i] = np.sqrt(frac) * 0.88

    angles_list = []
    for i, n in enumerate(ring_sizes):
        if i == 0:
            angles_list.append(np.array([0.0]))
        else:
            angles_list.append(np.linspace(0, 2 * np.pi, n, endpoint=False))

    bounds = []
    for i in range(n_rings):
        if i == 0:
            bounds.append((0.0, 0.0))
        else:
            bounds.append((0.0, 0.95))
    for i in range(1, n_rings):
        bounds.extend([(0, 2 * np.pi)] * ring_sizes[i])

    def ring_obj(vec):
        if time.time() > deadline - 5:
            return 1e10
        radii, angles_list = _unflatten_ring(vec, ring_sizes)
        radii = np.clip(radii, 0.0, 0.95)
        pts = _ring_to_points(radii, angles_list)
        return -_smooth_objective(pts.ravel(), beta=30.0)

    try:
        result = differential_evolution(
            ring_obj, bounds,
            maxiter=150, popsize=20, tol=1e-10,
            mutation=(0.3, 1.0), recombination=0.8,
            seed=seed, polish=False
        )
        x_opt = result.x
        radii_opt, angles_opt = _unflatten_ring(x_opt, ring_sizes)
        radii_opt = np.clip(radii_opt, 0.0, 0.95)
        pts_opt = _ring_to_points(radii_opt, angles_opt)
        s = _score(pts_opt)
        return pts_opt, s
    except Exception:
        return None, -1.0


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE

    best_points = None
    best_score = -1.0

    def update_best(pts):
        nonlocal best_points, best_score
        if pts is None:
            return
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_points = pts.copy()
        return s

    # --- Stage 1: Multi-topology DE search ---
    topologies = [
        [1, 6, 12, 18, 13],   # classic 5-ring
        [1, 8, 14, 27],       # 4-ring
        [1, 5, 10, 15, 19],   # alternate 5-ring
        [1, 10, 20, 19],      # 4-ring wide
    ]

    for topo in topologies:
        if time.time() > deadline - 40:
            break
        pts_topo, s_topo = _de_ring_search(topo, deadline, seed=42 + len(topo))
        if pts_topo is not None:
            update_best(pts_topo)

    if best_points is None:
        # Fallback: simple hexagonal lattice
        spacing = 0.24
        points = []
        for j in range(-10, 11):
            for i in range(-10, 11):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if np.sqrt(x * x + y * y) <= 1.0 - spacing / 2.0:
                    points.append([x, y])
        points = np.array(points)
        if len(points) > N:
            d = np.linalg.norm(points, axis=1)
            points = points[np.argsort(d, kind='stable')[:N]]
        best_points = points
        best_score = _score(best_points)

    # --- Stage 2: Multi-stage annealing on full coordinates ---
    if time.time() < deadline - 10:
        best_points, best_score = _anneal_points(best_points, deadline)

    # --- Stage 3: Constraint push polish ---
    if time.time() < deadline - 3 and best_score > 0:
        r_target = best_score
        polished = _constraint_push(best_points, r_target, deadline)
        update_best(polished)

    # --- Stage 4: Final tight Nelder-Mead polish ---
    if time.time() < deadline - 1 and best_points is not None:
        def tight_obj(vec):
            if time.time() > deadline - 1:
                return 1e10
            pts = vec.reshape(N, 2)
            return -_score(pts)

        try:
            res = minimize(tight_obj, best_points.ravel(), method='Nelder-Mead',
                           options={'maxiter': 20000, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True})
            pts_final = res.x.reshape(N, 2)
            update_best(pts_final)
        except Exception:
            pass

    return best_points
