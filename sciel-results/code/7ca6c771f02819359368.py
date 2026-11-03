"""Packing 50 equal circles in a unit circle: ring-based init + soft-min annealing + constraint polish."""
import time
import numpy as np
from scipy.optimize import minimize

N = 50
DEADLINE = 135.0  # safety margin before 150s kill


def _pairwise_distances(pts):
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=-1))


def _score(pts):
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
    if val <= 0:
        return 0.0
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
    """Smooth objective: minimize negative soft-min of margins."""
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


def _ring_to_points(radii, angles_list):
    """Convert ring parameters to point coordinates."""
    pts = []
    for i in range(len(radii)):
        r = radii[i]
        a = angles_list[i]
        if len(a) == 0:
            continue
        pts.append(np.column_stack([r * np.cos(a), r * np.sin(a)]))
    return np.vstack(pts)


def _hexagonal_init(scale=0.12):
    points = []
    for i in range(-10, 11):
        for j in range(-10, 11):
            x = scale * (i + 0.5 * j)
            y = scale * (np.sqrt(3) / 2 * j)
            if x**2 + y**2 <= 0.81:
                points.append([x, y])
    points = np.array(points)
    if len(points) == 0:
        return None
    dists = np.sqrt(np.sum(points**2, axis=1))
    idx = np.argsort(dists)
    points = points[idx[:N]]
    return points


def _anneal_points(pts, deadline, max_betas=10):
    """Multi-stage annealing optimization on full point coordinates."""
    current = pts.copy()
    current_score = _score(current)
    
    betas = [3.0, 8.0, 20.0, 50.0, 120.0, 300.0, 700.0, 1500.0, 3000.0, 6000.0, 12000.0, 25000.0]
    betas = betas[:max_betas]
    
    for beta in betas:
        if time.time() > deadline - 5:
            break
        try:
            res = minimize(
                _smooth_objective,
                current.ravel(),
                args=(beta,),
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-14, 'maxfun': 30000}
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


def _constraint_push(pts, r_target, deadline, max_iters=500):
    """Polish solution by iteratively pushing violating constraints."""
    pts = pts.copy()
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
    """Binary search for maximum feasible radius with projection polish."""
    current = pts.copy()
    low = _score(current) - 1e-8
    high = _score(current) + 1e-4
    
    for _ in range(30):
        if time.time() > deadline - 2:
            break
        mid = (low + high) / 2.0
        polished = _constraint_push(current, mid, deadline, max_iters=200)
        s = _score(polished)
        if s >= mid - 1e-10:
            low = mid
            current = polished
        else:
            high = mid
    
    return current, _score(current)


def construct():
    t0 = time.time()
    deadline = t0 + DEADLINE
    
    best_points = None
    best_score = -1.0
    
    # Ring topologies for 50 circles
    ring_structures = [
        [1, 6, 12, 18, 13],
        [1, 6, 12, 14, 17],
        [1, 6, 12, 16, 15],
        [1, 6, 12, 15, 16],
        [1, 6, 10, 14, 19],
        [1, 6, 13, 13, 17],
    ]
    
    candidates = []
    
    # Generate ring-based initializations
    for counts in ring_structures:
        n_rings = len(counts)
        radii0 = np.linspace(0, 0.85, n_rings)
        radii0[0] = 0.0
        
        angles_list = []
        for i, n_pts in enumerate(counts):
            if n_pts == 1:
                angles_list.append(np.array([0.0]))
            else:
                angles_list.append(np.linspace(0, 2 * np.pi, n_pts, endpoint=False))
        
        pts = _ring_to_points(radii0, angles_list)
        s = _score(pts)
        if s > 0:
            candidates.append((pts.copy(), s))
    
    # Hexagonal initialization
    hex_pts = _hexagonal_init(scale=0.12)
    if hex_pts is not None and len(hex_pts) == N:
        hex_score = _score(hex_pts)
        if hex_score > 0:
            candidates.append((hex_pts.copy(), hex_score))
    
    # Sort candidates by initial score (best first)
    candidates.sort(key=lambda x: x[1], reverse=True)
    
    # Anneal top candidates
    max_candidates_to_anneal = min(4, len(candidates))
    for idx in range(max_candidates_to_anneal):
        if time.time() > deadline - 30:
            break
        pts, s = candidates[idx]
        polished_pts, polished_score = _anneal_points(pts, deadline)
        if polished_score > best_score:
            best_score = polished_score
            best_points = polished_pts.copy()
    
    # If no good candidates, use hex
    if best_points is None:
        best_points = hex_pts.copy()
        best_score = _score(best_points)
    
    # Final binary search polish
    if time.time() < deadline - 10:
        best_points, best_score = _binary_search_polish(best_points, deadline)
    
    # Ensure validity: clamp points
    r_final = _score(best_points)
    if r_final > 0:
        r2 = np.hypot(best_points[:, 0], best_points[:, 1])
        for i in range(N):
            if r2[i] > 1 - r_final + 1e-10:
                scale = (1 - r_final) / r2[i]
                best_points[i] *= scale
    
    return best_points.reshape(N, 2)
