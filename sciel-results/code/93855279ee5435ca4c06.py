import numpy as np
import time
from scipy.optimize import minimize


def _compute_scores(points):
    """
    Vectorized vertex-score computation.
    Returns an array of 20 scores in [0, 1].
    """
    n = len(points)
    # Pairwise distances: (n, n)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))

    # For each vertex i, sort distances to the other n-1 vertices.
    # Set self-distance to +inf so it sorts to the end, then drop it.
    dists_self_inf = dists.copy()
    np.fill_diagonal(dists_self_inf, np.inf)
    sorted_d = np.sort(dists_self_inf, axis=1)  # (n, n), last col is inf
    d = sorted_d[:, :n - 1]  # (n, n-1)

    scores = np.zeros(n)

    # Check: any distance < 0.05 → score 0
    has_close = np.any(d < 0.05, axis=1)

    # Gaps: d_{k+1} - d_k, shape (n, n-2)
    gaps = d[:, 1:] - d[:, :-1]
    k_idx = np.argmin(gaps, axis=1)  # (n,)

    # D = (d_k + d_{k+1}) / 2
    D = (d[np.arange(n), k_idx] + d[np.arange(n), k_idx + 1]) / 2.0

    # Validity mask
    valid = (~has_close) & (D >= 0.1)

    # Select 4 distances based on k
    # k=0 → d[0:4], k=n-3 → d[n-5:n-1], else d[k-1:k+3]
    # We handle this with three masks and gather.
    n_minus_1 = n - 1  # 19
    n_minus_3 = n - 3  # 17

    selected = np.zeros((n, 4))

    mask_k0 = valid & (k_idx == 0)
    mask_k17 = valid & (k_idx == n_minus_3)
    mask_mid = valid & (k_idx > 0) & (k_idx < n_minus_3)

    # k = 0: d[0], d[1], d[2], d[3]
    for i in np.where(mask_k0)[0]:
        selected[i] = d[i, 0:4]

    # k = n-3 (17): d[n-5], d[n-4], d[n-3], d[n-2]  i.e. d[15:19]
    for i in np.where(mask_k17)[0]:
        selected[i] = d[i, n_minus_5:n_minus_1] if False else d[i, 15:19]

    # Middle: d[k-1], d[k], d[k+1], d[k+2]
    for i in np.where(mask_mid)[0]:
        k = k_idx[i]
        selected[i] = d[i, k - 1:k + 3]

    # raw = mean of max(d/D, D/d) over the 4 selected
    with np.errstate(divide='ignore', invalid='ignore'):
        ratios = np.maximum(selected / D[:, np.newaxis], D[:, np.newaxis] / selected)
    raw = np.mean(ratios, axis=1)

    # score = min(1, 1/raw), only for valid
    with np.errstate(divide='ignore', invalid='ignore'):
        score_vals = np.minimum(1.0, 1.0 / raw)
    scores = np.where(valid, score_vals, 0.0)

    return scores


def _min_vertex_score(radii):
    """
    radii: array of 20 radii (positive).
    Returns the minimum vertex score after normalization and convexity check.
    """
    angles = _angles  # precomputed
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    points = np.column_stack((x, y))

    # Normalize
    centroid = np.mean(points, axis=0)
    points = points - centroid
    dists_to_centroid = np.linalg.norm(points, axis=1)
    mean_dist = np.mean(dists_to_centroid)
    if mean_dist <= 0.01:
        return 0.0
    points = points / mean_dist

    # Convexity check
    p = points
    p_next = np.roll(p, -1, axis=0)
    p_next2 = np.roll(p, -2, axis=0)
    e1 = p_next - p
    e2 = p_next2 - p_next
    cross = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]

    if np.any(cross == 0):
        return 0.0
    if np.any(cross > 0) and np.any(cross < 0):
        return 0.0

    # No repeated points
    if np.min(np.linalg.norm(p_next - p, axis=1)) < 1e-9:
        return 0.0

    scores = _compute_scores(points)
    return float(np.min(scores))


def equidistant_points_polygon20() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 600.0  # 30s safety margin before 630s kill

    n = 20
    global _angles
    _angles = np.linspace(0, 2 * np.pi, n, endpoint=False)

    def objective(r):
        r = np.maximum(r, 0.05)
        return -_min_vertex_score(r)

    # --- Measure per-eval cost with a quick probe ---
    probe_start = time.time()
    _min_vertex_score(np.ones(n))
    eval_time = max(time.time() - probe_start, 1e-4)

    # --- Build a diverse set of initializations ---
    inits = []
    # 1. Regular
    inits.append(np.ones(n))
    # 2. Wavy cos(2θ)
    inits.append(np.maximum(1.0 + 0.15 * np.cos(2 * _angles), 0.05))
    # 3. Wavy cos(4θ)
    inits.append(np.maximum(1.0 + 0.15 * np.cos(4 * _angles), 0.05))
    # 4. Wavy cos(5θ)
    inits.append(np.maximum(1.0 + 0.15 * np.cos(5 * _angles), 0.05))
    # 5. Wavy cos(3θ)
    inits.append(np.maximum(1.0 + 0.10 * np.cos(3 * _angles), 0.05))
    # 6. Random perturbation
    rng = np.random.default_rng(12345)
    inits.append(np.maximum(1.0 + 0.08 * rng.standard_normal(n), 0.05))
    # 7. Random perturbation (different seed)
    rng2 = np.random.default_rng(67890)
    inits.append(np.maximum(1.0 + 0.12 * rng2.standard_normal(n), 0.05))
    # 8. Wavy cos(6θ)
    inits.append(np.maximum(1.0 + 0.10 * np.cos(6 * _angles), 0.05))
    # 9. Wavy cos(7θ)
    inits.append(np.maximum(1.0 + 0.10 * np.cos(7 * _angles), 0.05))
    # 10. Slight radial gradient (asymmetric)
    inits.append(np.maximum(1.0 + 0.05 * np.arange(n) / n, 0.05))

    best_score = -1.0
    best_radii = np.ones(n)

    # Determine max iterations per restart based on measured eval time
    # Nelder-Mead in 20D: roughly ~20 function evals per iteration
    # Budget per restart: ~30% of remaining time, at least 5s
    max_iter_base = max(200, int(30.0 / max(eval_time * 20, 1e-3)))
    max_iter_cap = 5000

    for idx, x0 in enumerate(inits):
        remaining = deadline - time.time()
        if remaining < 10.0:
            break

        # Adaptive max_iter: use what fits in ~40% of remaining time
        max_iter = min(max_iter_cap, max(200, int(remaining * 0.4 / max(eval_time * 20, 1e-3))))

        try:
            res = minimize(
                objective, x0, method='Nelder-Mead',
                options={
                    'maxiter': max_iter,
                    'xatol': 1e-10,
                    'fatol': 1e-10,
                    'adaptive': True,
                }
            )
            if res.fun < best_score:
                best_score = -res.fun
                best_radii = res.x
        except Exception:
            pass

    # --- If we still have time, do a second pass with finer tolerance on the best ---
    remaining = deadline - time.time()
    if remaining > 60.0 and best_score > 0:
        try:
            res2 = minimize(
                objective, best_radii, method='Nelder-Mead',
                options={
                    'maxiter': min(max_iter_cap, int(remaining * 0.7 / max(eval_time * 20, 1e-3))),
                    'xatol': 1e-12,
                    'fatol': 1e-12,
                    'adaptive': True,
                }
            )
            if res2.fun < best_score:
                best_score = -res2.fun
                best_radii = res2.x
        except Exception:
            pass

    # --- Construct final polygon ---
    radii = np.maximum(best_radii, 0.05)
    x = radii * np.cos(_angles)
    y = radii * np.sin(_angles)
    polygon = np.column_stack((x, y))

    # Final normalization
    centroid = np.mean(polygon, axis=0)
    polygon = polygon - centroid
    dists = np.linalg.norm(polygon, axis=1)
    mean_dist = np.mean(dists)
    if mean_dist > 0.01:
        polygon = polygon / mean_dist

    # Verify finite
    if not np.all(np.isfinite(polygon)):
        # Fallback to regular polygon
        polygon = np.column_stack((np.cos(_angles), np.sin(_angles)))

    return polygon
