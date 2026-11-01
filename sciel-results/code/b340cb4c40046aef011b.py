"""Radial-ring packing of 300 equal circles in the unit disk.
Circles are placed in concentric rings; ring radii and angular offsets are
optimized (via a smooth min surrogate) to maximize the common radius."""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 300
_TIME_LIMIT = 110.0  # safety margin under 120s
_T0 = time.time()


def _elapsed():
    return time.time() - _T0


def _ring_counts():
    """Choose a ring count distribution: center + rings with increasing counts.
    Counts grow roughly linearly to match the area density of a hexagonal pack."""
    # Start with center (1), then rings with counts 6, 12, 18, ...
    counts = [1]
    total = 1
    k = 1
    while total < N:
        c = 6 * k
        counts.append(c)
        total += c
        k += 1
    # Trim the last ring so the total is exactly N
    excess = total - N
    if excess > 0:
        last = counts[-1]
        counts[-1] = max(1, last - excess)
        # If still short (shouldn't be), adjust
    # Pad if under (shouldn't happen)
    while sum(counts) < N:
        counts[-1] += 1
    return counts


def _build_positions(radii, offsets, counts):
    """Given ring radii, angular offsets, and per-ring counts, return (N,2) array."""
    pts = []
    for m, (r, off, n) in enumerate(zip(radii, offsets, counts)):
        if n == 1:
            pts.append([0.0, 0.0])
        else:
            angles = off + 2.0 * np.pi * np.arange(n) / n
            pts.append(np.column_stack([r * np.cos(angles), r * np.sin(angles)]))
    return np.vstack(pts)


def _min_pairwise_dist(pts):
    """Minimum distance between any pair of points (O(N^2) but N=300 is fine)."""
    n = len(pts)
    if n < 2:
        return np.inf
    # Compute all pairwise distances via broadcasting in chunks to save memory
    min_d = np.inf
    chunk = 64
    for i in range(0, n, chunk):
        block = pts[i:i + chunk]
        # distances from block to all points with index > i
        for j in range(i + 1, n, chunk):
            block2 = pts[j:j + chunk]
            d = np.linalg.norm(block[:, None, :] - block2[None, :, :], axis=2)
            # only keep upper triangle within this block pair
            m_i = d.shape[0]
            m_j = d.shape[1]
            # We need pairs where global index a < b. Since block is rows i.. and block2 is cols j..,
            # all a in [i,i+m_i) and b in [j,j+m_j) have a < b only if i + a_idx < j + b_idx.
            # Simplest: just take min over all, but ensure we don't double count / self.
            # Since i < j, every (a in block, b in block2) has global a < global b. Good.
            min_d = min(min_d, np.min(d))
    return min_d


def _boundary_margin(pts):
    return 1.0 - np.max(np.linalg.norm(pts, axis=1))


def _score(pts):
    return min(_min_pairwise_dist(pts) / 2.0, _boundary_margin(pts))


def _smooth_min_surrogate(vals, temperature=0.01):
    """Log-sum-exp based soft-min: approximates min(vals)."""
    vals = np.asarray(vals, dtype=float)
    m = np.min(vals)
    return m - temperature * np.log(np.sum(np.exp(-(vals - m) / temperature)))


def _objective(params, counts, temperature):
    """Objective to MINIMIZE: negative of the smooth-min of the critical distances.
    params = [radii..., offsets...]"""
    nr = len(counts)
    radii = params[:nr]
    offsets = params[nr:]
    # Keep radii non-negative and within [0, 1]
    radii = np.clip(radii, 0.0, 1.0)
    pts = _build_positions(radii, offsets, counts)
    # Critical distances: pairwise min and boundary
    d_pair = _min_pairwise_dist(pts)
    d_bound = _boundary_margin(pts)
    # We want to maximize min(d_pair/2, d_bound). Minimize negative smooth-min.
    vals = np.array([d_pair / 2.0, d_bound])
    return -_smooth_min_surrogate(vals, temperature)


def _local_polish(x0, counts, temperature=0.01):
    nr = len(counts)
    bounds = [(0.0, 1.0)] * nr + [(0.0, 2.0 * np.pi)] * nr
    res = minimize(_objective, x0, args=(counts, temperature),
                   method='L-BFGS-B', bounds=bounds,
                   options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-10})
    return res


def _initial_guess(counts):
    """Heuristic initial radii: place rings so adjacent-ring and in-ring
    distances are balanced. Start with a hexagonal-like radial spacing."""
    nr = len(counts)
    # Total area ~ pi. Approximate ring area ~ 2*pi*r_m*dr. For equal circles
    # of radius r, density ~ 0.9069. Use a simple heuristic:
    # r_m roughly proportional to sqrt of cumulative count.
    cum = np.cumsum(counts)
    # Normalize so outermost ring is near 1 - r (margin)
    # Estimate common radius from area: N * pi * r^2 * (1/0.9069) ~ pi
    r_est = np.sqrt(0.9069 / N)
    # Radii: r_m = r_est * (m + 0.5) * something... use sqrt scaling to fill area
    # Area up to ring m ~ pi * r_m^2 should hold ~ cum[m] circles:
    # pi * r_m^2 * 0.9069 ~ cum[m] * pi * r_est^2  => r_m = r_est * sqrt(cum[m]/0.9069)
    radii = r_est * np.sqrt(cum / 0.9069)
    # Cap at 1 - r_est
    radii = np.clip(radii, 0.0, 1.0 - r_est)
    # Ensure monotonic increasing
    for i in range(1, nr):
        if radii[i] <= radii[i-1]:
            radii[i] = radii[i-1] + 1e-4
    radii = np.clip(radii, 0.0, 1.0 - r_est)
    # Offsets: stagger alternating rings by half a step
    offsets = np.array([0.0 if (m % 2 == 0) else np.pi / max(1, counts[m]) for m in range(nr)])
    return np.concatenate([radii, offsets])


def construct():
    counts = _ring_counts()
    nr = len(counts)
    x0 = _initial_guess(counts)

    # First, a coarse global search with differential evolution if time permits
    bounds = [(0.0, 1.0)] * nr + [(0.0, 2.0 * np.pi)] * nr
    best_x = x0.copy()
    best_score = _score(_build_positions(best_x[:nr], best_x[nr:], counts))

    # Try a few random restarts + local polish
    rng = np.random.default_rng(42)
    attempts = 0
    max_attempts = 40
    while _elapsed() < _TIME_LIMIT and attempts < max_attempts:
        attempts += 1
        if attempts == 1:
            x_try = x0.copy()
        else:
            # Perturb around best
            x_try = best_x + rng.normal(0, 0.05, size=len(best_x))
            x_try[:nr] = np.clip(x_try[:nr], 0.0, 1.0)
            x_try[nr:] = (x_try[nr:] + np.pi) % (2 * np.pi) - np.pi
            x_try[nr:] = (x_try[nr:] + np.pi) % (2 * np.pi)
        res = _local_polish(x_try, counts, temperature=0.01)
        if res.x is not None:
            s = _score(_build_positions(res.x[:nr], res.x[nr:], counts))
            if s > best_score:
                best_score = s
                best_x = res.x.copy()

    # Final: use the best parameters
    radii = np.clip(best_x[:nr], 0.0, 1.0)
    offsets = best_x[nr:]
    pts = _build_positions(radii, offsets, counts)

    # Safety: ensure all points are strictly inside the unit disk and finite
    norms = np.linalg.norm(pts, axis=1)
    if np.any(~np.isfinite(pts)) or np.any(norms >= 1.0):
        # Fallback: hexagonal lattice
        return _hexagonal_fallback()
    return pts


def _hexagonal_fallback():
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        mask = radius <= 1.0 - spacing / 2.0
        points, radius = points[mask], radius[mask]
        if len(points) >= N:
            return points[np.argsort(radius, kind="stable")[:N]]
        spacing *= 0.98


if __name__ == "__main__":
    pts = construct()
    print("shape:", pts.shape)
    print("score:", _score(pts))
