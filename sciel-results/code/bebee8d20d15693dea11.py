"""Optimized sphere covering with 130 points — multi-restart Lloyd + smooth-max gradient."""
import numpy as np
import time
from scipy.spatial import ConvexHull

N = 130
_DEADLINE = time.time() + 140  # 10s safety margin

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms

def _time_ok(margin=5.0):
    return time.time() < _DEADLINE - margin

def _covering_radius(pts):
    """Compute the covering radius in radians via convex hull facet circumcenters."""
    pts = _normalize(pts)
    try:
        hull = ConvexHull(pts)
    except Exception:
        return np.pi / 2

    max_r = 0.0
    for simplex in hull.simplices:
        a, b, c = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
        v1, v2 = b - a, c - a
        cross = np.cross(v1, v2)
        cnorm = np.linalg.norm(cross)
        if cnorm < 1e-12:
            continue
        x = cross / cnorm
        centroid = (a + b + c) / 3.0
        if np.dot(x, centroid) < 0:
            x = -x
        dot_val = np.clip(np.dot(x, a), -1.0, 1.0)
        r = np.arccos(dot_val)
        if r > max_r:
            max_r = r
    return max_r

def _covering_radius_fast(pts):
    """Alias for _covering_radius."""
    return _covering_radius(pts)

def _compute_facet_data(pts):
    """
    Compute circumcenters and facet indices for all facets of the convex hull.
    Returns (CC, facet_indices, max_r) where:
      CC: (F, 3) array of circumcenter directions (unit vectors)
      facet_indices: (F, 3) array of point indices per facet
      max_r: covering radius in radians
    """
    pts = _normalize(pts.copy())
    try:
        hull = ConvexHull(pts)
    except Exception:
        return None, None, np.pi / 2

    CC_list = []
    facet_indices = []
    max_r = 0.0

    for simplex in hull.simplices:
        a, b, c = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
        v1, v2 = b - a, c - a
        cross = np.cross(v1, v2)
        cnorm = np.linalg.norm(cross)
        if cnorm < 1e-12:
            continue
        x = cross / cnorm
        centroid = (a + b + c) / 3.0
        if np.dot(x, centroid) < 0:
            x = -x
        dot_val = np.clip(np.dot(x, a), -1.0, 1.0)
        r = np.arccos(dot_val)
        if r > max_r:
            max_r = r
        CC_list.append(x)
        facet_indices.append(simplex)

    if not CC_list:
        return None, None, max_r

    CC = np.array(CC_list)
    facet_indices = np.array(facet_indices)
    return CC, facet_indices, max_r

def _fibonacci_spiral(n, jitter=0.0, seed=0):
    """Fibonacci spiral on the unit sphere with optional jitter."""
    rng = np.random.RandomState(seed)
    i = np.arange(n) + 0.5
    if jitter > 0:
        i = i + rng.randn(n) * jitter
        i = np.clip(i, 0.5, n - 0.5)
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _normalize(pts)

def _polish_largest_facets(pts, max_iters=50, lr=0.05, beta=20.0):
    """
    Iteratively polish the configuration by moving points toward the circumcenters
    of their largest adjacent facets. This is a targeted local refinement.
    """
    pts = _normalize(pts.copy())
    best_pts = pts.copy()
    best_r = _covering_radius_fast(pts)

    for it in range(max_iters):
        if not _time_ok(3.0):
            break

        CC, facet_indices, cr = _compute_facet_data(pts)
        if CC is None:
            break

        if cr < best_r:
            best_r = cr
            best_pts = pts.copy()

        # For each point, find adjacent facets and their circumcenter distances
        new_pts = pts.copy()
        for i in range(N):
            mask = np.any(facet_indices == i, axis=1)
            if not np.any(mask):
                continue
            cc_i = CC[mask]
            dots = np.clip(cc_i @ pts[i], -1.0, 1.0)
            dists = np.arccos(dots)
            # Smooth-max weight: emphasize the farthest circumcenter
            shifted = dists - np.max(dists)
            w = np.exp(beta * shifted)
            w_sum = np.sum(w)
            if w_sum < 1e-15:
                continue
            w /= w_sum
            # Weighted centroid of circumcenters
            target = w @ cc_i
            tn = np.linalg.norm(target)
            if tn < 1e-12:
                continue
            target /= tn
            new_pts[i] = pts[i] + lr * (target - pts[i])

        new_pts = _normalize(new_pts)

        # Accept if improved
        new_r = _covering_radius_fast(new_pts)
        if new_r <= cr + 1e-10:
            pts = new_pts
            if new_r < best_r:
                best_r = new_r
                best_pts = pts.copy()
        else:
            # Reduce learning rate if not improving
            lr *= 0.9
            if lr < 0.001:
                break

    return best_pts, best_r

def _lloyd_step(pts, lr=0.08, smooth_beta=15.0):
    """
    One Lloyd-type step using convex hull facet circumcenters.
    """
    pts = _normalize(pts.copy())
    CC, facet_indices, _ = _compute_facet_data(pts)
    if CC is None:
        return pts

    new_pts = pts.copy()
    for i in range(N):
        mask = np.any(facet_indices == i, axis=1)
        if not np.any(mask):
            continue
        cc_i = CC[mask]
        dots = np.clip(cc_i @ pts[i], -1.0, 1.0)
        dists = np.arccos(dots)
        shifted = dists - np.max(dists)
        w = np.exp(smooth_beta * shifted)
        w_sum = np.sum(w)
        if w_sum < 1e-15:
            continue
        w /= w_sum
        target = w @ cc_i
        tn = np.linalg.norm(target)
        if tn < 1e-12:
            continue
        target /= tn
        new_pts[i] = pts[i] + lr * (target - pts[i])

    return _normalize(new_pts)

def _gradient_descent(pts, max_iters=200, lr_init=0.02, lr_decay=0.995, beta=15.0):
    """
    Gradient descent on a smooth approximation of the covering radius.
    """
    pts = _normalize(pts.copy())
    best_pts = pts.copy()
    best_r = _covering_radius_fast(pts)
    lr = lr_init

    for it in range(max_iters):
        if not _time_ok(5.0):
            break

        CC, facet_indices, cr = _compute_facet_data(pts)
        if CC is None:
            break

        if cr < best_r:
            best_r = cr
            best_pts = pts.copy()

        # For each point, compute gradient toward weighted circumcenter
        grad = np.zeros_like(pts)
        for i in range(N):
            mask = np.any(facet_indices == i, axis=1)
            if not np.any(mask):
                continue
            cc_i = CC[mask]
            dots = np.clip(cc_i @ pts[i], -1.0, 1.0)
            dists = np.arccos(dots)
            shifted = dists - np.max(dists)
            w = np.exp(beta * shifted)
            w_sum = np.sum(w)
            if w_sum < 1e-15:
                continue
            w /= w_sum
            target = w @ cc_i
            tn = np.linalg.norm(target)
            if tn < 1e-12:
                continue
            target /= tn
            grad[i] = target - pts[i]

        # Normalize gradient magnitude per point
        gnorms = np.linalg.norm(grad, axis=1, keepdims=True)
        gnorms[gnorms < 1e-12] = 1.0
        grad /= gnorms

        # Take step
        new_pts = _normalize(pts + lr * grad)

        # Evaluate and accept
        new_r = _covering_radius_fast(new_pts)
        if new_r <= cr + 1e-10:
            pts = new_pts
            if new_r < best_r:
                best_r = new_r
                best_pts = pts.copy()
        else:
            lr *= 0.95

        lr *= lr_decay
        if lr < 0.001:
            break

    return best_pts, best_r

def _simulated_annealing(pts, max_iters=100, t_init=0.05, t_decay=0.97, t_min=0.001):
    """
    Simulated annealing to escape local minima.
    """
    pts = _normalize(pts.copy())
    best_pts = pts.copy()
    best_r = _covering_radius_fast(pts)
    T = t_init
    rng = np.random.RandomState(42)

    for it in range(max_iters):
        if not _time_ok(3.0):
            break
        if T < t_min:
            break

        perturb = rng.randn(N, 3) * T
        new_pts = _normalize(pts + perturb)
        new_r = _covering_radius_fast(new_pts)
        delta = new_r - best_r

        if delta < 0 or rng.rand() < np.exp(-delta / T):
            pts = new_pts
            if new_r < best_r:
                best_r = new_r
                best_pts = new_pts.copy()

        T *= t_decay

    return best_pts, best_r

def construct():
    """
    Multi-restart strategy:
    1. Generate diverse initial configs.
    2. For each, run Lloyd + gradient descent + simulated annealing.
    3. Final high-precision polish on the best.
    """
    start_time = time.time()
    best_pts = None
    best_r = np.pi

    # Generate diverse initial configurations
    inits = []
    for seed in [42, 123, 7, 999, 31415, 271828, 161803, 8675309, 14159265]:
        inits.append(_fibonacci_spiral(N, jitter=0.03, seed=seed))
    inits.append(_fibonacci_spiral(N, jitter=0.0, seed=0))
    rng = np.random.RandomState(555)
    inits.append(_normalize(rng.randn(N, 3)))
    for seed in [111, 222, 333, 444, 555]:
        rng = np.random.RandomState(seed)
        inits.append(_normalize(rng.randn(N, 3)))

    # Sort by initial covering radius, try best first
    init_scores = []
    for p in inits:
        try:
            r = _covering_radius_fast(p)
        except Exception:
            r = np.pi
        init_scores.append((r, p))
    init_scores.sort(key=lambda x: x[0])

    for r0, p0 in init_scores:
        if not _time_ok(10.0):
            break

        # Phase 1: Lloyd iterations
        pts = p0.copy()
        for lloyd_iter in range(25):
            if not _time_ok(8.0):
                break
            pts = _lloyd_step(pts, lr=0.08, smooth_beta=15.0)

        cr_lloyd = _covering_radius_fast(pts)
        if cr_lloyd < best_r:
            best_r = cr_lloyd
            best_pts = pts.copy()

        # Phase 2: Gradient descent refinement
        if _time_ok(10.0):
            pts2, cr2 = _gradient_descent(pts, max_iters=150, lr_init=0.015, lr_decay=0.99, beta=12.0)
            if cr2 < best_r:
                best_r = cr2
                best_pts = pts2.copy()

        # Phase 3: Simulated annealing
        if _time_ok(8.0):
            base = best_pts if best_pts is not None else pts
            pts3, cr3 = _simulated_annealing(base, max_iters=80, t_init=0.04, t_decay=0.97)
            if cr3 < best_r:
                best_r = cr3
                best_pts = pts3.copy()

    # Final high-precision polish
    if best_pts is not None and _time_ok(5.0):
        polished, cr_polished = _polish_largest_facets(best_pts, max_iters=40, lr=0.03, beta=25.0)
        if cr_polished < best_r:
            best_r = cr_polished
            best_pts = polished

        # One more round of very fine gradient descent
        if _time_ok(4.0):
            final_pts, cr_final = _gradient_descent(best_pts, max_iters=100, lr_init=0.005, lr_decay=0.995, beta=20.0)
            if cr_final < best_r:
                best_r = cr_final
                best_pts = final_pts

    if best_pts is None:
        best_pts = _fibonacci_spiral(N)

    return _normalize(best_pts)

# Test
if __name__ == "__main__":
    t0 = time.time()
    pts = construct()
    print("Shape:", pts.shape)
    print("Covering radius (deg):", np.degrees(_covering_radius_fast(pts)))
    print("Time: %.1f s" % (time.time() - t0))
