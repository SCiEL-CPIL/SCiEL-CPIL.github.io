"""Circle packing in unit square n=32."""
import numpy as np
import time
import random
from scipy.optimize import minimize

N = 32
# Wall-clock safety margin (evaluator kills at 150s; use 140s for safety)
DEADLINE = 140.0


def _time_left():
    return DEADLINE - time.time()


def _pairwise_dist2(pos):
    """pos: (n,2) -> (n,n) squared distances"""
    diff = pos[:, None, :] - pos[None, :, :]
    return np.sum(diff * diff, axis=-1)


def _validate_and_fix(circles, eps=1e-9):
    """Ensure circles are inside [0,1]^2 and non-overlapping.
    Returns a valid array of shape (n,3) or None if impossible (shouldn't happen)."""
    circles = np.array(circles, dtype=float)
    n = len(circles)
    # Clip radii to be positive and not too large
    circles[:, 2] = np.clip(circles[:, 2], 1e-9, 0.5)
    # Clip centers
    circles[:, 0] = np.clip(circles[:, 0], circles[:, 2], 1.0 - circles[:, 2])
    circles[:, 1] = np.clip(circles[:, 1], circles[:, 2], 1.0 - circles[:, 2])
    # Iteratively resolve overlaps by shrinking radii
    for _ in range(50):
        d2 = _pairwise_dist2(circles[:, :2])
        np.fill_diagonal(d2, np.inf)
        min_d = np.sqrt(d2)
        r_sum = circles[:, 2][:, None] + circles[:, 2][None, :]
        # Where r_sum > min_d, we have overlap
        overlap = np.maximum(r_sum - min_d, 0)
        if np.max(overlap) < eps:
            break
        # Shrink each radius proportionally to its share of overlap
        max_overlap_per_i = np.max(overlap, axis=1)
        # Reduce radius by half the max overlap needed
        circles[:, 2] -= 0.5 * max_overlap_per_i
        circles[:, 2] = np.clip(circles[:, 2], 1e-9, 0.5)
    return circles


def _objective(x, n=N):
    """x: (2n,) -> centers. Objective = -sum(radii) where radii are the max feasible
    given centers. We compute the max radius for each circle given the centers."""
    pos = x.reshape(n, 2)
    d2 = _pairwise_dist2(pos)
    np.fill_diagonal(d2, np.inf)
    d = np.sqrt(d2)
    # Distance to boundary for each circle
    dist_x = np.minimum(pos[:, 0], 1.0 - pos[:, 0])
    dist_y = np.minimum(pos[:, 1], 1.0 - pos[:, 1])
    dist_bound = np.minimum(dist_x, dist_y)
    # For each circle i, max radius = min(dist_bound[i], min_j (d[i,j] - r_j)...)
    # This is coupled. Use iterative approach or just compute min over j of d[i,j]/2
    # Actually, the max radius for circle i given others is complex.
    # Instead, use a different formulation: minimize a penalty for overlaps.
    # Let's use: for each pair (i,j), constraint d_ij >= r_i + r_j
    # and for each i, r_i <= dist_bound_i
    # Objective: maximize sum r_i
    # This is a nonlinear program. Let's use scipy minimize with constraints.
    # But that's slow for many calls. Let's use a simpler approach:
    # Define r_i = min(dist_bound_i, min_j d_ij / 2) as a lower bound estimate.
    # Actually, a better approach: use the fact that at optimum, each circle is
    # tangent to at least 3 others or boundary.
    # For now, let's just use a penalty-based objective.
    # r_i = dist_bound_i * 0.99 (slightly conservative)
    # Then check overlaps and penalize.
    # Better: use a smooth surrogate.
    # Let's define the objective as -sum(r_i) where r_i is computed as:
    # r_i = min(dist_bound_i, min_{j != i} d_ij / 2)
    # This gives a valid packing (each circle has radius at most half the distance to nearest neighbor).
    r = np.minimum(dist_bound, np.min(d, axis=1) / 2.0)
    r = np.maximum(r, 1e-9)
    return -np.sum(r)


def _construct_from_centers(pos):
    """Given centers, compute max feasible radii."""
    n = len(pos)
    d2 = _pairwise_dist2(pos)
    np.fill_diagonal(d2, np.inf)
    d = np.sqrt(d2)
    dist_x = np.minimum(pos[:, 0], 1.0 - pos[:, 0])
    dist_y = np.minimum(pos[:, 1], 1.0 - pos[:, 1])
    dist_bound = np.minimum(dist_x, dist_y)
    # Max radius for each circle is min(dist_bound, min_j d_ij / 2)
    # But this is conservative. A better estimate uses the fact that
    # at optimum, r_i + r_j = d_ij for some pairs.
    # For now, use the conservative estimate.
    r = np.minimum(dist_bound, np.min(d, axis=1) / 2.0)
    r = np.maximum(r, 1e-9)
    circles = np.column_stack([pos, r])
    return _validate_and_fix(circles)


def _grid_init(n=N, side=6):
    """Initialize centers on a grid with some randomness."""
    centers = []
    for i in range(side):
        for j in range(side):
            if len(centers) >= n:
                break
            x = (i + 0.5 + random.uniform(-0.2, 0.2)) / side
            y = (j + 0.5 + random.uniform(-0.2, 0.2)) / side
            x = np.clip(x, 0.05, 0.95)
            y = np.clip(y, 0.05, 0.95)
            centers.append([x, y])
    # If we don't have enough, add random points
    while len(centers) < n:
        centers.append([random.uniform(0.1, 0.9), random.uniform(0.1, 0.9)])
    return np.array(centers[:n])


def _random_init(n=N):
    """Random initialization."""
    return np.array([[random.uniform(0.1, 0.9), random.uniform(0.1, 0.9)] for _ in range(n)])


def _optimize_centers(pos0, max_iter=500, timeout=None):
    """Optimize centers to maximize sum of radii."""
    n = len(pos0)
    x0 = pos0.flatten()

    def obj(x):
        return _objective(x, n)

    # Use L-BFGS-B
    try:
        result = minimize(obj, x0, method='L-BFGS-B',
                          options={'maxiter': max_iter, 'maxfun': 5000, 'ftol': 1e-10, 'gtol': 1e-8})
        return result.x.reshape(n, 2)
    except Exception:
        return pos0


def _perturb_and_optimize(pos, scale=0.05, max_iter=300):
    """Perturb centers and re-optimize."""
    n = len(pos)
    best_pos = pos
    best_obj = _objective(pos.flatten(), n)

    for _ in range(10):
        if timeout is not None and _time_left() < 5:
            break
        pos_pert = pos + np.random.randn(n, 2) * scale
        pos_pert = np.clip(pos_pert, 0.01, 0.99)
        pos_opt = _optimize_centers(pos_pert, max_iter=max_iter)
        obj_val = _objective(pos_opt.flatten(), n)
        if obj_val < best_obj:  # Remember objective is negative sum of radii
            best_obj = obj_val
            best_pos = pos_opt
    return best_pos


def construct():
    start_time = time.time()
    best_circles = None
    best_score = -np.inf

    # Multiple restarts with different initializations
    for trial in range(50):
        if _time_left() < 10:
            break

        # Different initialization strategies
        if trial % 3 == 0:
            pos0 = _grid_init(N, side=6)
        elif trial % 3 == 1:
            pos0 = _random_init(N)
        else:
            # Hexagonal-like packing init
            pos0 = np.array([[0.15 + 0.3 * (i % 3) + random.uniform(-0.05, 0.05),
                              0.15 + 0.3 * (i // 3) + random.uniform(-0.05, 0.05)]
                             for i in range(N)])
            pos0 = np.clip(pos0, 0.05, 0.95)

        # Optimize
        pos_opt = _optimize_centers(pos0, max_iter=500)
        circles = _construct_from_centers(pos_opt)
        score = np.sum(circles[:, 2])

        if score > best_score:
            best_score = score
            best_circles = circles

        # Try perturbation around the best so far
        if _time_left() > 15:
            pos_pert = _perturb_and_optimize(pos_opt, scale=0.03, max_iter=200)
            circles_pert = _construct_from_centers(pos_pert)
            score_pert = np.sum(circles_pert[:, 2])
            if score_pert > best_score:
                best_score = score_pert
                best_circles = circles_pert

    if best_circles is None:
        # Fallback to grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        best_circles = np.array(cells[:N])

    return best_circles
