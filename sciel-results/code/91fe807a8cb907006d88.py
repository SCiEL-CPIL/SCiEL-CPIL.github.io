"""Packing 50 equal circles in a unit square: simulated annealing + local refinement."""
import time
import numpy as np
from scipy.optimize import minimize

N = 50
DEADLINE = 110.0  # safety margin before 120s kill


def _score(pts):
    """Compute the max radius for given centres inside [-0.5, 0.5]^2."""
    if pts.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    # Check inside square
    if np.any(pts < -0.5 - 1e-12) or np.any(pts > 0.5 + 1e-12):
        return 0.0
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist) / 2.0
    # Distance to walls
    wall = np.min(np.minimum(0.5 - np.abs(pts[:, 0]), 0.5 - np.abs(pts[:, 1])))
    return min(min_pair, wall)


def _initial_grid():
    """7x7 grid minus 2 corners to get 50 points, then refine."""
    m = 7
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    # Remove 2 points to get exactly 50 (remove two from the last row)
    pts = grid[:N].copy()
    return pts


def _random_start():
    """Random initial configuration."""
    return np.random.uniform(-0.45, 0.45, size=(N, 2))


def _sa_optimize(pts, time_budget):
    """Simulated annealing to maximize the minimum distance."""
    pts = pts.copy()
    current_score = _score(pts)
    best_pts = pts.copy()
    best_score = current_score

    T = 0.05
    T_min = 1e-6
    alpha = 0.9995
    step_size = 0.05

    start_time = time.time()
    n_steps = 0

    while time.time() - start_time < time_budget:
        if T < T_min:
            break

        # Pick a random point to move
        i = np.random.randint(N)
        old_pos = pts[i].copy()
        pts[i, 0] += np.random.normal(0, step_size)
        pts[i, 1] += np.random.normal(0, step_size)

        # Keep inside square
        pts[i, 0] = np.clip(pts[i, 0], -0.49, 0.49)
        pts[i, 1] = np.clip(pts[i, 1], -0.49, 0.49)

        new_score = _score(pts)
        delta = new_score - current_score

        if delta >= 0 or np.random.random() < np.exp(delta / T):
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_pts = pts.copy()
        else:
            pts[i] = old_pos

        T *= alpha
        n_steps += 1

        # Reduce step size gradually
        if n_steps % 1000 == 0:
            step_size *= 0.98

    return best_pts, best_score


def _local_refine(pts):
    """Local refinement using scipy minimize to push apart close pairs."""
    pts = pts.copy()

    for _ in range(5):
        # Find the minimum distance pair
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dist, np.inf)
        min_idx = np.unravel_index(np.argmin(dist), dist.shape)
        i, j = min_idx

        # Also check wall constraints
        wall_dists = np.min(np.minimum(0.5 - np.abs(pts[:, 0]), 0.5 - np.abs(pts[:, 1])), axis=1)
        min_wall_idx = np.argmin(wall_dists)

        # Optimize to increase the bottleneck
        def objective(x):
            p = x.reshape(-1, 2)
            # Penalty for going outside
            outside = np.maximum(0, np.abs(p) - 0.5).sum()
            if outside > 1e-9:
                return -outside
            return -_score(p)

        x0 = pts.flatten()
        res = minimize(objective, x0, method='Nelder-Mead',
                       options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10})
        new_pts = res.x.reshape(-1, 2)
        new_pts = np.clip(new_pts, -0.4999, 0.4999)
        if _score(new_pts) >= _score(pts) - 1e-12:
            pts = new_pts

    return pts


def construct():
    np.random.seed(42)
    start_time = time.time()

    # Start with a grid-based initial solution
    pts = _initial_grid()
    best_pts = pts.copy()
    best_score = _score(pts)

    # Phase 1: Simulated annealing with grid start
    remaining = DEADLINE - (time.time() - start_time)
    if remaining > 5:
        budget1 = min(30, remaining - 5)
        pts1, score1 = _sa_optimize(pts, budget1)
        if score1 > best_score:
            best_pts = pts1
            best_score = score1

    # Phase 2: Local refinement
    remaining = DEADLINE - (time.time() - start_time)
    if remaining > 10:
        refined = _local_refine(best_pts)
        refined_score = _score(refined)
        if refined_score > best_score:
            best_pts = refined
            best_score = refined_score

    # Phase 3: More SA from refined solution if time allows
    remaining = DEADLINE - (time.time() - start_time)
    if remaining > 15:
        budget2 = min(20, remaining - 10)
        pts2, score2 = _sa_optimize(best_pts, budget2)
        if score2 > best_score:
            best_pts = pts2
            best_score = score2

    # Phase 4: Final local refinement
    remaining = DEADLINE - (time.time() - start_time)
    if remaining > 5:
        refined = _local_refine(best_pts)
        refined_score = _score(refined)
        if refined_score > best_score:
            best_pts = refined
            best_score = refined_score

    # Try a few more random restarts if time remains
    remaining = DEADLINE - (time.time() - start_time)
    restart = 0
    while remaining > 20 and restart < 5:
        pts_r = _random_start()
        budget_r = min(10, remaining - 15)
        if budget_r < 5:
            break
        pts_r, score_r = _sa_optimize(pts_r, budget_r)
        if score_r > best_score:
            best_pts = pts_r
            best_score = score_r
        # Quick local refine
        remaining = DEADLINE - (time.time() - start_time)
        if remaining > 10:
            refined = _local_refine(best_pts)
            refined_score = _score(refined)
            if refined_score > best_score:
                best_pts = refined
                best_score = refined_score
        restart += 1
        remaining = DEADLINE - (time.time() - start_time)

    # Final clip to ensure validity
    best_pts = np.clip(best_pts, -0.499999, 0.499999)

    # Verify
    final_score = _score(best_pts)
    if final_score < best_score - 1e-10:
        # Something went wrong, use a safe grid
        best_pts = _initial_grid()

    return best_pts
