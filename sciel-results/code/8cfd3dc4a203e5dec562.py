import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import cdist
import time
import itertools
import random

def squares_in_square10() -> tuple:
    n = 10
    start_time = time.time()
    deadline = start_time + 580  # Safety margin

    def elapsed():
        return time.time() - start_time

    # Baseline: 4x4 grid
    k = 4
    side = 1.0 / k
    idx = np.arange(n)
    cols = idx % k
    rows = idx // k
    cx_base = (cols + 0.5) * side
    cy_base = (rows + 0.5) * side
    centers_base = np.stack([cx_base, cy_base], axis=1)
    sides_base = np.full(n, side)
    best_score = float(np.sum(sides_base))
    best_centers = centers_base.copy()
    best_sides = sides_base.copy()

    tol = 1e-7

    def is_valid(centers, sides):
        """Check if configuration is valid (no overlaps, within bounds)."""
        if np.any(sides <= 0) or np.any(~np.isfinite(sides)) or np.any(~np.isfinite(centers)):
            return False
        for i in range(n):
            cx, cy = centers[i]
            s = sides[i]
            if cx - s/2 < -tol or cx + s/2 > 1 + tol:
                return False
            if cy - s/2 < -tol or cy + s/2 > 1 + tol:
                return False
        for i in range(n):
            for j in range(i+1, n):
                t = (sides[i] + sides[j]) / 2 - tol
                if abs(centers[i][0] - centers[j][0]) < t and abs(centers[i][1] - centers[j][1]) < t:
                    return False
        return True

    def objective(x):
        centers = x[:2*n].reshape(n, 2)
        sides = x[2*n:]
        if np.any(sides < 1e-8):
            return 1e8
        # Penalty for boundary violations
        p = 0.0
        for i in range(n):
            cx, cy = centers[i]
            s = sides[i]
            if cx - s/2 < 0:
                p += (cx - s/2) ** 2
            if cx + s/2 > 1:
                p += (cx + s/2 - 1) ** 2
            if cy - s/2 < 0:
                p += (cy - s/2) ** 2
            if cy + s/2 > 1:
                p += (cy + s/2 - 1) ** 2
        # Penalty for overlaps
        for i in range(n):
            for j in range(i+1, n):
                dx = centers[i][0] - centers[j][0]
                dy = centers[i][1] - centers[j][1]
                t = (sides[i] + sides[j]) / 2
                if abs(dx) < t and abs(dy) < t:
                    p += (t - abs(dx)) ** 2 + (t - abs(dy)) ** 2
        return -np.sum(sides) + 100.0 * p

    def try_local_refinement(x0):
        """Try to refine x0 using SLSQP."""
        if elapsed() > deadline - 10:
            return None, None
        try:
            bounds = [(0, 1)] * (2*n) + [(1e-6, 1)] * n
            res = minimize(
                objective, x0, method='SLSQP',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-9}
            )
            if elapsed() > deadline - 10:
                return None, None
            c, s = res.x[:2*n].reshape(n, 2), res.x[2*n:]
            if is_valid(c, s):
                score = float(np.sum(s))
                return res.x, score
        except Exception:
            pass
        return None, None

    # 2. Run DE with multiple restarts
    bounds = [(0, 1) for _ in range(2*n)] + [(1e-6, 1) for _ in range(n)]
    
    seeds = [42, 123, 456, 789, 111, 222, 333, 444]
    for seed in seeds:
        if elapsed() > deadline - 60:
            break
        try:
            result = differential_evolution(
                objective,
                bounds,
                maxiter=400,
                popsize=10,
                tol=1e-6,
                seed=seed,
                polish=False,
                init='sobol'
            )
            if elapsed() > deadline - 30:
                break
            c, s = result.x[:2*n].reshape(n, 2), result.x[2*n:]
            if is_valid(c, s):
                score = float(np.sum(s))
                if score > best_score:
                    best_centers = c.copy()
                    best_sides = s.copy()
                    best_score = score
                    # Try local refinement
                    refined, refined_score = try_local_refinement(result.x)
                    if refined is not None and refined_score > best_score:
                        c2, s2 = refined[:2*n].reshape(n, 2), refined[2*n:]
                        best_centers = c2.copy()
                        best_sides = s2.copy()
                        best_score = refined_score
        except Exception:
            pass

    # 3. Try from the grid baseline with local refinement
    if elapsed() < deadline - 30:
        x0 = np.concatenate([centers_base.flatten(), sides_base])
        refined, refined_score = try_local_refinement(x0)
        if refined is not None and refined_score > best_score:
            c2, s2 = refined[:2*n].reshape(n, 2), refined[2*n:]
            best_centers = c2.copy()
            best_sides = s2.copy()
            best_score = refined_score

    # 4. Final validation and cleanup
    best_centers = np.clip(best_centers, 0, 1)
    best_sides = np.clip(best_sides, 1e-6, 1)
    
    if not is_valid(best_centers, best_sides):
        best_centers = centers_base.copy()
        best_sides = sides_base.copy()

    return best_centers.astype(np.float64), best_sides.astype(np.float64)
