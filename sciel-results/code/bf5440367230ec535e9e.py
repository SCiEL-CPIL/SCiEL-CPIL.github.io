"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16
D = 2

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time
import math


def _pairwise_distances(pts):
    """Compute all pairwise distances for points of shape (n, 2)."""
    n = pts.shape[0]
    # Use broadcasting for clarity; n=16 is small
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    return dists


def _score(pts):
    """Compute the squared ratio of max to min pairwise distance.
    Returns (score, min_dist, max_dist). Higher is worse for the raw metric,
    but the search score is seed/value, so we want to minimize the raw metric.
    Actually, the task says: "score is (max pairwise distance / min pairwise distance)^2".
    "The evaluator measures the raw value (lower is better)."
    So we want to minimize this value.
    """
    dists = _pairwise_distances(pts)
    # Exclude diagonal (zero distance)
    iu = np.triu_indices_from(dists, k=1)
    d = dists[iu]
    min_d = np.min(d)
    max_d = np.max(d)
    if min_d < 1e-12:
        return np.inf, 0.0, max_d
    ratio_sq = (max_d / min_d) ** 2
    return ratio_sq, min_d, max_d


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    best_pts = None
    best_score = np.inf

    # Strategy: Represent 16 points as a centrally symmetric set.
    # We define 8 "base" points in the upper half-plane (or just 8 points)
    # and reflect them through the origin to get 16 points.
    # This enforces central symmetry, which is a strong structural constraint
    # that often leads to good packing/distribution properties.
    
    # Let the 8 base points be defined by angles and radii?
    # Or just 8 arbitrary points in R^2, and the full set is [p_1, ..., p_8, -p_1, ..., -p_8].
    # This reduces the dimension from 32 to 16.
    # We must ensure no point is at the origin (min_dist > 0) and points are distinct.
    
    # Initial guess:
    # A regular octagon scaled appropriately?
    # Or a grid-like structure?
    # Let's try a mix: some points on a circle, some inside?
    
    # Let's define the variable vector x of length 16 (8 points * 2 coords).
    # The full set of 16 points is constructed from x.
    
    def _unpack_and_expand(x):
        # x is shape (16,), representing 8 points
        base_pts = x.reshape(8, 2)
        # Reflect through origin
        reflected = -base_pts
        all_pts = np.vstack((base_pts, reflected))
        return all_pts

    def _objective(x):
        pts = _unpack_and_expand(x)
        score, min_d, max_d = _score(pts)
        # Add a penalty if points are too close or identical to avoid singularities
        if min_d < 1e-6:
            return score + 1e10 * (1e-6 - min_d)
        return score

    # Initial configurations
    inits = []
    
    # 1. Regular octagon (in first quadrant and reflected)
    # Actually, just 8 points on a circle, then reflect.
    # If we take 8 points on a circle, reflecting them gives 16 points on a circle.
    # A regular 16-gon is a good candidate.
    # Let's generate a regular 16-gon and take the first 8 as base.
    angles_16 = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    pts_16 = np.column_stack((np.cos(angles_16), np.sin(angles_16)))
    # Take first 8 as base
    base_1 = pts_16[:8].flatten()
    inits.append(base_1)
    
    # 2. Two concentric squares?
    # 4 points on outer square, 4 on inner square?
    # Outer square: (1,0), (0,1), (-1,0), (0,-1) -> rotated 45 deg
    # Inner square: (0.5, 0.5), (-0.5, 0.5), (-0.5, -0.5), (0.5, -0.5)
    # Let's just randomize a bit around the regular 16-gon.
    
    # 3. Random points
    np.random.seed(42)
    for _ in range(5):
        rand_base = np.random.randn(8, 2) * 1.0
        # Scale to avoid origin issues
        rand_base += 0.5
        inits.append(rand_base.flatten())
    
    # 4. Grid-like
    # 4x4 grid, take 8, reflect?
    # Grid points: (0,0)..(3,3). Centered at (1.5, 1.5).
    # This might have points at origin after centering?
    # Let's stick to the symmetry approach.
    
    best_x = None
    
    for i, init in enumerate(inits):
        if time.time() > deadline - 20:
            break
            
        try:
            res = minimize(
                _objective,
                init,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if res.success or res.fun < best_score:
                x_opt = res.x
                pts_opt = _unpack_and_expand(x_opt)
                score_opt, min_d, max_d = _score(pts_opt)
                
                # Check for distinctness
                if min_d > 1e-6:
                    if score_opt < best_score:
                        best_score = score_opt
                        best_pts = pts_opt
                        best_x = x_opt.copy()
                        
        except Exception:
            pass

    # If we have a good result from symmetry, try to refine it with full 32-dim optimization
    # starting from the symmetric solution. This allows breaking symmetry if it helps.
    if best_pts is not None and time.time() < deadline - 10:
        try:
            # Define objective for full 16 points
            def _objective_full(x):
                pts = x.reshape(16, 2)
                score, min_d, max_d = _score(pts)
                if min_d < 1e-6:
                    return score + 1e10 * (1e-6 - min_d)
                return score

            # Start from the symmetric solution
            x_start = best_pts.flatten()
            
            res_full = minimize(
                _objective_full,
                x_start,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if res_full.fun < best_score:
                pts_full = res_full.x.reshape(16, 2)
                score_full, min_d, max_d = _score(pts_full)
                if min_d > 1e-6:
                    best_score = score_full
                    best_pts = pts_full
                    
        except Exception:
            pass

    if best_pts is None:
        # Fallback to grid if optimization failed
        side = int(np.ceil(N ** (1 / D)))
        grid = []
        for i in range(side):
            for j in range(side):
                grid.append([i, j])
                if len(grid) == N:
                    break
            if len(grid) == N:
                break
        best_pts = np.array(grid, dtype=float)

    return best_pts

# EVOLVE-BLOCK-END
