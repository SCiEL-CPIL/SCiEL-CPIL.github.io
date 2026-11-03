"""Rewritten program: multi-start aspect-ratio search with gradient-based expansion."""
import numpy as np
from scipy.optimize import minimize, LinearConstraint
import time
import math

N = 21

# EVOLVE-BLOCK-START

def _build_pairwise_indices(n):
    """Precompute pairwise indices for O(n^2) distance checks."""
    iu = np.triu_indices(n, k=1)
    return iu

def _check_validity(circles, W, H, tol=1e-6):
    """Check if circles are valid: within rectangle and non-overlapping."""
    cx = circles[:, 0]
    cy = circles[:, 1]
    r = circles[:, 2]
    # Boundary check
    if np.any(cx - r < -tol) or np.any(cx + r > W + tol):
        return False
    if np.any(cy - r < -tol) or np.any(cy + r > H + tol):
        return False
    # Pairwise distance check
    iu = _build_pairwise_indices(len(circles))
    if len(iu[0]) == 0:
        return True
    dx = cx[iu[0]] - cx[iu[1]]
    dy = cy[iu[0]] - cy[iu[1]]
    dist = np.sqrt(dx*dx + dy*dy)
    min_dist = r[iu[0]] + r[iu[1]]
    if np.any(dist < min_dist - tol):
        return False
    return True

def _sum_radii(circles):
    return np.sum(circles[:, 2])

def _initial_grid(n, W, H):
    """Create a grid-like initial placement of n circles in a W x H rectangle."""
    # Choose grid dimensions
    aspect = W / H
    if aspect >= 1:
        cols = int(round(math.sqrt(n * aspect)))
        rows = int(math.ceil(n / cols))
    else:
        rows = int(round(math.sqrt(n / aspect)))
        cols = int(math.ceil(n / rows))
    
    # Ensure we have at least n cells
    while cols * rows < n:
        if cols <= rows:
            cols += 1
        else:
            rows += 1
    
    positions = []
    for idx in range(n):
        row = idx // cols
        col = idx % cols
        # Center in each cell
        x = (col + 0.5) * W / cols
        y = (row + 0.5) * H / rows
        positions.append([x, y, 0.01])  # small initial radius
    
    return np.array(positions)

def _expand_and_optimize(circles, W, H, max_time=10.0):
    """
    Try to expand circles while maintaining non-overlap and boundary constraints.
    Uses a gradient-based approach to maximize sum of radii.
    """
    n = len(circles)
    iu = _build_pairwise_indices(n)
    n_pairs = len(iu[0])
    
    # Decision variables: [x1, y1, r1, x2, y2, r2, ...]
    # We'll optimize positions and radii jointly.
    
    x0 = circles.flatten().copy()
    
    # Constraints:
    # 1. Boundary: xi - ri >= 0, W - xi - ri >= 0, yi - ri >= 0, H - yi - ri >= 0
    # 2. Non-overlap: (xi-xj)^2 + (yi-yj)^2 >= (ri+rj)^2 -> linearized approximation
    #    For smooth optimization, use: xi-xj >= (ri+rj)*cos(theta) ... 
    #    Better: use the constraint dist_ij >= ri + rj directly with a smooth penalty
    #    or use linear constraints based on current angles.
    
    # Let's use a different approach: fix positions, expand radii, then adjust positions.
    # Actually, let's do a proper nonlinear optimization.
    
    def objective(v):
        """Negative sum of radii (to minimize)."""
        r = v[2::3]
        return -np.sum(r)
    
    # We'll use a penalty method for the non-overlap constraints
    def penalty(v):
        cx = v[0::3]
        cy = v[1::3]
        r = v[2::3]
        pen = 0.0
        # Boundary penalties
        pen += np.sum(np.maximum(0, (r - cx))**2)
        pen += np.sum(np.maximum(0, (r - (W - cx)))**2)
        pen += np.sum(np.maximum(0, (r - cy))**2)
        pen += np.sum(np.maximum(0, (r - (H - cy)))**2)
        # Non-overlap penalties
        if n_pairs > 0:
            dx = cx[iu[0]] - cx[iu[1]]
            dy = cy[iu[0]] - cy[iu[1]]
            dist = np.sqrt(dx*dx + dy*dy)
            min_dist = r[iu[0]] + r[iu[1]]
            overlap = np.maximum(0, min_dist - dist)
            pen += np.sum(overlap**2)
        return pen
    
    def objective_with_penalty(v, mu):
        return objective(v) + mu * penalty(v)
    
    # Start with a small penalty weight and increase
    mu = 1.0
    v = x0.copy()
    start_time = time.time()
    
    for iteration in range(50):
        if time.time() - start_time > max_time:
            break
        
        # Optimize with current penalty weight
        res = minimize(
            lambda v: objective_with_penalty(v, mu),
            v,
            method='L-BFGS-B',
            options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        v = res.x
        
        # Check if constraints are satisfied
        cx = v[0::3]
        cy = v[1::3]
        r = v[2::3]
        
        # Check boundary violations
        bnd_viol = np.max([
            np.max(np.maximum(0, r - cx)),
            np.max(np.maximum(0, r - (W - cx))),
            np.max(np.maximum(0, r - cy)),
            np.max(np.maximum(0, r - (H - cy)))
        ])
        
        # Check non-overlap violations
        if n_pairs > 0:
            dx = cx[iu[0]] - cx[iu[1]]
            dy = cy[iu[0]] - cy[iu[1]]
            dist = np.sqrt(dx*dx + dy*dy)
            min_dist = r[iu[0]] + r[iu[1]]
            ov_viol = np.max(np.maximum(0, min_dist - dist))
        else:
            ov_viol = 0.0
        
        total_viol = max(bnd_viol, ov_viol)
        
        if total_viol < 1e-8:
            break
        else:
            mu *= 10.0
            if mu > 1e8:
                break
    
    # Reconstruct circles
    cx = v[0::3]
    cy = v[1::3]
    r = v[2::3]
    new_circles = np.column_stack([cx, cy, r])
    
    # Clamp radii to be positive and within bounds
    r = np.maximum(r, 1e-6)
    r = np.minimum(r, np.minimum(np.minimum(cx, W - cx), np.minimum(cy, H - cy)))
    new_circles[:, 2] = r
    
    return new_circles

def _optimize_for_aspect(W, H, max_time=15.0):
    """Optimize packing for a given rectangle with perimeter 2*(W+H)=4, i.e., W+H=2."""
    # Initial grid
    circles = _initial_grid(N, W, H)
    
    # Expand and optimize
    circles = _expand_and_optimize(circles, W, H, max_time=max_time)
    
    # Check validity
    if not _check_validity(circles, W, H):
        # Try to repair by shrinking radii
        cx = circles[:, 0]
        cy = circles[:, 1]
        r = circles[:, 2]
        # Shrink to fit boundaries
        r = np.minimum(r, np.minimum(np.minimum(cx, W - cx), np.minimum(cy, H - cy)))
        r = np.maximum(r, 1e-6)
        circles[:, 2] = r
        # Check pairwise
        iu = _build_pairwise_indices(N)
        if len(iu[0]) > 0:
            dx = cx[iu[0]] - cx[iu[1]]
            dy = cy[iu[0]] - cy[iu[1]]
            dist = np.sqrt(dx*dx + dy*dy)
            # Scale down overlapping pairs
            for _ in range(10):
                min_dist = circles[iu[0], 2] + circles[iu[1], 2]
                overlaps = dist < min_dist - 1e-9
                if not np.any(overlaps):
                    break
                # Reduce radii of overlapping circles
                idx0 = iu[0][overlaps]
                idx1 = iu[1][overlaps]
                for i0, i1 in zip(idx0, idx1):
                    avg_r = (circles[i0, 2] + circles[i1, 2]) / 2
                    max_r = (dist[np.where((iu[0] == i0) & (iu[1] == i1))[0][0]] - 1e-6) / 2
                    if max_r > 0:
                        circles[i0, 2] = min(circles[i0, 2], max_r)
                        circles[i1, 2] = min(circles[i1, 2], max_r)
    
    return circles

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    best_score = -np.inf
    best_circles = None
    
    # Try a range of aspect ratios
    # W + H = 2, so H = 2 - W
    # W in [0.5, 1.5] for reasonable aspect ratios
    aspect_ratios = np.linspace(0.5, 1.5, 11)
    
    for W in aspect_ratios:
        if time.time() > deadline:
            break
        H = 2.0 - W
        if H <= 0.01:
            continue
        
        time_left = deadline - time.time()
        if time_left < 5.0:
            break
        
        max_iter_time = min(15.0, time_left * 0.8)
        circles = _optimize_for_aspect(W, H, max_time=max_iter_time)
        
        if circles is not None and len(circles) == N:
            if _check_validity(circles, W, H):
                score = _sum_radii(circles)
                if score > best_score:
                    best_score = score
                    best_circles = circles.copy()
    
    # If no valid result found, fall back to a simple grid in a square
    if best_circles is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        best_circles = np.array(cells[:N])
        # Scale to fit in a 1x1 square (perimeter 4)
        # The 1x1 square has perimeter 4, which is correct.
    
    return best_circles

# EVOLVE-BLOCK-END
