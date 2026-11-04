"""Circle packing in unit square: pack 26 circles, maximize sum of radii.
Strategy: Vector-Field (Force-Directed) Global Optimizer with distance-to-obstacle attraction,
followed by high-resolution best-fit local search."""

import numpy as np
import time
import random
from scipy.optimize import minimize

N = 26

def _eval_packing(xy, r):
    """
    Evaluate packing validity and sum of radii.
    Returns (sum_radii, is_valid).
    """
    n = len(xy)
    for i in range(n):
        x, y, ri = xy[i, 0], xy[i, 1], r[i]
        if ri < -1e-9:
            return 0.0, False
        if x - ri < -1e-9 or 1.0 - x - ri < -1e-9 or y - ri < -1e-9 or 1.0 - y - ri < -1e-9:
            return 0.0, False
    for i in range(n):
        for j in range(i + 1, n):
            dx = xy[i, 0] - xy[j, 0]
            dy = xy[i, 1] - xy[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - 1e-9:
                return 0.0, False
    return float(np.sum(r)), True

def _grid_init(side):
    """Initialize circles in a grid pattern."""
    positions = []
    for i in range(side):
        for j in range(side):
            if len(positions) < N:
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side - 0.002
                positions.append([x, y, r])
    while len(positions) < N:
        positions.append([0.5, 0.5, 0.01])
    return np.array(positions[:N])

def _random_init(seed):
    """Initialize circles randomly."""
    rng = np.random.RandomState(seed)
    xy = rng.uniform(0.1, 0.9, size=(N, 2))
    r = rng.uniform(0.02, 0.05, size=N)
    return xy, r

def _vector_field_optimize(xy, r, deadline, grid_res=100, max_iter=3000, seed=0):
    """
    Optimize packing using a force-directed approach.
    Forces:
    1. Repulsion between overlapping circles (inverse distance).
    2. Attraction to empty space (gradient of distance-to-nearest-other-circle field).
    3. Wall constraints (soft penalty).
    
    This acts as a global optimizer by allowing circles to "flow" into gaps.
    """
    n = len(xy)
    xy = xy.copy()
    r = r.copy()
    
    # Precompute a coarse grid of "empty space" potential?
    # Actually, calculating the field on the fly is O(N^2) per step which is fine for N=26.
    # But we can use a grid to accelerate the "attraction to empty space" part.
    # Let's stick to direct calculation for simplicity and robustness, but use momentum.
    
    rng = np.random.RandomState(seed)
    momentum = np.zeros((n, 2))
    lr = 0.05
    decay = 0.95
    beta = 50.0 # Softness of repulsion
    
    best_xy = xy.copy()
    best_r = r.copy()
    best_score = 0.0
    
    # Initial score
    s, v = _eval_packing(xy, r)
    if v:
        best_score = s
        best_xy = xy.copy()
        best_r = r.copy()

    for it in range(max_iter):
        if time.time() > deadline:
            break
            
        # 1. Calculate Forces
        forces_x = np.zeros(n)
        forces_y = np.zeros(n)
        forces_r = np.zeros(n)
        
        # Wall forces
        # Push away from walls if too close
        for i in range(n):
            x, y, ri = xy[i, 0], xy[i, 1], r[i]
            # Left wall
            if x < ri:
                forces_x[i] += (ri - x) * 10.0
            # Right wall
            if x > 1.0 - ri:
                forces_x[i] -= (x - (1.0 - ri)) * 10.0
            # Bottom wall
            if y < ri:
                forces_y[i] += (ri - y) * 10.0
            # Top wall
            if y > 1.0 - ri:
                forces_y[i] -= (y - (1.0 - ri)) * 10.0
            
        # Pairwise forces
        for i in range(n):
            for j in range(i + 1, n):
                dx = xy[i, 0] - xy[j, 0]
                dy = xy[i, 1] - xy[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = r[i] + r[j]
                
                if dist < min_dist:
                    # Overlap: Repel
                    # Force magnitude proportional to overlap
                    overlap = min_dist - dist
                    if dist < 1e-6:
                        # Random kick if centers coincide
                        angle = rng.uniform(0, 2 * np.pi)
                        fx = np.cos(angle) * overlap
                        fy = np.sin(angle) * overlap
                    else:
                        fx = (dx / dist) * overlap
                        fy = (dy / dist) * overlap
                    
                    forces_x[i] += fx * 5.0
                    forces_y[i] += fy * 5.0
                    forces_x[j] -= fx * 5.0
                    forces_y[j] -= fy * 5.0
                    
                    # Radii shrink slightly if overlapping to allow movement
                    forces_r[i] -= 0.5 * overlap
                    forces_r[j] -= 0.5 * overlap
                else:
                    # No overlap: Attract to increase radii (if space allows)
                    # We want to grow radii if there is space.
                    # Space available = dist - min_dist
                    # Limit growth to half the gap
                    gap = dist - min_dist
                    growth = 0.1 * gap
                    forces_r[i] += growth
                    forces_r[j] += growth
                    
                    # Also, if there is a lot of gap, pull centers slightly apart?
                    # No, usually we want to fill gaps.
                    # Actually, pulling apart helps if they are stuck.
                    # But growing radii is the primary goal.
                    
        # 2. Update positions and radii
        for i in range(n):
            # Momentum update
            momentum[i, 0] = momentum[i, 0] * decay + forces_x[i]
            momentum[i, 1] = momentum[i, 1] * decay + forces_y[i]
            
            # Step
            xy[i, 0] += lr * momentum[i, 0]
            xy[i, 1] += lr * momentum[i, 1]
            
            # Radius update
            r[i] += lr * 0.5 * forces_r[i]
            
            # Constraints
            # Keep inside square
            xy[i, 0] = np.clip(xy[i, 0], 0.0, 1.0)
            xy[i, 1] = np.clip(xy[i, 1], 0.0, 1.0)
            
            # Keep radius positive and not exceeding wall distance
            max_r = np.minimum(np.minimum(xy[i, 0], 1.0 - xy[i, 0]),
                               np.minimum(xy[i, 1], 1.0 - xy[i, 1]))
            r[i] = np.clip(r[i], 0.01, max_r)
            
            # If radius hit wall limit, push center back?
            # The wall forces handle this.
            
        # 3. Evaluate
        s, v = _eval_packing(xy, r)
        if v and s > best_score:
            best_score = s
            best_xy = xy.copy()
            best_r = r.copy()
            
        # Decay learning rate slightly
        lr *= 0.999
        
        # Occasionally add noise to escape local minima
        if it % 100 == 0 and it > 0:
            noise = rng.normal(0, 0.01, size=(n, 2))
            xy += noise
            xy = np.clip(xy, 0.0, 1.0)
            
    return best_xy, best_r, best_score

def _best_fit_local_search(xy, r, deadline, grid_size=100):
    """
    Local search: for each circle, find the best position and radius
    given the others are fixed.
    """
    n = len(xy)
    best_total, best_valid = _eval_packing(xy, r)
    best_xy = xy.copy()
    best_r = r.copy()
    if best_valid:
        best_total = float(np.sum(best_r))
    else:
        best_total = 0.0

    gx = np.linspace(0.0, 1.0, grid_size)
    gy = np.linspace(0.0, 1.0, grid_size)
    GX, GY = np.meshgrid(gx, gy, indexing='ij')
    candidates = np.column_stack([GX.ravel(), GY.ravel()])
    M = len(candidates)

    wall_clear = np.minimum(np.minimum(candidates[:, 0], 1.0 - candidates[:, 0]),
                            np.minimum(candidates[:, 1], 1.0 - candidates[:, 1]))

    max_sweeps = 10
    for sweep in range(max_sweeps):
        if time.time() > deadline:
            break
        improved = False
        for i in range(n):
            if time.time() > deadline:
                break
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            others_xy = xy[mask]
            others_r = r[mask]

            if len(others_xy) > 0:
                dx = candidates[:, 0:1] - others_xy[:, 0][np.newaxis, :]
                dy = candidates[:, 1:2] - others_xy[:, 1][np.newaxis, :]
                dists = np.sqrt(dx * dx + dy * dy)
                circle_clear = np.min(dists - others_r[np.newaxis, :], axis=1)
                r_candidates = np.minimum(wall_clear, circle_clear)
            else:
                r_candidates = wall_clear.copy()

            idx = int(np.argmax(r_candidates))
            new_x, new_y = candidates[idx, 0], candidates[idx, 1]
            new_r = r_candidates[idx]

            old_r = r[i]
            if new_r > old_r + 1e-10:
                xy[i, 0] = new_x
                xy[i, 1] = new_y
                r[i] = new_r
                improved = True

        total, valid = _eval_packing(xy, r)
        if valid and total > best_total + 1e-10:
            best_total = total
            best_valid = True
            best_xy = xy.copy()
            best_r = r.copy()

        if not improved:
            break

    return best_xy, best_r, best_total, best_valid

def construct():
    start_time = time.time()
    # Safety margin: 10s before 150s limit
    deadline = start_time + 140.0

    best_xy = None
    best_r = None
    best_total = 0.0
    best_valid = False

    # Phase 1: Multiple Vector Field Optimizations with different initializations
    # This is the core structural change: using force-directed dynamics instead of topology/Newton.
    
    # 1a. Grid init
    xy_grid = _grid_init(5)
    r_grid = xy_grid[:, 2].copy()
    xy_grid = xy_grid[:, :2].copy()
    
    vf_xy, vf_r, vf_total = _vector_field_optimize(
        xy_grid, r_grid, deadline - 20, grid_res=100, max_iter=2000, seed=1
    )
    if vf_total > best_total:
        best_total = vf_total
        best_valid = True
        best_xy = vf_xy.copy()
        best_r = vf_r.copy()

    # 1b. Random init 1
    xy_rand1, r_rand1 = _random_init(42)
    vf_xy2, vf_r2, vf_total2 = _vector_field_optimize(
        xy_rand1, r_rand1, deadline - 20, grid_res=100, max_iter=2000, seed=2
    )
    if vf_total2 > best_total:
        best_total = vf_total2
        best_valid = True
        best_xy = vf_xy2.copy()
        best_r = vf_r2.copy()

    # 1c. Random init 2
    xy_rand2, r_rand2 = _random_init(123)
    vf_xy3, vf_r3, vf_total3 = _vector_field_optimize(
        xy_rand2, r_rand2, deadline - 20, grid_res=100, max_iter=2000, seed=3
    )
    if vf_total3 > best_total:
        best_total = vf_total3
        best_valid = True
        best_xy = vf_xy3.copy()
        best_r = vf_r3.copy()

    # Phase 2: Local Search Refinement
    if best_valid and best_xy is not None and time.time() < deadline - 10:
        ls_xy, ls_r, ls_total, ls_valid = _best_fit_local_search(
            best_xy, best_r, deadline - 5, grid_size=80
        )
        if ls_valid and ls_total > best_total:
            best_total = ls_total
            best_xy = ls_xy
            best_r = ls_r

    # Phase 3: Another Vector Field pass from the local search result?
    # Sometimes VF can escape local minima that LS found.
    if best_valid and best_xy is not None and time.time() < deadline - 5:
        vf_xy4, vf_r4, vf_total4 = _vector_field_optimize(
            best_xy, best_r, deadline - 2, grid_res=100, max_iter=1000, seed=4
        )
        if vf_total4 > best_total:
            best_total = vf_total4
            best_xy = vf_xy4
            best_r = vf_r4
            
            # Final polish
            if time.time() < deadline - 1:
                 ls_xy2, ls_r2, ls_total2, ls_valid2 = _best_fit_local_search(
                    best_xy, best_r, deadline, grid_size=100
                )
                 if ls_valid2 and ls_total2 > best_total:
                    best_total = ls_total2
                    best_xy = ls_xy2
                    best_r = ls_r2

    # Fallback
    if best_xy is None or not best_valid:
        side = int(np.ceil(np.sqrt(N)))
        xy = _grid_init(side)
        r = xy[:, 2].copy()
        xy = xy[:, :2].copy()
        total, valid = _eval_packing(xy, r)
        if valid:
            best_total = total
            best_valid = True
            best_xy = xy
            best_r = r
        else:
            result = np.zeros((N, 3))
            for i in range(N):
                gx = i % side
                gy = i // side
                result[i, 0] = (gx + 0.5) / side
                result[i, 1] = (gy + 0.5) / side
                result[i, 2] = 0.4 / side
            return result

    # Final validation and clipping
    n = len(best_xy)
    result = np.zeros((n, 3))
    for i in range(n):
        x, y, ri = best_xy[i, 0], best_xy[i, 1], best_r[i]
        ri = min(ri, x, 1.0 - x, y, 1.0 - y)
        ri = max(ri, 0.001)
        result[i, 2] = ri
        result[i, 0] = np.clip(x, ri, 1.0 - ri)
        result[i, 1] = np.clip(y, ri, 1.0 - ri)

    final_xy = result[:, :2]
    final_r = result[:, 2]
    total, valid = _eval_packing(final_xy, final_r)
    if not valid:
        side = int(np.ceil(np.sqrt(N)))
        result = np.zeros((N, 3))
        for i in range(N):
            gx = i % side
            gy = i // side
            result[i, 0] = (gx + 0.5) / side
            result[i, 1] = (gy + 0.5) / side
            result[i, 2] = 0.4 / side

    return result
