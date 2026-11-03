"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.spatial.distance import cdist

_T0 = time.time()
_DEADLINE = _T0 + 140.0


def _time_left():
    return _DEADLINE - time.time()


def _compute_radius(pos):
    """Compute packing radius: min(min_pair_dist/2, min_boundary_dist)."""
    n = len(pos)
    if n == 0:
        return 0.0
    dist_boundary = np.min(0.5 - np.abs(pos), axis=1)
    min_boundary = float(np.min(dist_boundary))
    if min_boundary <= 0.0:
        return 0.0
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    min_pair = math.sqrt(float(np.min(dist_sq)))
    return min(min_pair * 0.5, min_boundary)


def _generate_hex_candidates(spacing):
    """Dense hexagonal lattice of candidate centres inside [-0.5,0.5]^2."""
    dy = spacing * math.sqrt(3.0) / 2.0
    max_coord = 0.5
    points = []
    row = 0
    y = -max_coord
    while y <= max_coord:
        x_offset = 0.0 if (row % 2 == 0) else spacing / 2.0
        x = -max_coord + x_offset
        while x <= max_coord:
            if -max_coord <= x <= max_coord and -max_coord <= y <= max_coord:
                points.append((x, y))
            x += spacing
        y += dy
        row += 1
    return np.array(points, dtype=float) if points else np.zeros((0, 2))


def _greedy_select(candidates, n_select):
    """
    Greedy best-fit: repeatedly pick the candidate that maximises the minimum
    distance to (already-selected points, boundary).
    """
    m = len(candidates)
    if m == 0:
        return np.zeros((0, 2)), 0.0
    if m <= n_select:
        sel = candidates.copy()
        return sel, _compute_radius(sel)

    boundary = np.min(0.5 - np.abs(candidates), axis=1)
    score = boundary.copy()
    sel_idx = []

    for _ in range(n_select):
        if _time_left() < 5.0:
            break
        best = int(np.argmax(score))
        sel_idx.append(best)
        p = candidates[best]
        diff = candidates - p
        d = np.sqrt(np.sum(diff * diff, axis=1))
        d[best] = np.inf
        score = np.minimum(score, d)

    sel = candidates[sel_idx]
    return sel, _compute_radius(sel)


def _initial_hex_grid(n):
    """Create a centered hexagonal grid initialization for n points."""
    # Approximate hexagonal packing
    cols = int(math.ceil(math.sqrt(n * 2.0 / math.sqrt(3.0))))
    rows = int(math.ceil(n / cols))
    
    spacing_x = 1.0 / (cols + 0.5)
    spacing_y = spacing_x * math.sqrt(3.0) / 2.0
    
    points = []
    for row in range(rows):
        y = -0.5 + (row + 0.5) * spacing_y
        offset = 0.0 if row % 2 == 0 else spacing_x / 2.0
        for col in range(cols):
            x = -0.5 + (col + 0.5) * spacing_x + offset
            if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                points.append([x, y])
            if len(points) >= n:
                break
        if len(points) >= n:
            break
    
    if len(points) < n:
        # Fill remaining with random points
        rng = np.random.RandomState(42)
        while len(points) < n:
            x = rng.uniform(-0.45, 0.45)
            y = rng.uniform(-0.45, 0.45)
            points.append([x, y])
    
    return np.array(points[:n], dtype=float)


def _gradient_refine(pos, max_iter=200, lr=0.001):
    """
    Gradient-based refinement using soft-min approximation of the radius.
    Uses shared cdist for efficient gradient computation.
    """
    pos = np.array(pos, dtype=float, copy=True)
    n = len(pos)
    best_r = _compute_radius(pos)
    
    # Temperature for soft-min
    temp = 0.01
    best_pos = pos.copy()
    
    for iteration in range(max_iter):
        if _time_left() < 10.0:
            break
        
        # Compute pairwise distances
        dists = cdist(pos, pos)
        np.fill_diagonal(dists, np.inf)
        
        # Boundary distances
        boundary = np.min(0.5 - np.abs(pos), axis=1)
        
        # Current radius
        min_pair = np.min(dists)
        min_boundary = np.min(boundary)
        cur_r = min(min_pair / 2.0, min_boundary)
        
        if cur_r > best_r + 1e-12:
            best_r = cur_r
            best_pos = pos.copy()
        
        # Compute gradients using soft-min
        # Gradient of -min_pair_dist w.r.t. positions
        # For the closest pair, push them apart
        
        # Find closest pairs
        min_pair_idx = np.unravel_index(np.argmin(dists), dists.shape)
        i, j = min_pair_idx
        
        # Gradient contribution from closest pair
        grad = np.zeros_like(pos)
        diff_ij = pos[i] - pos[j]
        dist_ij = np.linalg.norm(diff_ij)
        if dist_ij > 1e-10:
            dir_ij = diff_ij / dist_ij
            # Push i away from j and j away from i
            grad[i] += dir_ij
            grad[j] -= dir_ij
        
        # Boundary gradients
        for k in range(n):
            if boundary[k] < min_pair / 2.0 + 1e-6:
                # Point is near boundary, push inward
                for dim in range(2):
                    if pos[k, dim] > 0:
                        grad[k, dim] -= 1.0
                    else:
                        grad[k, dim] += 1.0
        
        # Normalize gradient
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-10:
            grad /= grad_norm
        
        # Update positions
        pos_new = pos + lr * grad
        pos_new = np.clip(pos_new, -0.4999, 0.4999)
        
        # Check if improvement
        new_r = _compute_radius(pos_new)
        if new_r > cur_r - 1e-12:
            pos = pos_new
            # Decrease learning rate slowly
            lr *= 0.99
        else:
            # Backtrack
            lr *= 0.8
            if lr < 1e-8:
                break
    
    return best_pos, best_r


def _multi_start_refine(pos_list, total_time_budget):
    """Run gradient refinement on multiple starts, return best result."""
    best_pos = None
    best_r = 0.0
    time_per_start = total_time_budget / max(len(pos_list), 1)
    
    for idx, pos in enumerate(pos_list):
        if _time_left() < 5.0:
            break
        
        start_time = time.time()
        max_iter = int(1000 * min(1.0, time_per_start / 5.0))
        
        try:
            refined, r = _gradient_refine(pos, max_iter=max_iter, lr=0.002)
            if r > best_r:
                best_r = r
                best_pos = refined
        except Exception:
            continue
    
    return best_pos, best_r


def construct():
    best_pos = None
    best_r = 0.0
    
    # Strategy 1: Greedy selection from hexagonal lattice
    spacings = [0.04, 0.03, 0.025]
    for sp in spacings:
        if _time_left() < 15.0:
            break
        try:
            candidates = _generate_hex_candidates(sp)
            if len(candidates) < N:
                continue
            sel, r = _greedy_select(candidates, N)
            if len(sel) >= N and r > best_r:
                best_r = r
                best_pos = sel.copy()
        except Exception:
            continue
    
    # Strategy 2: Centered hexagonal grid initialization
    if _time_left() > 20.0:
        try:
            hex_init = _initial_hex_grid(N)
            r = _compute_radius(hex_init)
            if r > best_r:
                best_r = r
                best_pos = hex_init.copy()
        except Exception:
            pass
    
    # Strategy 3: Random initialization
    if _time_left() > 15.0:
        try:
            rng = np.random.RandomState(123)
            rand_init = rng.uniform(-0.45, 0.45, size=(N, 2))
            r = _compute_radius(rand_init)
            if r > best_r:
                best_r = r
                best_pos = rand_init.copy()
        except Exception:
            pass
    
    # Fallback: coarse uniform grid
    if best_pos is None or best_r < 0.02:
        m = int(math.ceil(math.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])[:N]
        r = _compute_radius(grid)
        if r > best_r:
            best_r = r
            best_pos = grid.copy()
    
    if best_pos is None:
        m = int(math.ceil(math.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        best_pos = np.array([(x, y) for y in ticks for x in ticks])[:N]
        best_r = _compute_radius(best_pos)
    
    # Multi-start gradient refinement
    if _time_left() > 15.0:
        # Create multiple starting points for refinement
        starts = [best_pos.copy()]
        
        # Add perturbed versions
        rng = np.random.RandomState(456)
        for _ in range(3):
            perturbed = best_pos + rng.randn(N, 2) * 0.01
            perturbed = np.clip(perturbed, -0.4999, 0.4999)
            starts.append(perturbed)
        
        # Also try the hex initialization if it was different
        if len(starts) < 5:
            try:
                hex_init2 = _initial_hex_grid(N)
                if not np.allclose(hex_init2, best_pos, atol=1e-6):
                    starts.append(hex_init2)
            except Exception:
                pass
        
        time_budget = min(_time_left() - 10.0, 100.0)
        if time_budget > 10.0:
            refined_pos, refined_r = _multi_start_refine(starts, time_budget)
            if refined_r > best_r:
                best_r = refined_r
                best_pos = refined_pos
    
    # Final safety: guarantee validity
    best_pos = np.asarray(best_pos, dtype=float)
    if best_pos.shape != (N, 2):
        m = int(math.ceil(math.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])[:N]
        best_pos = grid
    best_pos = np.clip(best_pos, -0.49999, 0.49999)
    if not np.all(np.isfinite(best_pos)):
        m = int(math.ceil(math.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        best_pos = np.array([(x, y) for y in ticks for x in ticks])[:N]
    
    return best_pos
# EVOLVE-BLOCK-END
