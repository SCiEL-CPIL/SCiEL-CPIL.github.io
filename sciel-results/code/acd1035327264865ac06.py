"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.spatial import cKDTree

# Global deadline guard: evaluator kills at 150s; keep a safety margin.
_T0 = time.time()
_DEADLINE = _T0 + 140.0  # 10s safety margin before the 150s kill


def _time_left():
    return _DEADLINE - time.time()


def _compute_radius(pos):
    """Vectorized packing radius: min(min_pair/2, min_boundary)."""
    n = len(pos)
    if n == 0:
        return 0.0
    # Boundary constraint: distance from each centre to the nearest side.
    dist_boundary = np.min(0.5 - np.abs(pos), axis=1)
    min_boundary = float(np.min(dist_boundary))
    if min_boundary <= 0.0:
        return 0.0
    # Pairwise distances via squared distances.
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
        if _time_left() < 8.0:
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


def _force_directed_spread(pos, iterations=500, k=0.002):
    """
    Force-directed relaxation: points repel each other (inverse-square) and
    are attracted toward the square center to maintain density.
    Uses scipy's cKDTree for O(n log n) neighbor queries.
    """
    pos = np.array(pos, dtype=float, copy=True)
    n = len(pos)
    if n < 2:
        return pos
    
    # Target spacing estimate
    target_r = _compute_radius(pos)
    if target_r <= 0:
        target_r = 0.036
    
    # Repulsion strength scales with current radius
    rep_strength = target_r * 2.0
    
    for it in range(iterations):
        if _time_left() < 8.0:
            break
        
        # Build KD-tree for efficient neighbor queries
        tree = cKDTree(pos)
        
        # Find all pairs within 2.5 * target_r (potential conflicts)
        pairs = tree.query_pairs(r=target_r * 2.5)
        
        forces = np.zeros_like(pos)
        
        for i, j in pairs:
            d = pos[i] - pos[j]
            dist = np.linalg.norm(d)
            if dist < 1e-12:
                # Coincident points: push apart randomly
                d = np.random.randn(2) * 1e-8
                dist = np.linalg.norm(d)
            d_norm = d / dist
            # Inverse-square repulsion
            f = rep_strength / (dist * dist)
            forces[i] += d_norm * f
            forces[j] -= d_norm * f
        
        # Gentle attraction toward center to prevent drift
        center_pull = -pos * 0.001
        
        # Update positions with damping
        pos += (forces + center_pull) * k
        
        # Clip to square interior
        pos = np.clip(pos, -0.4999, 0.4999)
    
    return pos


def _projected_gradient_phase(pos, lr=0.001, momentum=0.9, max_iter=3000):
    """
    Projected gradient descent with momentum to maximize sum of squared pairwise distances.
    This is a smooth surrogate that correlates well with packing radius.
    """
    pos = np.array(pos, dtype=float, copy=True)
    n = len(pos)
    if n < 2:
        return pos
    
    vel = np.zeros_like(pos)
    
    for it in range(max_iter):
        if _time_left() < 8.0:
            break
        
        # Compute gradient of sum of squared pairwise distances
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        np.fill_diagonal(dist_sq, np.inf)
        
        # Gradient of sum_{i<j} |p_i - p_j|^2 w.r.t. p_i is 2 * sum_{j != i} (p_i - p_j)
        # = 2 * (n * p_i - sum_j p_j)
        # More efficiently: grad_i = 2 * (n * p_i - sum_all_p)
        sum_p = np.sum(pos, axis=0)
        grad = 2.0 * (n * pos - sum_p[np.newaxis, :])
        
        # Add boundary penalty: push points away from boundaries
        # Penalty: -sum(0.5 - |p_ik|)^2 for each coordinate
        # Gradient: 2 * (0.5 - |p_ik|) * sign(p_ik)
        bx = 0.5 - np.abs(pos[:, 0])
        by = 0.5 - np.abs(pos[:, 1])
        # Only add penalty if point is close to boundary (within 2*r)
        r_est = _compute_radius(pos)
        margin = 2.0 * r_est
        mask_x = bx < margin
        mask_y = by < margin
        grad[:, 0] += 2.0 * bx * np.sign(pos[:, 0]) * mask_x
        grad[:, 1] += 2.0 * by * np.sign(pos[:, 1]) * mask_y
        
        # Update velocity with momentum
        vel = momentum * vel + lr * grad
        pos += vel
        
        # Project back to square
        pos = np.clip(pos, -0.4999, 0.4999)
        
        # Decay learning rate
        lr *= 0.9995
    
    return pos


def _soft_min_optimize(pos, beta=200.0, lr=0.0005, max_iter=2000):
    """
    Gradient-based soft-min optimization. Minimizes a smooth approximation
    of the negative packing radius using the soft-min of constraints.
    """
    pos = np.array(pos, dtype=float, copy=True)
    n = len(pos)
    
    def soft_min_neg_r(p, b):
        """
        Compute -r_soft where r_soft approximates min(min_pair/2, min_boundary).
        We minimize this to maximize r.
        """
        # Compute boundary distances
        bx = 0.5 - np.abs(p[:, 0])
        by = 0.5 - np.abs(p[:, 1])
        
        # Compute pairwise distances / 2
        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        np.fill_diagonal(dist_sq, np.inf)
        dist_full = np.sqrt(dist_sq)
        
        # Objective: sum of exp(-c_i/beta) for all constraints
        # Boundary: bx_i and by_i for each i
        # Pair: dist_ij/2 for each i<j
        
        exp_bx = np.exp(-bx / b)
        exp_by = np.exp(-by / b)
        
        # Pair terms - only upper triangle
        iu = np.triu_indices(n, k=1)
        pair_vals = dist_full[iu] / 2.0
        exp_p = np.exp(-pair_vals / b)
        
        obj = np.sum(exp_bx) + np.sum(exp_by) + np.sum(exp_p)
        
        # Gradient
        grad = np.zeros_like(p)
        
        # Boundary gradient
        grad[:, 0] += exp_bx * np.sign(p[:, 0]) / b
        grad[:, 1] += exp_by * np.sign(p[:, 1]) / b
        
        # Pair gradient
        weight = np.exp(-dist_full / (2 * b)) / b
        factor = weight / (2 * dist_full)
        factor = np.where(dist_full > 1e-12, factor, 0.0)
        factor = -factor
        
        diff_full = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        grad_pair = np.sum(factor[:, :, np.newaxis] * diff_full, axis=1)
        grad += grad_pair
        
        return obj, grad
    
    # Gradient descent with adaptive learning rate
    for it in range(max_iter):
        if _time_left() < 8.0:
            break
        
        obj, grad = soft_min_neg_r(pos, beta)
        
        # Normalize gradient to prevent explosion
        gnorm = np.linalg.norm(grad)
        if gnorm > 1e-12:
            grad = grad / gnorm * min(gnorm, 0.01)
        
        pos -= lr * grad
        pos = np.clip(pos, -0.4999, 0.4999)
        
        # Adaptive beta: increase over time for sharper approximation
        beta = min(beta * 1.01, 500.0)
    
    return pos


def _polish_pass(pos, max_iter=500):
    """
    High-precision polish: iteratively find the closest pair and boundary,
    then move points to increase the minimum distance.
    """
    pos = np.array(pos, dtype=float, copy=True)
    n = len(pos)
    
    for it in range(max_iter):
        if _time_left() < 8.0:
            break
        
        r = _compute_radius(pos)
        if r <= 0:
            break
        
        # Find the minimum pair and boundary
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        np.fill_diagonal(dist_sq, np.inf)
        min_idx = np.unravel_index(np.argmin(dist_sq), dist_sq.shape)
        min_pair_dist = math.sqrt(dist_sq[min_idx])
        
        dist_boundary = np.min(0.5 - np.abs(pos), axis=1)
        min_bnd_idx = int(np.argmin(dist_boundary))
        min_bnd_dist = dist_boundary[min_bnd_idx]
        
        # Determine which constraint is active
        if min_pair_dist / 2.0 <= min_bnd_dist:
            # Pair constraint is active
            i, j = min_idx
            d = pos[i] - pos[j]
            dist = np.linalg.norm(d)
            if dist < 1e-12:
                d = np.random.randn(2) * 1e-8
                dist = np.linalg.norm(d)
            d_norm = d / dist
            # Move i and j apart by a small amount
            step = 0.001 * (dist - 2 * r)
            if step > 0:
                pos[i] += d_norm * step / 2
                pos[j] -= d_norm * step / 2
        else:
            # Boundary constraint is active
            p = pos[min_bnd_idx]
            # Move toward center
            step = 0.001 * (min_bnd_dist - r)
            if step > 0:
                pos[min_bnd_idx] = p + (-p / np.linalg.norm(p)) * step
        
        pos = np.clip(pos, -0.4999, 0.4999)
        
        # Check if we made progress
        new_r = _compute_radius(pos)
        if new_r <= r + 1e-10:
            # No progress, try a small random perturbation
            pos += np.random.randn(n, 2) * 0.0001
            pos = np.clip(pos, -0.4999, 0.4999)
    
    return pos


def _local_refine(pos):
    """
    Combined local refinement: force-directed spread, projected gradient,
    soft-min optimization, and polish pass.
    """
    pos = np.array(pos, dtype=float, copy=True)
    
    # Phase 1: Force-directed spread to resolve obvious overlaps
    if _time_left() > 20.0:
        pos = _force_directed_spread(pos, iterations=300, k=0.001)
    
    # Phase 2: Projected gradient descent with momentum
    if _time_left() > 15.0:
        pos = _projected_gradient_phase(pos, lr=0.001, momentum=0.9, max_iter=2000)
    
    # Phase 3: Soft-min gradient optimization for precise refinement
    if _time_left() > 10.0:
        pos = _soft_min_optimize(pos, beta=100.0, lr=0.0003, max_iter=1000)
    
    # Phase 4: Polish pass for high-precision refinement
    if _time_left() > 8.0:
        pos = _polish_pass(pos, max_iter=300)
    
    return pos


def construct():
    best_pos = None
    best_r = 0.0

    # 1) Greedy selection from dense hexagonal lattices at multiple spacings.
    spacings = [0.035, 0.03, 0.025, 0.02]
    for sp in spacings:
        if _time_left() < 25.0:
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

    # Fallback: coarse uniform grid if greedy produced nothing usable.
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

    # 2) Local refinement: force-directed + projected gradient + soft-min + polish.
    if _time_left() > 25.0:
        refined = _local_refine(best_pos)
        r = _compute_radius(refined)
        if r > best_r:
            best_r = r
            best_pos = refined

    # 3) Try a second refinement pass with different parameters if time allows.
    if _time_left() > 20.0:
        # Perturb slightly and refine again
        rng = np.random.RandomState(42)
        perturbed = best_pos + rng.randn(N, 2) * 0.002
        perturbed = np.clip(perturbed, -0.4999, 0.4999)
        refined2 = _local_refine(perturbed)
        r2 = _compute_radius(refined2)
        if r2 > best_r:
            best_r = r2
            best_pos = refined2

    # 4) Final polish if time allows
    if _time_left() > 10.0:
        polished = _polish_pass(best_pos, max_iter=200)
        r3 = _compute_radius(polished)
        if r3 > best_r:
            best_r = r3
            best_pos = polished

    # Final safety: guarantee validity.
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
