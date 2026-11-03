"""Pack 200 equal circles in a unit square via hex-grid init + greedy pair separation."""
import numpy as np
import time

N = 200
HALF = 0.5
TIME_LIMIT = 145.0  # Safety margin

def _score(c):
    """Return the common radius for a configuration (higher is better)."""
    c = np.asarray(c)
    if c.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(c)):
        return 0.0
    # inside square
    m = np.max(np.abs(c), axis=1)
    if np.any(m > HALF + 1e-12):
        return 0.0
    # pairwise min distance
    d = np.linalg.norm(c[:, None, :] - c[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    min_pair = np.min(d)
    return min(min_pair / 2.0, HALF - np.max(m))

def _hex_points(s, ox, oy):
    """Hexagonal lattice points with spacing s, offset (ox, oy)."""
    dx = s
    dy = s * np.sqrt(3.0) / 2.0
    cols = int(np.ceil(1.0 / dx)) + 3
    rows = int(np.ceil(1.0 / dy)) + 3
    xs = np.arange(-2, cols) * dx + ox
    ys = np.arange(-2, rows) * dy + oy
    # offset every other column
    pts = []
    for j, x in enumerate(xs):
        yoff = (dy / 2.0) if (j % 2 == 1) else 0.0
        for i, y in enumerate(ys):
            pts.append((x + 0.0, y + yoff))
    arr = np.array(pts)
    # keep inside square
    keep = np.all(np.abs(arr) <= HALF, axis=1)
    return arr[keep]

def _greedy_select(cands, k, r_min):
    """Greedy farthest-point selection of k points maximizing min distance."""
    n = cands.shape[0]
    if n <= k:
        return cands.copy()
    selected = []
    idx = int(np.random.randint(n))
    selected.append(idx)
    min_dist = np.full(n, np.inf)
    for _ in range(k - 1):
        last = cands[selected[-1]]
        d = np.linalg.norm(cands - last, axis=1)
        min_dist = np.minimum(min_dist, d)
        j = int(np.argmax(min_dist))
        if min_dist[j] < r_min - 1e-9:
            break
        selected.append(j)
    return cands[selected]

def _compute_dist_matrix(pos):
    """Compute full pairwise distance matrix."""
    diff = pos[:, None, :] - pos[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, np.inf)
    return dist

def _local_search(pos, time_budget=60.0):
    """
    Local search using greedy pair separation and wall avoidance.
    Maintains distance matrix for O(N) updates.
    """
    t0 = time.time()
    pos = pos.copy().astype(float)
    dist = _compute_dist_matrix(pos)
    
    # Calculate initial radius
    min_pair = np.min(dist)
    margins = HALF - np.max(np.abs(pos), axis=1)
    min_margin = np.min(margins)
    r = min(min_pair / 2.0, min_margin)
    
    best_r = r
    best_pos = pos.copy()
    
    # Precompute indices for upper triangle to speed up min finding?
    # np.argmin on full matrix is fast enough for N=200.
    
    # Step size for moves
    step = 0.01
    
    # We iterate until time runs out or no improvement
    no_improve_count = 0
    
    while time.time() - t0 < time_budget:
        # Find the limiting constraint
        min_pair_idx = np.unravel_index(np.argmin(dist), dist.shape)
        i, j = min_pair_idx
        min_pair_val = dist[i, j]
        
        min_margin_idx = np.argmin(margins)
        min_margin_val = margins[min_margin_idx]
        
        current_r = min(min_pair_val / 2.0, min_margin_val)
        
        # If we improved, reset no_improve counter
        if current_r > best_r + 1e-12:
            best_r = current_r
            best_pos = pos.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
            
        if no_improve_count > 1000:
            # Try a random perturbation to escape local optimum
            if time.time() - t0 > time_budget * 0.5:
                break
            # Perturb a few random points
            for _ in range(5):
                idx = np.random.randint(N)
                pos[idx] += np.random.uniform(-0.01, 0.01, 2)
                pos[idx] = np.clip(pos[idx], -HALF, HALF)
                # Update dist row/col
                diff_i = pos[idx] - pos
                dist[idx, :] = np.sqrt(np.sum(diff_i * diff_i, axis=1))
                dist[:, idx] = dist[idx, :]
                dist[idx, idx] = np.inf
                # Update margins
                margins[idx] = HALF - np.max(np.abs(pos[idx]))
            no_improve_count = 0
            continue

        # Determine move direction
        if min_pair_val / 2.0 < min_margin_val:
            # Pairwise distance is limiting. Move i and j apart.
            diff_ij = pos[i] - pos[j]
            norm_ij = np.linalg.norm(diff_ij)
            if norm_ij < 1e-12:
                # Coincident, random move
                pos[i] += np.random.uniform(-step, step, 2)
                pos[j] -= np.random.uniform(-step, step, 2)
            else:
                v = diff_ij / norm_ij
                # Move i by +v*step, j by -v*step
                # We need to ensure they stay inside the square.
                # And we need to check if this move reduces other distances.
                
                # Try a few step sizes
                accepted = False
                for sc in [1.0, 0.5, 0.1]:
                    s_try = step * sc
                    new_pos_i = pos[i] + v * s_try
                    new_pos_j = pos[j] - v * s_try
                    
                    # Check bounds
                    if not (np.all(np.abs(new_pos_i) <= HALF) and np.all(np.abs(new_pos_j) <= HALF)):
                        continue
                        
                    # Check if this move causes any other distance to drop below current best_r?
                    # Actually, we just want to maximize the min.
                    # Calculate new distances from i and j to all others.
                    
                    # Temporarily update pos
                    old_i = pos[i].copy()
                    old_j = pos[j].copy()
                    pos[i] = new_pos_i
                    pos[j] = new_pos_j
                    
                    # Update dist for i
                    diff_i = pos[i] - pos
                    dist_i = np.sqrt(np.sum(diff_i * diff_i, axis=1))
                    dist_i[i] = np.inf
                    # Update dist for j
                    diff_j = pos[j] - pos
                    dist_j = np.sqrt(np.sum(diff_j * diff_j, axis=1))
                    dist_j[j] = np.inf
                    
                    # The new global min is min(
                    #   min(dist_i), min(dist_j), min(dist without i,j)
                    # )
                    # dist without i,j is min of dist[k,l] where k,l != i,j
                    # This is expensive to compute exactly if we don't maintain a sorted list.
                    # Approximation: The new min is at least min(old_min_pair/2, min_margin).
                    # If min(dist_i) and min(dist_j) are > current_r, and we moved the min pair apart,
                    # the new min_pair_val will be max(old_min_pair_val, min(dist_i, dist_j))? No.
                    # The new min_pair_val is min(
                    #    min_{k != i,j} dist[k,l], 
                    #    min(dist_i), 
                    #    min(dist_j)
                    # )
                    # We don't have min_{k != i,j} easily.
                    # However, if we only move if min(dist_i) and min(dist_j) are > current_r,
                    # and we increased dist[i,j], then the new global min is likely increased or same.
                    
                    # Let's check if min(dist_i) and min(dist_j) are > current_r.
                    if min(np.min(dist_i), np.min(dist_j)) > current_r:
                        # Accept move
                        dist[i, :] = dist_i
                        dist[:, i] = dist_i
                        dist[j, :] = dist_j
                        dist[:, j] = dist_j
                        dist[i, j] = np.linalg.norm(pos[i] - pos[j])
                        dist[j, i] = dist[i, j]
                        
                        # Update margins
                        margins[i] = HALF - np.max(np.abs(pos[i]))
                        margins[j] = HALF - np.max(np.abs(pos[j]))
                        
                        accepted = True
                        break
                    else:
                        # Revert
                        pos[i] = old_i
                        pos[j] = old_j
                        
                if not accepted:
                    # Reduce step size if stuck
                    step *= 0.8
                    if step < 1e-6:
                        step = 0.01
        else:
            # Wall margin is limiting. Move particle k towards center.
            k = min_margin_idx
            # Direction towards center (0,0)
            pos_k = pos[k]
            if np.linalg.norm(pos_k) < 1e-12:
                v = np.array([0, 0])
            else:
                v = -pos_k / np.linalg.norm(pos_k)
                
            accepted = False
            for sc in [1.0, 0.5, 0.1]:
                s_try = step * sc
                new_pos_k = pos_k + v * s_try
                if not np.all(np.abs(new_pos_k) <= HALF):
                    continue
                
                old_k = pos[k].copy()
                pos[k] = new_pos_k
                
                # Update dist for k
                diff_k = pos[k] - pos
                dist_k = np.sqrt(np.sum(diff_k * diff_k, axis=1))
                dist_k[k] = np.inf
                
                if np.min(dist_k) > current_r:
                    dist[k, :] = dist_k
                    dist[:, k] = dist_k
                    margins[k] = HALF - np.max(np.abs(pos[k]))
                    accepted = True
                    break
                else:
                    pos[k] = old_k
            
            if not accepted:
                step *= 0.8
                if step < 1e-6:
                    step = 0.01

    return best_pos, best_r

def construct():
    t0 = time.time()
    DEADLINE = TIME_LIMIT - 5.0
    
    rng = np.random.default_rng(2024)
    
    best_c = None
    best_v = 0.0
    
    # Target spacing estimate
    s0 = 1.0 / np.sqrt(N * np.sqrt(3.0) / 2.0)
    
    # Phase 1: Generate diverse initial configurations
    n_trials = 20
    for trial in range(n_trials):
        if time.time() > DEADLINE - 30.0:  # Leave time for polish
            break
        
        # Vary spacing and offset
        s = s0 * (0.98 + 0.04 * rng.random())
        ox = rng.uniform(-0.5, 0.5)
        oy = rng.uniform(-0.5, 0.5)
        cands = _hex_points(s, ox, oy)
        
        if cands.shape[0] < N:
            continue
        
        r_min = 0.5 * s * 0.9
        sel = _greedy_select(cands, N, r_min)
        if sel.shape[0] < N:
            continue
        
        v = _score(sel)
        if v > best_v:
            best_v = v
            best_c = sel.copy()
            
    # Phase 2: Local Search Polish
    if best_c is not None:
        remaining = DEADLINE - time.time()
        if remaining > 10.0:
            polished, pv = _local_search(best_c, time_budget=min(remaining - 5.0, 120.0))
            if pv > best_v:
                best_c = polished
                best_v = pv
    else:
        # Fallback
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_c = grid[:N]
        best_v = _score(best_c)
        
    # Final validity clamp
    best_c = np.clip(best_c, -HALF, HALF)
    
    return best_c
