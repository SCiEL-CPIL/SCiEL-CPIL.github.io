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


def _simulated_annealing(pos, time_limit=100.0):
    """
    Simulated Annealing with bottleneck-targeted perturbations.
    Uses cKDTree for efficient nearest-neighbor search.
    """
    pos = np.array(pos, dtype=float, copy=True)
    n = len(pos)
    start_time = time.time()
    
    # Initial radius
    current_r = _compute_radius(pos)
    best_r = current_r
    best_pos = pos.copy()
    
    # Temperature parameters
    # Start with a temperature that allows significant movement
    T = 0.01 * max(current_r, 0.03)
    T_min = 1e-6
    cooling_rate = 0.99
    
    # Pre-allocate noise array for speed
    noise = np.random.randn(n, 2)
    
    while _time_left() > 5.0 and (time.time() - start_time) < time_limit:
        if T < T_min:
            break
            
        # 1. Identify Bottleneck
        # Build KDTree for current positions
        tree = cKDTree(pos)
        
        # Find closest pair
        # query_pairs returns indices of all pairs within distance r
        # We need the global minimum distance.
        # A faster way is to query each point's nearest neighbor and take the min.
        # However, query_pairs with a small r might miss if r is too small.
        # Let's use a safe r for finding *some* close pairs, then refine.
        # Actually, for N=200, full distance matrix is fast enough for one check, 
        # but KDTree is better for repeated checks if we can leverage it.
        
        # Let's compute the full distance matrix once per iteration for simplicity and robustness 
        # since N=200 is small (40k ops), which is ~microseconds.
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        np.fill_diagonal(dist_sq, np.inf)
        min_idx_flat = np.argmin(dist_sq)
        i, j = np.unravel_index(min_idx_flat, (n, n))
        min_pair_dist = math.sqrt(dist_sq[i, j])
        
        # Boundary
        dist_boundary = np.min(0.5 - np.abs(pos), axis=1)
        min_bnd_idx = int(np.argmin(dist_boundary))
        min_bnd_dist = dist_boundary[min_bnd_idx]
        
        # Determine active constraint
        if min_pair_dist / 2.0 < min_bnd_dist:
            # Pair is bottleneck
            constraint_val = min_pair_dist / 2.0
            is_pair = True
        else:
            # Boundary is bottleneck
            constraint_val = min_bnd_dist
            is_pair = False
            
        # 2. Perturb
        # We will perturb the bottleneck points
        # Magnitude of kick is proportional to T
        kick_mag = T * 0.5
        
        if is_pair:
            # Move i and j apart
            vec = pos[i] - pos[j]
            dist = np.linalg.norm(vec)
            if dist < 1e-10:
                vec = np.random.randn(2)
                dist = np.linalg.norm(vec)
            vec = vec / dist
            
            # Add random noise to all points? No, just bottleneck points + small global noise
            # To escape local minima, add small noise to everyone, but kick the pair
            pos += np.random.randn(n, 2) * (T * 0.1) # Global small noise
            
            pos[i] += vec * kick_mag
            pos[j] -= vec * kick_mag
            
        else:
            # Move point min_bnd_idx away from boundary
            p = pos[min_bnd_idx]
            # Determine direction to center
            # If x is close to 0.5, push -x. If -0.5, push +x.
            # Simpler: push towards origin
            vec = -p
            dist = np.linalg.norm(vec)
            if dist < 1e-10:
                vec = np.random.randn(2)
                dist = np.linalg.norm(vec)
            vec = vec / dist
            
            pos += np.random.randn(n, 2) * (T * 0.1)
            pos[min_bnd_idx] += vec * kick_mag
            
        # Project to square
        pos = np.clip(pos, -0.49999, 0.49999)
        
        # 3. Evaluate
        new_r = _compute_radius(pos)
        
        # 4. Accept/Reject
        if new_r >= current_r:
            current_r = new_r
            if new_r > best_r:
                best_r = new_r
                best_pos = pos.copy()
        else:
            # Accept with probability exp((new_r - current_r) / T)
            delta = new_r - current_r
            # Avoid overflow/underflow issues
            if delta > -10 * T: 
                prob = math.exp(delta / T)
            else:
                prob = 0.0
            if np.random.rand() < prob:
                current_r = new_r
            else:
                # Revert
                # We need to revert the changes. 
                # Since we modified pos in place, we should have kept a copy or reverted manually.
                # For efficiency, we can just keep the old pos if rejected?
                # But we already overwrote pos.
                # Let's restructure: compute new_pos, then decide.
                # Actually, reverting is hard if we don't store old_pos.
                # Let's store old_pos for the bottleneck points only? 
                # Or just accept the cost of copying pos for N=200? 
                # Copying 200x2 floats is very fast.
                # Let's refactor to use a candidate array.
                # Wait, the loop is tight. Let's assume we keep the current pos if rejected?
                # No, that's wrong. We must revert.
                # I will implement a simple revert by keeping a snapshot of the modified points.
                # But for now, to keep the code simple and fast, I will just 
                # re-run the perturbation on the *previous* state? 
                # No, that's complex.
                # Let's just copy pos at the start of the iteration.
                # 200 * 8 bytes = 1.6KB copy. Negligible.
                pass # This branch is currently broken logic-wise without revert.
                # I will fix the main loop below to handle revert properly.
                
        # Cool down
        T *= cooling_rate
        
    return best_pos, best_r

def _sa_local_search_fixed(pos, time_limit=100.0):
    """
    Proper Simulated Annealing with revert capability.
    """
    pos = np.array(pos, dtype=float, copy=True)
    n = len(pos)
    start_time = time.time()
    
    current_r = _compute_radius(pos)
    best_r = current_r
    best_pos = pos.copy()
    
    T = 0.01 * max(current_r, 0.03)
    T_min = 1e-6
    cooling_rate = 0.99
    
    while _time_left() > 5.0 and (time.time() - start_time) < time_limit:
        if T < T_min:
            break
            
        # 1. Identify Bottleneck
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        np.fill_diagonal(dist_sq, np.inf)
        min_idx_flat = np.argmin(dist_sq)
        i, j = np.unravel_index(min_idx_flat, (n, n))
        min_pair_dist = math.sqrt(dist_sq[i, j])
        
        dist_boundary = np.min(0.5 - np.abs(pos), axis=1)
        min_bnd_idx = int(np.argmin(dist_boundary))
        min_bnd_dist = dist_boundary[min_bnd_idx]
        
        is_pair = (min_pair_dist / 2.0 < min_bnd_dist)
        
        # 2. Create Candidate
        candidate = pos.copy()
        kick_mag = T * 0.5
        
        if is_pair:
            vec = candidate[i] - candidate[j]
            dist = np.linalg.norm(vec)
            if dist < 1e-10:
                vec = np.random.randn(2)
                dist = np.linalg.norm(vec)
            vec = vec / dist
            
            candidate += np.random.randn(n, 2) * (T * 0.1)
            candidate[i] += vec * kick_mag
            candidate[j] -= vec * kick_mag
            
        else:
            p = candidate[min_bnd_idx]
            vec = -p
            dist = np.linalg.norm(vec)
            if dist < 1e-10:
                vec = np.random.randn(2)
                dist = np.linalg.norm(vec)
            vec = vec / dist
            
            candidate += np.random.randn(n, 2) * (T * 0.1)
            candidate[min_bnd_idx] += vec * kick_mag
            
        candidate = np.clip(candidate, -0.49999, 0.49999)
        
        # 3. Evaluate Candidate
        new_r = _compute_radius(candidate)
        
        # 4. Accept/Reject
        if new_r >= current_r:
            pos = candidate
            current_r = new_r
            if new_r > best_r:
                best_r = new_r
                best_pos = pos.copy()
        else:
            delta = new_r - current_r
            if delta > -10 * T:
                prob = math.exp(delta / T)
            else:
                prob = 0.0
            if np.random.rand() < prob:
                pos = candidate
                current_r = new_r
        
        T *= cooling_rate
        
    return best_pos, best_r


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
                norm_p = np.linalg.norm(p)
                if norm_p > 1e-10:
                    pos[min_bnd_idx] = p + (-p / norm_p) * step
                else:
                    pos[min_bnd_idx] += np.random.randn(2) * step

        pos = np.clip(pos, -0.49999, 0.49999)
        
        # Check if we made progress
        new_r = _compute_radius(pos)
        if new_r <= r + 1e-10:
            # No progress, try a small random perturbation
            pos += np.random.randn(n, 2) * 0.0001
            pos = np.clip(pos, -0.49999, 0.49999)
    
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

    # 2) Simulated Annealing refinement
    if _time_left() > 30.0:
        sa_pos, sa_r = _sa_local_search_fixed(best_pos, time_limit=_time_left() - 10.0)
        if sa_r > best_r:
            best_r = sa_r
            best_pos = sa_pos

    # 3) Polish pass for high-precision refinement
    if _time_left() > 10.0:
        polished = _polish_pass(best_pos, max_iter=300)
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
