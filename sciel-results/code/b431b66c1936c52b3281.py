"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _score(c):
    """Compute the largest common radius for centres c (shape (N,3))."""
    if c.shape != (N, 3):
        return -1.0
    if not np.all(np.isfinite(c)):
        return -1.0
    # distance to walls
    wall = np.min(np.min(0.5 - np.abs(c), axis=1), axis=0)
    # pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_dist = np.min(dist)
    return min(wall, min_dist / 2.0)

def _init_grid():
    """Simple cubic grid as starting point."""
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N].copy()

def _lattice_search(deadline):
    """
    Search on a dense integer lattice.
    1. Generate a dense grid of candidate points.
    2. Select N points greedily to maximize minimum distance (approximation).
    3. Refine via local search: try moving a point to a nearby lattice site if it improves min distance.
    """
    # Lattice resolution. 
    # N=100. Volume=1. Volume per sphere ~ 0.01. 
    # r ~ 0.11. 
    # Lattice spacing should be smaller than 2r to allow resolution.
    # Let's try 20x20x20 = 8000 points.
    # Or 25x25x25 = 15625 points.
    # 20x20x20 is 8000 points. N=100.
    # Let's use 20 divisions per axis.
    M = 20
    # Coordinates: -0.5 to 0.5
    # We want points inside.
    # If we use indices 0..M-1, coords are (i+0.5)/M - 0.5
    # Spacing = 1/M = 0.05
    # r ~ 0.11, so 2r ~ 0.22. 4 lattice steps.
    # This should be fine.
    
    coords = (np.arange(M) + 0.5) / M - 0.5
    grid_points = np.array([(x, y, z) for z in coords for y in coords for x in coords])
    
    # Initial selection: Greedy
    # To maximize min distance, we can start with a coarser grid or random.
    # Let's start with a 5x5x4 grid (100 points) mapped to the fine lattice.
    # Actually, let's just pick 100 random points from the lattice to start,
    # or better, a regular grid subset.
    
    # Regular grid subset:
    # 5x5x4 = 100 points.
    m_x, m_y, m_z = 5, 5, 4
    indices = []
    for z in range(m_z):
        for y in range(m_y):
            for x in range(m_x):
                # Map to fine lattice indices
                # We want to space them out evenly.
                # Fine lattice has M=20.
                # Coarse step: M / m_dim
                ix = int(x * M / m_x)
                iy = int(y * M / m_y)
                iz = int(z * M / m_z)
                indices.append(iz * M*M + iy * M + ix)
                
    # Ensure we have exactly N unique indices
    indices = np.array(indices[:N])
    
    # Get coordinates
    c = grid_points[indices].copy()
    
    # Local search on lattice
    # Objective: maximize min( wall_dist, min_pair_dist / 2 )
    # We will maximize min_pair_dist primarily, but wall constraint is hard.
    # If a point is too close to wall, it's bad.
    
    # Precompute neighbors in lattice?
    # For each point, check moves to 6 neighbors (up, down, left, right, front, back).
    # If the neighbor is occupied, skip.
    # If the neighbor is outside, skip.
    # If the move improves the "bottleneck" distance, accept.
    
    # To speed up, we maintain a set of occupied indices.
    occupied = set(indices.tolist())
    
    # Calculate current score
    current_score = _score(c)
    best_score = current_score
    best_c = c.copy()
    
    # We want to maximize the minimum clearance.
    # Let's iterate.
    
    # To make it faster, we can just check if min_pair_dist increases.
    # But wall distance matters too.
    
    # Let's do a simple hill climbing on the lattice.
    # Since N=100 and M=20, total points 8000.
    # Checking 100 points * 6 moves * O(N) cost = 6000 * 100 = 600k ops per iteration?
    # No, checking if a move is valid requires checking distance to all other 99 points.
    # 100 points * 6 moves * 99 checks * 3 dims = ~180k float ops.
    # This is very fast in NumPy if vectorized per move?
    # Or just loop in Python? 180k ops is ~0.1ms in Python? Maybe 1ms.
    # 1000 iterations = 1s.
    # We have 120s. So we can do many iterations.
    
    max_iters = 2000
    no_improve_count = 0
    
    for it in range(max_iters):
        if time.time() > deadline - 10: # Reserve 10s for polish
            break
            
        improved = False
        # Shuffle order to avoid bias
        order = np.random.permutation(N)
        
        for i in order:
            if time.time() > deadline - 10:
                break
                
            # Try 6 neighbors
            # Get current index
            idx = indices[i]
            x, y, z = np.unravel_index(idx, (M, M, M))
            
            # Neighbors
            neighbors = []
            if x > 0: neighbors.append((x-1, y, z))
            if x < M-1: neighbors.append((x+1, y, z))
            if y > 0: neighbors.append((x, y-1, z))
            if y < M-1: neighbors.append((x, y+1, z))
            if z > 0: neighbors.append((x, y, z-1))
            if z < M-1: neighbors.append((x, y, z+1))
            
            for (nx, ny, nz) in neighbors:
                n_idx = nz * M*M + ny * M + nx
                if n_idx in occupied:
                    continue
                
                # Tentative move
                old_idx = idx
                indices[i] = n_idx
                occupied.remove(old_idx)
                occupied.add(n_idx)
                
                # Update c
                old_c = c[i].copy()
                c[i] = grid_points[n_idx]
                
                # Check score
                # We only need to check distances involving i
                # And wall distance for i
                # But global min might be elsewhere.
                # If we only increase min distance for i, global min might stay same or increase.
                # If we decrease min distance for i, global min might decrease.
                
                # Let's compute the new min distance involving i
                # dist to walls
                w = np.min(0.5 - np.abs(c[i]))
                
                # dist to others
                diffs = c - c[i]
                diffs[i] = np.inf
                dists = np.sqrt(np.sum(diffs**2, axis=1))
                min_d_to_others = np.min(dists)
                
                new_local_min = min(w, min_d_to_others / 2.0)
                
                # If new_local_min > current global min, it's a definite improvement?
                # Not necessarily, but if it's > current global min, it helps.
                # Actually, we want to maximize the global min.
                # If new_local_min > best_score, we should accept?
                # No, we should compare to the *current* global min.
                
                # Let's just compute full score if it looks promising?
                # Or compute full score every time?
                # _score is O(N^2). 100^2 = 10k.
                # 100 points * 6 moves = 600 moves per iter.
                # 600 * 10k = 6M ops per iter.
                # 2000 iters = 12B ops. Too slow in Python.
                
                # Optimization:
                # Maintain global_min.
                # If new_local_min > global_min, accept.
                # If new_local_min < global_min, reject.
                # If equal, it's a tie, maybe reject to avoid cycling.
                
                # But global_min might be determined by a pair (j,k) not involving i.
                # If we move i, and new_local_min > global_min, then the new global min is at least global_min (actually, it might be higher if the old bottleneck was i).
                # If the old bottleneck was NOT i, then moving i doesn't change the global min (unless we break a constraint, which we don't).
                # Wait, if old bottleneck was (j,k) with dist D.
                # We move i. New dists for i are > D.
                # Dist(j,k) is still D.
                # So global min is still D.
                # So score doesn't improve.
                
                # So we only improve if i was involved in the bottleneck.
                
                # Strategy:
                # 1. Identify current bottleneck pair(s) or wall.
                # 2. Only try to move spheres involved in the bottleneck.
                
                # This is complex.
                # Alternative: Just do random moves and accept if _score improves.
                # But _score is expensive.
                
                # Let's use a faster approximate check.
                # If new_local_min > current_global_min + epsilon, accept.
                # Else revert.
                
                # We need to track current_global_min.
                # Let's recompute _score occasionally or maintain it.
                # For now, let's just check if new_local_min > best_score.
                # If yes, accept and update best_score.
                
                if new_local_min > best_score + 1e-9:
                    best_score = new_local_min
                    best_c = c.copy()
                    improved = True
                    # Note: we don't update global_min for all pairs, but this is a heuristic.
                    # Actually, if we accept, the global min *is* at least new_local_min?
                    # No, other pairs might be smaller.
                    # But if new_local_min > best_score (which was the global min),
                    # then the new global min is at least min(new_local_min, old_other_min).
                    # Since new_local_min > old_global_min, and old_other_min >= old_global_min,
                    # the new global min is >= old_global_min.
                    # It might be higher if the bottleneck was i.
                    
                    # To be safe and accurate, let's just compute full score if we accept?
                    # Or just trust the local check for "improvement potential".
                    
                    # Actually, if we accept a move where new_local_min > best_score,
                    # it is guaranteed that the new global min >= best_score.
                    # It might be strictly greater.
                    
                    # Let's just accept and continue.
                    
                else:
                    # Revert
                    indices[i] = old_idx
                    occupied.remove(n_idx)
                    occupied.add(old_idx)
                    c[i] = old_c
        
        if not improved:
            no_improve_count += 1
            if no_improve_count > 50:
                break
        else:
            no_improve_count = 0

    return best_c

def _polish(c, deadline):
    """
    Final continuous relaxation using L-BFGS-B on a smooth objective.
    """
    # Use soft-min objective
    # f(x) = -softmin(dists, walls)
    # We want to maximize min dist, so minimize -min dist.
    # Softmin: -log(sum(exp(-beta * d_i))) / beta
    # As beta -> inf, this approaches -min(d_i).
    
    beta = 50.0
    
    def objective(x):
        c = x.reshape(-1, 3)
        # Walls
        walls = 0.5 - np.abs(c)
        wall_dists = np.min(walls, axis=1)
        
        # Pairs
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        pair_dists = dists / 2.0
        
        # Combine
        # We want min of all these
        # Softmin over all values
        # Flatten
        vals = np.concatenate([wall_dists, pair_dists.ravel()])
        
        # Softmin
        # To avoid overflow, subtract max
        m = np.max(vals)
        # -log(sum(exp(-beta*(v - m))))/beta + m?
        # Actually, softmin(v) = -1/beta * log(sum(exp(-beta*v)))
        # = -1/beta * log(sum(exp(-beta*(v - m)))) - m
        # Let's just use:
        # -1/beta * log(sum(exp(-beta * (v - m)))) - m
        
        # Wait, if we want to maximize min(v), we minimize -min(v).
        # -min(v) approx -softmin(v).
        # softmin(v) = -1/beta * log(sum(exp(-beta*v)))
        
        # Numerical stability:
        # exp(-beta * v) can underflow if v is large.
        # v are distances ~ 0.1. beta=50. 50*0.1=5. exp(-5) is fine.
        
        # But there are many terms.
        # Sum of exps.
        
        # Let's compute -softmin directly
        # sum_exp = np.sum(np.exp(-beta * vals))
        # return -1.0 / beta * np.log(sum_exp)
        
        # Gradient
        # d/dx (-1/beta log(sum(exp(-beta v_i))))
        # = -1/beta * 1/sum * sum(exp(-beta v_i) * (-beta) * dv_i/dx)
        # = 1/sum * sum(exp(-beta v_i) * dv_i/dx)
        # = sum( p_i * dv_i/dx ) where p_i = exp(-beta v_i) / sum(exp(-beta v_j))
        # p_i are softmax weights (inverted).
        
        # This is a weighted average of gradients.
        
        # For walls:
        # v_wall_i = min_k (0.5 - |c_ik|)
        # This is non-differentiable at min.
        # Approximate with softmin over walls for each sphere?
        # Or just use the closest wall.
        # Let's use softmin over walls for each sphere to make it smooth.
        
        # Let's redefine objective with softmins everywhere.
        
        # 1. Wall softmin per sphere
        # w_i = -1/beta * log(sum_k exp(-beta * (0.5 - |c_ik|)))
        
        # 2. Pair softmin
        # p_ij = -1/beta * log(sum_{kl != ij} ... )? No, that's global.
        # Usually we do:
        # Objective = -min( min_i w_i, min_{i<j} d_ij/2 )
        # = softmin( {w_i}, {d_ij/2} )
        
        # Let's stick to the global softmin over all constraints.
        
        # For gradient, we need dv/dx.
        # This is messy with abs and min.
        
        # Simpler: Use scipy L-BFGS-B with numerical gradient?
        # N=100, 300 vars.
        # NumPy gradient is 300 evaluations.
        # Each eval is O(N^2) = 10k.
        # 3M ops per gradient.
        # L-BFGS-B might take 100 steps.
        # 300M ops. ~1-3 seconds.
        # Acceptable.
        
        c = x.reshape(-1, 3)
        # Ensure inside
        # Clip? No, let optimizer handle bounds.
        
        walls = 0.5 - np.abs(c)
        wall_dists = np.min(walls, axis=1)
        
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        pair_dists = dists / 2.0
        
        vals = np.concatenate([wall_dists, pair_dists.ravel()])
        
        # Softmin
        # Use max for stability
        m = np.max(vals)
        # exp(-beta * (vals - m))
        exps = np.exp(-beta * (vals - m))
        sum_exps = np.sum(exps)
        softmin_val = -1.0 / beta * np.log(sum_exps) - m
        
        # We want to minimize -softmin_val
        return -softmin_val

    # Bounds
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    # Initial point
    x0 = c.flatten()
    
    try:
        res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-12})
        
        c_new = res.x.reshape(-1, 3)
        s_new = _score(c_new)
        s_old = _score(c)
        
        if s_new > s_old:
            return c_new
        else:
            return c
    except:
        return c

def construct():
    start_time = time.time()
    deadline = 135.0  # safety margin

    # 1. Lattice Search
    c_lat = _lattice_search(deadline)
    best_c = c_lat.copy()
    best_score = _score(c_lat)

    # 2. Polish
    if time.time() < deadline - 5:
        c_pol = _polish(c_lat, deadline)
        s_pol = _score(c_pol)
        if s_pol > best_score:
            best_score = s_pol
            best_c = c_pol

    # 3. Try a few random restarts on lattice if time permits
    # Maybe not necessary if lattice search is good.
    # Let's just return best_c.

    return best_c
# EVOLVE-BLOCK-END
