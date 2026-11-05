import time
import random
import math
import numpy as np

N = 64
TOTAL = N * N

# Precompute squared distances for all pairs of grid points
xs = np.repeat(np.arange(N), N)
ys = np.tile(np.arange(N), N)

dx = xs[:, None] - xs[None, :]
dy = ys[:, None] - ys[None, :]
dist2 = (dx * dx + dy * dy).astype(np.int32)

def construct():
    start_time = time.time()
    deadline = start_time + 290  # Safety margin

    best_set = []
    best_size = 0
    
    # Strategy: "Prune-and-Fill"
    # 1. Initialize a dense random set.
    # 2. Iteratively remove points involved in the most isosceles triangles (Batch Deletion).
    # 3. Once valid, use Greedy Additive to add more points.
    # 4. Use Local Search (Swap/Move) to refine.
    
    # We will run this multiple times with different random seeds for initialization.
    
    for attempt in range(200):
        if time.time() > deadline - 5:
            break
            
        # 1. Initialize a dense set
        # Start with a random subset of size K_init
        # K_init should be large enough to have many conflicts, but small enough to prune fast.
        # Let's try K_init = 150
        K_init = 150
        if K_init > TOTAL:
            K_init = TOTAL
            
        S_indices = np.random.choice(TOTAL, K_init, replace=False)
        
        # 2. Prune Phase
        # We maintain S_indices as a list for easy removal, but use numpy for distance checks.
        # To make removal fast, we can use a boolean mask or a set.
        # Since K is small, a list is fine, but we need to recompute distances for the remaining set.
        
        current_S = list(S_indices)
        
        # Limit pruning iterations to avoid getting stuck
        max_prune_iters = 500
        prune_iter = 0
        
        while prune_iter < max_prune_iters:
            if time.time() > deadline - 10:
                break
                
            k = len(current_S)
            if k < 2:
                break
                
            # Check if valid
            # Compute pairwise distances for current_S
            S_arr = np.array(current_S)
            D_sub = dist2[S_arr][:, S_arr]
            
            # Find isosceles triangles
            # For each point i, check if any distance D_sub[i, j] == D_sub[i, k] for j != k
            # We can do this by checking for duplicates in each row of D_sub.
            
            is_valid = True
            conflict_counts = np.zeros(k, dtype=int)
            
            for i in range(k):
                # Distances from i to others
                d_i = D_sub[i, :]
                # We only care about j > i to avoid double counting? 
                # No, for apex i, we look at all j.
                # If d_i has duplicates, i is an apex.
                # Also, i could be a base vertex. 
                # If d(i, j) == d(i, k), apex is i.
                # If d(i, j) == d(j, k), apex is j. This is covered when we check row j.
                # So checking rows for duplicates covers all isosceles triangles where the apex is the row index.
                # Every isosceles triangle has exactly one apex (unless equilateral, which is also isosceles).
                # Wait, equilateral triangles have 3 apices? No, in an equilateral triangle, all 3 pairs of equal sides exist.
                # So checking for duplicates in rows catches all isosceles triangles.
                
                # Find duplicates in d_i
                # We can use np.unique
                # But we want to count how many triangles i is part of as an apex.
                # If d_i has values [1, 1, 2, 2, 3], then i is apex for (1,1) and (2,2).
                # Number of pairs with same distance: sum over each unique value v of count(v) choose 2.
                
                # Efficient way:
                # Sort d_i? Or use a dictionary?
                # For small k, np.unique is fine.
                vals, counts = np.unique(d_i, return_counts=True)
                
                # Calculate number of isosceles triangles with apex i
                # For each distance value v with count c, there are c*(c-1)/2 triangles.
                tri_count = np.sum(counts * (counts - 1) // 2)
                
                if tri_count > 0:
                    is_valid = False
                    conflict_counts[i] += tri_count
            
            if is_valid:
                break
                
            # Remove the point with the highest conflict count
            # Tie-breaker: random
            max_conflict = np.max(conflict_counts)
            candidates = np.where(conflict_counts == max_conflict)[0]
            if len(candidates) == 0:
                # Should not happen if is_valid is False
                break
                
            remove_idx = candidates[np.random.choice(len(candidates))]
            current_S.pop(remove_idx)
            prune_iter += 1
            
        # 3. Greedy Additive Phase
        # Now current_S is valid. Try to add more points.
        # We use the fast check from the "current program" logic but vectorized.
        
        # To speed up, we precompute D_used matrix? 
        # Or just use the list-based check which is O(K) per candidate.
        # Since K is now smaller (maybe 50-100), O(K) check is fast.
        
        S_set = set(current_S)
        
        # Try to add points in random order
        remaining = [i for i in range(TOTAL) if i not in S_set]
        np.random.shuffle(remaining)
        
        for p_idx in remaining:
            if time.time() > deadline - 5:
                break
                
            # Check if p_idx can be added to current_S
            if not current_S:
                current_S.append(p_idx)
                S_set.add(p_idx)
                continue
                
            S_arr = np.array(current_S)
            dists = dist2[p_idx, S_arr]
            
            # Check 1: Distances from p to all points in S must be distinct
            if np.unique(dists).size != len(current_S):
                continue
                
            # Check 2: For each s in S, d(p, s) must not be a distance already used by s to another point in S.
            # We need to know if d(p, s) is in the set of distances from s to other points in S.
            # We can precompute a set of used distances for each s in S?
            # That's O(K^2) to build.
            # Alternatively, check: Is there a t in S such that d(s, t) == d(p, s)?
            # If yes, then (p, s, t) is isosceles with apex s.
            # So for each s_i in S, check if dists[i] is in dist2[s_i, S_arr].
            
            conflict = False
            for i, s_i in enumerate(current_S):
                d_ps = dists[i]
                # Check if d_ps is in distances from s_i to all other points in S
                # dist2[s_i, S_arr] gives distances from s_i to all points in S (including itself, which is 0)
                # We need to check if d_ps exists in dist2[s_i, S_arr] for any index j != i.
                # Actually, if d_ps == dist2[s_i, s_j] for some j != i, then conflict.
                # Note: dist2[s_i, s_i] = 0. d_ps > 0. So we can just check the whole row.
                # But we must ensure j != i.
                # If d_ps == 0, it's a duplicate point, which is not allowed in a set.
                
                # Vectorized check for this s_i:
                # d_s_to_S = dist2[s_i, S_arr]
                # We want to know if d_ps is in d_s_to_S, and if the index where it occurs is not i.
                # Actually, if d_ps == d(s_i, s_j), and j != i, conflict.
                # If j == i, d(s_i, s_i) = 0. d_ps is distance to p, so d_ps > 0. So j != i is guaranteed if d_ps > 0.
                
                # So we just check if d_ps is in dist2[s_i, S_arr].
                # But wait, dist2[s_i, S_arr] includes the distance to p? No, p is not in S yet.
                # So dist2[s_i, S_arr] contains distances to existing points.
                
                # If d_ps is in dist2[s_i, S_arr], then there exists j in S such that d(s_i, s_j) == d(p, s_i).
                # This forms an isosceles triangle (p, s_i, s_j) with apex s_i.
                
                # We can vectorize this inner loop?
                # For all i, check if dists[i] is in dist2[current_S[i], S_arr].
                
                # Let's do it in a loop for clarity and speed (K is small)
                if np.any(dist2[s_i, S_arr] == d_ps):
                    conflict = True
                    break
                    
            if not conflict:
                current_S.append(p_idx)
                S_set.add(p_idx)
                
        # 4. Local Search / Swap Phase
        # Try to swap points in current_S with points not in current_S to allow more additions?
        # Or just try to add points again in a different order?
        # Let's try a simple "restart greedy" on the remaining points.
        
        # Update best
        if len(current_S) > best_size:
            best_size = len(current_S)
            best_set = [(idx // N, idx % N) for idx in current_S]
            
        # Optional: Intensive refinement with multiple passes
        # We can try to add points in sorted order of "degree of freedom"
        
    return best_set
