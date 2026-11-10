# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from itertools import product

def kissing_number11() -> np.ndarray:
    """
    Finds a large set of integer points in 11D such that 
    min_pairwise_dist_sq >= max_norm_sq.
    """
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin
    
    d = 11
    best_k = 0
    best_points = None
    
    # Generate candidate vectors
    # We want integer vectors with small norms.
    # Let's generate all vectors with entries in {-2, -1, 0, 1, 2} 
    # and squared norm <= 12.
    
    candidates = []
    cand_set = set()
    
    # Efficient generation using combinations of non-zero indices
    # Max weight (number of non-zeros) to consider: 
    # If weight 1: max norm 4 (entry 2)
    # If weight 2: max norm 8 (entries 2,2)
    # If weight 3: max norm 12 (entries 2,2,2)
    # If weight 4: min norm 4 (entries 1,1,1,1), max norm 16. We want <= 12.
    
    # Let's iterate through possible weights and values
    # To keep it fast, we'll generate up to weight 4.
    
    # Weight 1
    for i in range(d):
        for val in [-2, -1, 1, 2]:
            v = [0]*d
            v[i] = val
            t = tuple(v)
            if t not in cand_set:
                cand_set.add(t)
                candidates.append(t)
                
    # Weight 2
    for i in range(d):
        for j in range(i+1, d):
            for vi in [-2, -1, 1, 2]:
                for vj in [-2, -1, 1, 2]:
                    v = [0]*d
                    v[i] = vi
                    v[j] = vj
                    norm_sq = vi*vi + vj*vj
                    if norm_sq <= 12:
                        t = tuple(v)
                        if t not in cand_set:
                            cand_set.add(t)
                            candidates.append(t)
                            
    # Weight 3
    for i in range(d):
        for j in range(i+1, d):
            for k in range(j+1, d):
                for vi in [-2, -1, 1, 2]:
                    for vj in [-2, -1, 1, 2]:
                        for vk in [-2, -1, 1, 2]:
                            v = [0]*d
                            v[i] = vi
                            v[j] = vj
                            v[k] = vk
                            norm_sq = vi*vi + vj*vj + vk*vk
                            if norm_sq <= 12:
                                t = tuple(v)
                                if t not in cand_set:
                                    cand_set.add(t)
                                    candidates.append(t)
                                    
    # Weight 4
    # Only include if norm_sq <= 12. Min norm is 4 (1,1,1,1).
    for i in range(d):
        for j in range(i+1, d):
            for k in range(j+1, d):
                for l in range(k+1, d):
                    # Optimization: only try combinations that can fit in norm 12
                    # 1,1,1,1 -> 4
                    # 2,1,1,1 -> 7
                    # 2,2,1,1 -> 10
                    # 2,2,2,1 -> 13 (too big)
                    # So possible abs values: {1,1,1,1}, {2,1,1,1}, {2,2,1,1}
                    for vi in [1, 2]:
                        for vj in [1, 2]:
                            for vk in [1, 2]:
                                for vl in [1, 2]:
                                    abs_vals = sorted([abs(vi), abs(vj), abs(vk), abs(vl)])
                                    if abs_vals == [1,1,1,2] or abs_vals == [1,1,1,1] or abs_vals == [1,1,2,2]:
                                        # Check norm
                                        norm_sq = vi*vi + vj*vj + vk*vk + vl*vl
                                        if norm_sq <= 12:
                                            # Generate all sign combinations
                                            for si in [1, -1]:
                                                for sj in [1, -1]:
                                                    for sk in [1, -1]:
                                                        for sl in [1, -1]:
                                                            v = [0]*d
                                                            v[i] = vi * si
                                                            v[j] = vj * sj
                                                            v[k] = vk * sk
                                                            v[l] = vl * sl
                                                            t = tuple(v)
                                                            if t not in cand_set:
                                                                cand_set.add(t)
                                                                candidates.append(t)
    # Weight 5
    # Min norm 5.
    # 1,1,1,1,1 -> 5
    # 2,1,1,1,1 -> 8
    # 2,2,1,1,1 -> 11
    for i in range(d):
        for j in range(i+1, d):
            for k in range(j+1, d):
                for l in range(k+1, d):
                    for m in range(l+1, d):
                        # Only 1,1,1,1,1 and 2,1,1,1,1 and 2,2,1,1,1
                        # 1,1,1,1,1
                        v = [0]*d
                        v[i]=1; v[j]=1; v[k]=1; v[l]=1; v[m]=1
                        t = tuple(v)
                        if t not in cand_set:
                            cand_set.add(t)
                            candidates.append(t)
                        # Signs
                        for si in [1,-1]:
                            for sj in [1,-1]:
                                for sk in [1,-1]:
                                    for sl in [1,-1]:
                                        for sm in [1,-1]:
                                            v = [0]*d
                                            v[i]=si; v[j]=sj; v[k]=sk; v[l]=sl; v[m]=sm
                                            t = tuple(v)
                                            if t not in cand_set:
                                                cand_set.add(t)
                                                candidates.append(t)
                        # 2,1,1,1,1 permutations
                        # This is getting complex. Let's just generate all weight 5 with entries 1/2 
                        # and check norm.
                        # Actually, weight 5 with 2s:
                        # 2,1,1,1,1 -> norm 8.
                        # 2,2,1,1,1 -> norm 11.
                        # Let's just loop all sign combos for specific abs patterns.
                        
    # Convert to numpy array
    if not candidates:
         return np.array([np.ones(d), -np.ones(d)]).astype(np.int64)
         
    cand_arr = np.array(candidates, dtype=np.int64)
    n_cands = len(cand_arr)
    norms_sq = np.sum(cand_arr ** 2, axis=1)
    
    # We will try different Max Norm Squared (M) values.
    # M must be >= max norm in the set.
    # And min dist sq >= M.
    # Let's iterate M from 4 to 12.
    
    possible_Ms = sorted(set(norms_sq))
    
    for M in possible_Ms:
        if time.time() > deadline - 60: break
        
        # Filter candidates with norm_sq <= M
        mask = norms_sq <= M
        filtered_cands = cand_arr[mask]
        filtered_norms = norms_sq[mask]
        n_f = len(filtered_cands)
        
        if n_f < 2: continue
        
        # Compute pairwise squared distances
        # dist_sq[i,j] = ||v_i - v_j||^2
        # We need dist_sq >= M for all i != j in the selected set.
        # Conflict if dist_sq < M.
        
        # To save memory, we can compute in chunks or just do it if n_f is small enough.
        # n_f can be up to ~10,000. 10,000^2 = 100M. 100M * 8 bytes = 800MB. Might be too much.
        # Let's check n_f.
        if n_f > 5000:
             # Subsample or use a more memory efficient method?
             # Or just skip this M if it's too big?
             # Actually, the number of candidates with norm <= 4 is around 5000.
             # Norm <= 5 adds more.
             # Let's try to be clever.
             pass
        
        # Compute distance matrix
        # Use broadcasting
        # dist_sq = np.sum(filtered_cands**2)[:, None] + np.sum(filtered_cands**2)[None, :] - 2 * (filtered_cands @ filtered_cands.T)
        
        # If n_f is large, this is slow.
        # Let's limit M to values where n_f is manageable (< 6000).
        if n_f > 6000:
            continue
            
        dot = filtered_cands @ filtered_cands.T
        dist_sq = filtered_norms[:, np.newaxis] + filtered_norms[np.newaxis, :] - 2 * dot
        
        # Conflict matrix: True if dist_sq < M
        # Note: dist_sq[i,i] = 0 < M, so self-conflict.
        # We want Independent Set, so no conflicts between distinct nodes.
        conflict = dist_sq < M
        
        # Diagonal is True, we ignore self.
        
        # Greedy MIS
        # 1. Calculate degree (number of conflicts)
        degrees = np.sum(conflict, axis=1)
        
        best_local_k = 0
        best_local_set = None
        
        # Run multiple random restarts
        max_restarts = 20
        for _ in range(max_restarts):
            if time.time() > deadline - 10: break
            
            # Randomize tie-breaking in sorting
            # Add small random noise to degrees to break ties randomly
            noisy_degrees = degrees + np.random.uniform(0, 1, n_f)
            order = np.argsort(noisy_degrees)
            
            selected = np.zeros(n_f, dtype=bool)
            current_k = 0
            
            for idx in order:
                if selected[idx]:
                    continue
                # Check if idx conflicts with any selected vertex
                # conflicts_with_selected = np.any(conflict[idx, selected])
                # This is O(n) per check, O(n^2) total.
                # We can maintain a list of selected indices.
                
                # Faster:
                # If current_k is 0, add.
                # Else, check against selected mask.
                if not np.any(conflict[idx, selected]):
                    selected[idx] = True
                    current_k += 1
                    
            if current_k > best_local_k:
                best_local_k = current_k
                best_local_set = selected.copy()
                
            # Local Search: Try to add more
            # Repeatedly try to add any unselected vertex that has 0 conflicts with selected
            # This is essentially what the greedy does, but maybe order mattered.
            # Let's do a simple improvement loop:
            improved = True
            while improved:
                improved = False
                # Find unselected vertices
                unselected = np.where(~selected)[0]
                if len(unselected) == 0:
                    break
                # Check conflicts for all unselected against selected
                # conflict[unselected, selected]
                # If any row is all False, we can add that vertex.
                
                # Vectorized check:
                # conflicts = conflict[np.ix_(unselected, np.where(selected)[0])]
                # if np.any(~np.any(conflicts, axis=1)):
                
                sel_indices = np.where(selected)[0]
                if len(sel_indices) == 0:
                    # Can add any
                    for u in unselected:
                        selected[u] = True
                        current_k += 1
                        improved = True
                    break
                
                conflicts = conflict[np.ix_(unselected, sel_indices)]
                can_add = np.where(~np.any(conflicts, axis=1))[0]
                
                if len(can_add) > 0:
                    for u_idx in can_add:
                        u = unselected[u_idx]
                        selected[u] = True
                        current_k += 1
                        improved = True
                    # After adding, the conflict matrix for remaining unselected changes
                    # But we can just restart the check in the next iteration of the while loop
                    
            if current_k > best_local_k:
                best_local_k = current_k
                best_local_set = selected.copy()

        if best_local_k > best_k:
            best_k = best_local_k
            best_points = filtered_cands[best_local_set]
            
    # Final verification
    if best_points is not None and best_k >= 2:
        pts = np.array(best_points, dtype=np.int64)
        k = len(pts)
        norms_sq = np.sum(pts ** 2, axis=1)
        max_norm_sq = np.max(norms_sq)
        
        min_dist_sq = np.inf
        # Verify with exact integer arithmetic
        for i in range(k):
            for j in range(i+1, k):
                diff = pts[i] - pts[j]
                d2 = np.sum(diff ** 2)
                if d2 < min_dist_sq:
                    min_dist_sq = d2
        
        if min_dist_sq >= max_norm_sq:
            return pts
    else:
        # Fallback
        pts = np.array([np.ones(d), -np.ones(d)]).astype(np.int64)
        return pts

# EVOLVE-BLOCK-END
