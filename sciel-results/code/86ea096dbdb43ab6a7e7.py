import numpy as np
import time
import itertools
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    D = 11

    # Precomputation
    supports = list(itertools.combinations(range(D), 4))
    N_S = len(supports)
    
    support_masks = []
    for s in supports:
        mask = 0
        for i in s:
            mask |= (1 << i)
        support_masks.append(mask)
    
    # Conflict matrix: conflict[i, j] = True if intersection >= 3
    conflict = np.zeros((N_S, N_S), dtype=bool)
    for i in range(N_S):
        for j in range(i + 1, N_S):
            inter = support_masks[i] & support_masks[j]
            if inter.bit_count() >= 3:
                conflict[i, j] = True
                conflict[j, i] = True

    # Precompute degree (number of conflicts) for each support
    degrees = conflict.sum(axis=1)

    rng = np.random.RandomState(42)

    def greedy_min_degree(randomize=False):
        """
        Greedy packing: repeatedly select the available support with minimum
        current degree (most constrained) to maximize future packing density.
        If randomize=True, add noise to break ties randomly.
        """
        selected = np.zeros(N_S, dtype=bool)
        blocked = np.zeros(N_S, dtype=bool)
        # Current degree of each unselected node (number of selected neighbors)
        current_degree = np.zeros(N_S, dtype=np.int32)
        
        count = 0
        remaining = N_S
        
        while remaining > 0:
            if time.time() > deadline - 5.0:
                break
            
            # Find unselected, non-blocked nodes
            available_mask = ~(selected | blocked)
            avail_indices = np.where(available_mask)[0]
            
            if len(avail_indices) == 0:
                break
            
            if randomize:
                # Add random noise to break ties and explore
                noise = rng.randint(0, 2, size=len(avail_indices))
                scores = current_degree[avail_indices] + noise
            else:
                scores = current_degree[avail_indices]
            
            # Select the one with minimum score (most constrained)
            min_idx = int(np.argmin(scores))
            v = avail_indices[min_idx]
            
            selected[v] = True
            count += 1
            remaining -= 1
            blocked[v] = True  # Mark as blocked so it won't be re-selected
            
            # Update degrees: all neighbors of v that are still available get +1
            neighbors = np.where(conflict[v])[0]
            neighbor_avail = neighbors[~(selected[neighbors] | blocked[neighbors])]
            if len(neighbor_avail) > 0:
                current_degree[neighbor_avail] += 1
                # Mark them as blocked if they're now too constrained? No, just update degree.
                # Actually, we should mark them blocked if they conflict with v.
                blocked[neighbors] = True
            
        return selected, count

    def local_search_polish(x):
        """
        Try to improve the solution via swap operations.
        """
        current_x = x.copy()
        current_score = np.sum(current_x)
        best_x = current_x.copy()
        best_score = current_score
        
        max_iter = 500
        for it in range(max_iter):
            if time.time() > deadline - 5.0:
                break
            
            in_set = np.where(current_x)[0]
            if len(in_set) == 0:
                break
            
            # Try swaps: remove u, add v
            u = int(rng.choice(in_set))
            
            temp_x = current_x.copy()
            temp_x[u] = False
            in_set_no_u = np.where(temp_x)[0]
            
            if len(in_set_no_u) > 0:
                blocked_no_u = np.any(conflict[in_set_no_u], axis=0)
            else:
                blocked_no_u = np.zeros(N_S, dtype=bool)
            
            candidates = np.where(~(current_x | blocked_no_u))[0]
            if len(candidates) > 0:
                # Prefer candidates with low degree for better future packing
                cand_degrees = degrees[candidates]
                noise = rng.randint(0, 2, size=len(candidates))
                scores = cand_degrees + noise
                v = candidates[int(np.argmin(scores))]
                
                new_x = current_x.copy()
                new_x[u] = False
                new_x[v] = True
                # Score unchanged for swap, but may enable further additions
                current_x = new_x
                
                # Try to add more after swap
                in_set2 = np.where(current_x)[0]
                blocked2 = np.any(conflict[in_set2], axis=0)
                candidates2 = np.where(~(current_x | blocked2))[0]
                added = 0
                while len(candidates2) > 0:
                    # Pick min degree
                    cand2_degrees = degrees[candidates2]
                    v2 = candidates2[int(np.argmin(cand2_degrees))]
                    current_x[v2] = True
                    added += 1
                    blocked2 |= conflict[v2]
                    candidates2 = np.where(~(current_x | blocked2))[0]
                
                new_score = current_score + added
                if new_score > best_score:
                    best_score = new_score
                    best_x = current_x.copy()
                    current_score = new_score
                else:
                    # Revert if no improvement
                    current_x = best_x.copy()
        
        return best_x

    best_support_set = set()
    best_size = 0
    
    # Phase 1: Deterministic greedy (fast baseline)
    x_det, count_det = greedy_min_degree(randomize=False)
    if count_det > best_size:
        best_size = count_det
        best_support_set = set(np.where(x_det)[0].tolist())
    
    # Phase 2: Multiple randomized greedy restarts
    # Each restart is fast (O(N_S * avg_degree)), so we can do many
    num_restarts = 2000
    for r in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
        
        x_r, count_r = greedy_min_degree(randomize=True)
        
        if count_r > best_size:
            # Polish the new best
            x_polished = local_search_polish(x_r)
            size_polished = np.sum(x_polished)
            if size_polished > best_size:
                best_size = int(size_polished)
                best_support_set = set(np.where(x_polished)[0].tolist())
            else:
                best_size = count_r
                best_support_set = set(np.where(x_r)[0].tolist())
        elif count_r == best_size:
            # Still try to polish even if tied, might find better arrangement
            if r % 10 == 0:  # Do this less frequently to save time
                x_polished = local_search_polish(x_r)
                size_polished = np.sum(x_polished)
                if size_polished > best_size:
                    best_size = int(size_polished)
                    best_support_set = set(np.where(x_polished)[0].tolist())
    
    # Final polish on the best solution
    if len(best_support_set) > 0:
        x_best = np.zeros(N_S, dtype=bool)
        for idx in best_support_set:
            x_best[idx] = True
        x_final = local_search_polish(x_best)
        size_final = np.sum(x_final)
        if size_final > best_size:
            best_size = int(size_final)
            best_support_set = set(np.where(x_final)[0].tolist())
    
    # Construct vectors
    vectors = []
    
    # Type A: ±2e_i for each coordinate
    for i in range(D):
        v1 = [0] * D
        v1[i] = 2
        vectors.append(v1)
        v2 = [0] * D
        v2[i] = -2
        vectors.append(v2)
        
    # Type D: all 16 sign combinations for each selected 4-subset
    for s_idx in best_support_set:
        support = supports[s_idx]
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, sign in zip(support, signs):
                v[idx] = sign
            vectors.append(v)
            
    if not vectors:
        return np.array([], dtype=np.int8).reshape(0, D)
        
    final_vectors = np.array(vectors, dtype=np.int8)
    
    return final_vectors
