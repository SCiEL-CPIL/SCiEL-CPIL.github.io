import numpy as np
import itertools
import time
import sys

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin for 150s limit

    # Generate candidates
    # We focus on vectors with squared norm 4 (norm 2) as they provide the tightest packing.
    # The task requires min pairwise distance >= max norm.
    # If we use only vectors of norm 2, we need dist >= 2, i.e., dist_sq >= 4.
    # dist_sq = ||v_i - v_j||^2 = ||v_i||^2 + ||v_j||^2 - 2 v_i.v_j = 4 + 4 - 2 v_i.v_j = 8 - 2 v_i.v_j.
    # Condition: 8 - 2 v_i.v_j >= 4  =>  4 >= 2 v_i.v_j  =>  v_i.v_j <= 2.
    # So we need an independent set in the graph where edges connect vectors with dot product > 2.
    
    candidates = []
    
    # Type 1: One 2, rest 0 (squared norm = 4)
    # 11 * 2 = 22 vectors
    for i in range(D):
        for s in [1, -1]:
            v = [0] * D
            v[i] = 2 * s
            candidates.append(v)
            
    # Type 2: Four 1s, rest 0 (squared norm = 4)
    # C(11,4) * 2^4 = 330 * 16 = 5280 vectors
    for idxs in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, s in zip(idxs, signs):
                v[idx] = s
            candidates.append(v)
            
    candidates = np.array(candidates, dtype=np.int32)
    N = len(candidates)
    
    # Compute dot product matrix
    M = candidates @ candidates.T
    
    # Conflict matrix: 1 if dot > 2 (and i != j)
    Conflict = (M > 2).astype(np.uint8)
    np.fill_diagonal(Conflict, 0)
    
    # We want to find the maximum independent set (MIS).
    # Greedy with random restarts is the base.
    # To improve, we can use a "weighted" greedy where we prefer nodes with lower degree
    # (standard MIS heuristic) but also consider the structure.
    # The inspiration suggests treating weight-4 vectors (norm 2) with weight 1.
    # Since all our vectors have norm 2, they all have weight 1 in that sense.
    # However, the "inspiration 1" mentioned adding lower norm vectors.
    # Let's stick to the pure norm-2 set first as it's the densest known structure for kissing.
    # The kissing number in 11 is known to be at least 460 (from Levenshtein-Rodchenko bound? No, that's upper).
    # Actually, known lower bounds for kissing number in 11 are around 460-500?
    # Wait, the current score is 2.57. The seed was 110 (from 11*2 + something? No, seed was likely just Type 1 or a small subset).
    # If score = k/seed, and seed=110 (maybe?), then k=283?
    # Or maybe seed=100?
    # Let's assume the goal is to maximize k.
    
    best_set = []
    best_size = 0
    
    # Precompute degrees
    deg = Conflict.sum(axis=1).astype(np.int64)
    
    def run_greedy(seed):
        if time.time() > deadline:
            return []
            
        rng = np.random.default_rng(seed)
        active = np.ones(N, dtype=bool)
        current_deg = deg.copy()
        selected = []
        
        while active.any():
            if time.time() > deadline:
                break
            
            active_indices = np.where(active)[0]
            if len(active_indices) == 0:
                break
                
            active_deg = current_deg[active_indices]
            min_deg = np.min(active_deg)
            candidates_for_pick = active_indices[active_deg == min_deg]
            idx = rng.choice(candidates_for_pick)
            
            selected.append(idx)
            
            neighbors_mask = Conflict[idx].astype(bool)
            removed_mask = np.zeros(N, dtype=bool)
            removed_mask[idx] = True
            removed_mask[neighbors_mask & active] = True
            
            removed_indices = np.where(removed_mask)[0]
            
            if len(removed_indices) > 0:
                # Update degrees for remaining active nodes
                # A node j loses a neighbor for every removed node r that is connected to j.
                # So current_deg[j] -= sum(Conflict[removed_indices, j]) for j in active
                # This is equivalent to: current_deg[active] -= Conflict[removed_indices, :].sum(axis=0)[active]
                removal_counts = Conflict[removed_indices, :].sum(axis=0).astype(np.int64)
                current_deg[active] -= removal_counts[active]
                
            active[removed_mask] = False
            
        return selected

    def local_search(selected_indices):
        if time.time() > deadline:
            return selected_indices
            
        selected_set = set(selected_indices)
        improved = True
        # Limit iterations to avoid getting stuck
        max_iters = 100
        iters = 0
        
        while improved and iters < max_iters and time.time() < deadline:
            improved = False
            iters += 1
            
            # Shuffle selected nodes to try different swaps
            indices_list = list(selected_set)
            rng_ls = np.random.default_rng()
            rng_ls.shuffle(indices_list)
            
            for i in indices_list:
                if time.time() > deadline:
                    return list(selected_set)
                
                # Find nodes j not in selected_set that conflict ONLY with i
                # i.e., j is a neighbor of i, and j is NOT a neighbor of any other k in selected_set \ {i}
                
                # Neighbors of i
                neighbors_of_i = np.where(Conflict[i])[0]
                
                # Candidates j must be in neighbors_of_i and not in selected_set
                valid_js = [j for j in neighbors_of_i if j not in selected_set]
                
                if len(valid_js) < 2:
                    continue
                
                # Check which valid_js are compatible with all k in selected_set \ {i}
                compatible_js = []
                for j in valid_js:
                    # j conflicts with i by definition.
                    # Check conflicts with others
                    is_compatible = True
                    for k in selected_set:
                        if k == i:
                            continue
                        if Conflict[j, k]:
                            is_compatible = False
                            break
                    if is_compatible:
                        compatible_js.append(j)
                
                if len(compatible_js) < 2:
                    continue
                
                # Find a pair (j1, j2) in compatible_js that are mutually compatible
                found_pair = False
                cwr = compatible_js
                for p in range(len(cwr)):
                    if found_pair: break
                    j1 = cwr[p]
                    for q in range(p+1, len(cwr)):
                        j2 = cwr[q]
                        if not Conflict[j1, j2]:
                            # Swap i -> j1, j2
                            selected_set.remove(i)
                            selected_set.add(j1)
                            selected_set.add(j2)
                            improved = True
                            found_pair = True
                            break
                    if found_pair:
                        break
                
                if found_pair:
                    # Restart loop to re-evaluate
                    break
                
        return list(selected_set)

    # Run multiple greedy restarts
    num_restarts = 500
    for s in range(num_restarts):
        if time.time() > deadline:
            break
        
        res = run_greedy(s)
        if not res:
            continue
            
        # Apply local search
        res_improved = local_search(res)
        
        if len(res_improved) > best_size:
            best_size = len(res_improved)
            best_set = res_improved
            
            # If we find a very good set, we might want to do more local search on it specifically
            # But for now, keep searching for new sets
    
    if not best_set:
        # Fallback: just take the first 22 (Type 1 vectors)
        # They are mutually compatible?
        # v_i = 2 e_i, v_j = 2 e_j. dot = 0 <= 2. Yes.
        best_set = list(range(22))
        
    return candidates[best_set].astype(np.int64)
