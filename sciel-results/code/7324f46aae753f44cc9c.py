import time
from math import gcd

def construct():
    start_time = time.time()
    DEADLINE = 110  # Safety margin
    
    N = 511
    # Map index i to value a = 511 - i
    # i=0 -> a=511, w=1/1
    # i=1 -> a=510, w=1/2
    # ...
    # i=510 -> a=1, w=1/511
    
    values = [511 - i for i in range(N)]
    weights = [1.0 / (512 - a) for a in values]
    
    # Precompute neighbors (conflict graph)
    # neighbors[i] is a bitset of j such that gcd(values[i], values[j]) > 1
    neighbors = [0] * N
    for i in range(N):
        bi = values[i]
        mask = 0
        for j in range(N):
            if i == j:
                continue
            if gcd(bi, values[j]) > 1:
                mask |= (1 << j)
        neighbors[i] = mask
        
    # Greedy Initialization
    chosen_indices = []
    excluded = 0
    for i in range(N):
        if not (excluded & (1 << i)):
            chosen_indices.append(i)
            excluded |= neighbors[i]
            excluded |= (1 << i)
            
    current_set = [values[i] for i in chosen_indices]
    current_score = sum(weights[i] for i in chosen_indices)
    best_set = list(current_set)
    best_score = current_score
    
    # Local Search
    def get_score(indices):
        return sum(weights[i] for i in indices)
        
    def is_valid_swap(remove_idx, add_idx, chosen_indices_set):
        # Check if add_idx is coprime to all in chosen_indices except remove_idx
        a_add = values[add_idx]
        for idx in chosen_indices_set:
            if idx == remove_idx:
                continue
            if gcd(a_add, values[idx]) > 1:
                return False
        return True
        
    chosen_indices_set = set(chosen_indices)
    
    improved = True
    while improved:
        improved = False
        if time.time() - start_time > DEADLINE:
            break
            
        # 1-swap
        for i_idx in list(chosen_indices_set):
            # Try to replace i_idx with some j_idx not in set
            # j_idx should have higher weight? Not necessarily, but if w(j) < w(i), 
            # it might allow other swaps later? 
            # For 1-swap, we only care if w(j) > w(i) AND valid.
            # If w(j) < w(i), score decreases, so we don't do it for 1-swap.
            best_j = -1
            best_w = weights[i_idx]
            for j_idx in range(N):
                if j_idx in chosen_indices_set:
                    continue
                if weights[j_idx] <= best_w:
                    continue
                if is_valid_swap(i_idx, j_idx, chosen_indices_set):
                    best_j = j_idx
                    best_w = weights[j_idx]
                    # Since we iterate j in 0..N-1 (highest weight first), 
                    # the first valid one is the best?
                    # Yes, weights are decreasing with index.
                    break
                    
            if best_j != -1:
                chosen_indices_set.remove(i_idx)
                chosen_indices_set.add(best_j)
                current_score += weights[best_j] - weights[i_idx]
                improved = True
                if current_score > best_score:
                    best_score = current_score
                    best_set = [values[i] for i in chosen_indices_set]
                break # Restart loop
                
    # Branch and Bound
    # We need to track the best set found in B&B
    # best_score is already updated by local search
    
    # Precompute suffix sums of weights for bound?
    # bound = curr_w + sum(w[j] for j in candidates)
    # We can compute this on the fly.
    
    def bnb(i, curr_w, excluded_mask):
        nonlocal best_score, best_set
        
        if time.time() - start_time > DEADLINE:
            return
            
        if i == N:
            if curr_w > best_score:
                # Reconstruct set? 
                # This is inefficient. We should track the set or reconstruct at the end.
                # For now, just update score. 
                # To return the set, we need to store the path.
                # Let's store the best indices.
                pass
            return
            
        # Bound calculation
        # Candidates are j >= i such that not excluded
        # mask_from_i = ((1 << (N - i)) - 1) << i
        # candidates = (~excluded_mask) & mask_from_i
        # sum_w = sum(weights[j] for j in set_bits(candidates))
        
        # Optimization: 
        # If curr_w + sum(weights[i:]) <= best_score, prune.
        # This is a loose bound.
        # Let's use a tighter bound:
        # candidates = [j for j in range(i, N) if not (excluded_mask & (1 << j))]
        # bound = curr_w + sum(weights[j] for j in candidates)
        
        # To speed up, we can precompute prefix sums? No, candidates depend on excluded.
        # We can iterate bits.
        
        mask_from_i = ((1 << (N - i)) - 1) << i
        candidates = (~excluded_mask) & mask_from_i
        
        if candidates == 0:
            return
            
        # Calculate bound
        # We can use a heuristic: if curr_w + sum of top K weights < best_score, prune.
        # But we don't know K.
        # Let's just sum all candidates.
        bound = curr_w
        temp_cand = candidates
        while temp_cand:
            l = temp_cand & -temp_cand
            idx = l.bit_length() - 1
            bound += weights[idx]
            temp_cand ^= l
            
        if bound <= best_score:
            return
            
        # Branch on the first available candidate (highest weight)
        # Find lowest set bit in candidates (which corresponds to highest weight)
        l = candidates & -candidates
        u = l.bit_length() - 1
        
        # Option 1: Exclude u
        new_excluded = excluded_mask | (1 << u)
        bnb(u + 1, curr_w, new_excluded)
        
        # Option 2: Include u
        # Check if u is not excluded (it isn't, by construction)
        new_excluded = excluded_mask | neighbors[u] | (1 << u)
        bnb(u + 1, curr_w + weights[u], new_excluded)
        
    # The B&B above doesn't track the set. 
    # Given the complexity and time limit, and that Local Search is often close to optimal,
    # and B&B might time out, let's rely on Local Search primarily, 
    # but run B&B only if we have lots of time and N is small enough?
    # Actually, the knowledge says B&B found 1.041. 
    # Let's try to make B&B track the set.
    
    # To track the set in B&B, we can pass a list of chosen indices.
    # But copying lists is slow.
    # We can use backtracking with a global array `path` and `path_len`.
    
    path = [0] * N
    path_len = 0
    
    def bnb_with_path(i, curr_w, excluded_mask):
        nonlocal best_score, best_set, path_len
        
        if time.time() - start_time > DEADLINE:
            return
            
        if i == N:
            if curr_w > best_score:
                best_score = curr_w
                best_set = [values[path[k]] for k in range(path_len)]
            return
            
        mask_from_i = ((1 << (N - i)) - 1) << i
        candidates = (~excluded_mask) & mask_from_i
        
        if candidates == 0:
            return
            
        # Bound
        bound = curr_w
        temp_cand = candidates
        while temp_cand:
            l = temp_cand & -temp_cand
            idx = l.bit_length() - 1
            bound += weights[idx]
            temp_cand ^= l
            
        if bound <= best_score:
            return
            
        # Branch on u
        l = candidates & -candidates
        u = l.bit_length() - 1
        
        # Option 1: Exclude u
        new_excluded = excluded_mask | (1 << u)
        bnb_with_path(u + 1, curr_w, new_excluded)
        
        # Option 2: Include u
        path[path_len] = u
        path_len += 1
        new_excluded = excluded_mask | neighbors[u] | (1 << u)
        bnb_with_path(u + 1, curr_w + weights[u], new_excluded)
        path_len -= 1
        
    # Run B&B
    # Initialize path with greedy solution? No, B&B starts from scratch.
    # But we can use the local search result as a lower bound.
    # We already have best_score and best_set from local search.
    
    # Check time
    if time.time() - start_time < DEADLINE:
        bnb_with_path(0, 0.0, 0)
        
    return best_set

# EVOLVE-BLOCK-END
