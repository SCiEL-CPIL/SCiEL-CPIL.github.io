"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

P = {'n': 12, 'm': 40}

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 60s limit
    
    n = P['n']
    m = P['m']
    num_masks = 1 << n
    
    # 1. Precompute comparability matrix
    # comp[i, j] = 1 if i and j are comparable (one strict subset of other)
    masks = np.arange(num_masks, dtype=np.int32)
    
    # Using broadcasting
    mask_i = masks[:, None]
    mask_j = masks[None, :]
    subset_ij = (mask_i | mask_j) == mask_j
    subset_ji = (mask_i | mask_j) == mask_i
    not_equal = mask_i != mask_j
    comp = ((subset_ij | subset_ji) & not_equal).astype(np.int8)
    
    best_overall_score = 0
    best_overall_indices = None
    
    # Try multiple restarts with different greedy strategies
    rng = np.random.RandomState(42)
    
    num_restarts = 8
    restarts_done = 0
    
    for restart in range(num_restarts):
        if time.time() > deadline - 20:
            break
        restarts_done += 1
        
        # 2. Greedy Initialization with variation
        selected = np.zeros(num_masks, dtype=bool)
        selected_indices = []
        
        # Different greedy strategies per restart
        strategy = restart % 3
        
        if strategy == 0:
            # Pure greedy by marginal gain
            for _ in range(m):
                if time.time() > deadline - 15:
                    break
                if len(selected_indices) == 0:
                    # Pick highest degree
                    scores = comp.sum(axis=1).astype(np.int32)
                    best_idx = int(np.argmax(scores))
                else:
                    gains = comp[:, selected_indices].sum(axis=1).astype(np.int32)
                    gains[selected] = -1
                    best_idx = int(np.argmax(gains))
                selected_indices.append(best_idx)
                selected[best_idx] = True
                
        elif strategy == 1:
            # Greedy with random tie-breaking
            scores = comp.sum(axis=1).astype(np.int32)
            for _ in range(m):
                if time.time() > deadline - 15:
                    break
                if len(selected_indices) == 0:
                    best_idx = int(np.argmax(scores))
                else:
                    gains = comp[:, selected_indices].sum(axis=1).astype(np.int32)
                    gains[selected] = -1
                    # Add small random noise to break ties
                    noise = rng.random(num_masks) * 0.5
                    gains_noisy = gains + noise
                    gains_noisy[selected] = -1
                    best_idx = int(np.argmax(gains_noisy))
                selected_indices.append(best_idx)
                selected[best_idx] = True
                
        else:
            # Random initialization followed by greedy refinement
            candidates = np.arange(num_masks)
            rng.shuffle(candidates)
            for idx in candidates[:m]:
                selected_indices.append(int(idx))
                selected[int(idx)] = True
        
        # 3. Local Search (Best Improvement Single Swap)
        current_set = set(selected_indices)
        in_set = np.zeros(num_masks, dtype=bool)
        for idx in selected_indices:
            in_set[idx] = True
            
        score_in_set = comp[:, list(current_set)].sum(axis=1).astype(np.int32)
        
        current_score = 0
        for idx in selected_indices:
            current_score += score_in_set[idx]
        current_score //= 2
        
        # Single-swap local search
        while time.time() < deadline - 10:
            # Compute masked scores once
            masked_scores = np.where(in_set, -1, score_in_set)
            max_val = int(np.max(masked_scores))
            max_idx = int(np.argmax(masked_scores))
            
            if max_val < 0:
                break
            
            # Find the element in set with minimum score_in_set
            min_score_in_set = np.inf
            min_idx_in_set = -1
            for A in selected_indices:
                if score_in_set[A] < min_score_in_set:
                    min_score_in_set = score_in_set[A]
                    min_idx_in_set = A
            
            best_delta = max_val - min_score_in_set
            
            if best_delta <= 0:
                break
            
            # Apply swap
            best_out = min_idx_in_set
            best_in = max_idx
            
            score_in_set += comp[:, best_in] - comp[:, best_out]
            
            selected_indices[selected_indices.index(best_out)] = best_in
            current_set.remove(best_out)
            current_set.add(best_in)
            in_set[best_out] = False
            in_set[best_in] = True
            
            current_score += best_delta
        
        # 4. 2-swap local search (remove 2, add 2)
        while time.time() < deadline - 10:
            best_delta_2 = 0
            best_out1 = -1
            best_out2 = -1
            best_in1 = -1
            best_in2 = -1
            
            # For each pair (A, B) in set, find best pair (C, D) not in set
            # Delta = (score_in_set[C] + score_in_set[D] - comp[C,D]) - (score_in_set[A] + score_in_set[B] - comp[A,B])
            # where score_in_set already includes comp with all elements in set
            # Actually, the gain from adding C and D while removing A and B:
            # = sum_{X in set\{A,B}} (comp[C,X] + comp[D,X]) + comp[C,D] 
            #   - sum_{X in set\{A,B}} (comp[A,X] + comp[B,X]) - comp[A,B]
            # = (score_in_set[C] - comp[C,A] - comp[C,B]) + (score_in_set[D] - comp[D,A] - comp[D,B]) + comp[C,D]
            #   - (score_in_set[A] - comp[A,B]) - (score_in_set[B] - comp[B,A])
            
            # Simpler: compute for each pair in set the "cost" and for each pair not in set the "benefit"
            # cost(A,B) = score_in_set[A] + score_in_set[B] - comp[A,B]
            # benefit(C,D) = (score_in_set[C] - comp[C,A] - comp[C,B]) + (score_in_set[D] - comp[D,A] - comp[D,B]) + comp[C,D]
            # But this depends on A,B so we need to iterate.
            
            # For efficiency, iterate over pairs in set and find best pair not in set
            # m=40, so ~780 pairs in set. For each, find best pair not in set (4096^2 is too much)
            # Instead, for each A in set, for each C not in set, compute partial gain, then optimize over B,D
            
            # Alternative: sample a subset of 2-swap moves
            # Try all A in set, all C not in set, then find best B in set\{A} and D in not_in_set\{C}
            
            # For each A in set:
            for A in selected_indices:
                if time.time() > deadline - 10:
                    break
                    
                # For each C not in set:
                # Compute gain_A_C = (score_in_set[C] - comp[C,A]) - (score_in_set[A] - comp[A,C])
                # Then we need to find B in set\{A} and D in not_in_set\{C} to maximize:
                # gain_A_C + (score_in_set[D] - comp[D,A] - comp[D,C] - comp[D,B] + comp[B,D]) - (score_in_set[B] - comp[B,A])
                
                # This is complex. Let's use a simpler approach:
                # For each pair (A,B) in set, compute the "removal cost"
                # For each pair (C,D) not in set, compute the "addition benefit" given A,B removed
                # But that's O(m^2 * N^2) which is too much.
                
                # Let's just try: for each A in set, for each C not in set,
                # find the best B in set\{A} and D in not_in_set\{C}
                
                # For fixed A and C:
                # The delta for swapping out A,B and in C,D is:
                # = [sum_{X in S\{A,B}} (comp[C,X]+comp[D,X])] + comp[C,D]
                #   - [sum_{X in S\{A,B}} (comp[A,X]+comp[B,X])] - comp[A,B]
                
                # = [score_in_set[C] - comp[C,A] - comp[C,B]] + [score_in_set[D] - comp[D,A] - comp[D,B]] + comp[C,D]
                #   - [score_in_set[A] - comp[A,B]] - [score_in_set[B] - comp[B,A]]
                
                # For fixed A, C, this is a function of B and D.
                # Let's precompute for each C not in set: base_C = score_in_set[C] - comp[C,A]
                # Then we need to find B in S\{A} and D in N\{C} to maximize:
                # base_C + (score_in_set[D] - comp[D,A] - comp[D,B]) + comp[C,D]
                #   - (score_in_set[A] - comp[A,B]) - (score_in_set[B] - comp[B,A])
                
                # = base_C - score_in_set[A] + comp[A,B]
                #   + score_in_set[D] - comp[D,A] - comp[D,B] + comp[C,D]
                #   - score_in_set[B] + comp[B,A]
                
                # = (base_C - score_in_set[A]) + comp[A,B] + comp[B,A]
                #   + score_in_set[D] - comp[D,A] - comp[D,B] + comp[C,D] - score_in_set[B]
                
                # Note comp[A,B] = comp[B,A] since comparability is symmetric.
                
                # This is still complex. Let's use a heuristic:
                # For each A in set, find best C not in set (by score_in_set[C] - comp[C,A])
                # Then for that (A,C), find best B in S\{A} and D in N\{C}
                
                # Simplified: just try swapping A with C, and B with D independently
                # i.e., do two single swaps. But that's already covered by single-swap.
                
                # Let's try a different approach: random 2-swap sampling
                # Sample random pairs and evaluate
                pass  # Skip 2-swap for now, focus on improving single-swap and restarts
            
            break  # Skip 2-swap for now
        
        # Update best
        if current_score > best_overall_score:
            best_overall_score = current_score
            best_overall_indices = list(selected_indices)
    
    # Convert to list of lists of Python ints
    if best_overall_indices is None:
        # Fallback: use the last restart's result
        result = []
        for mask in selected_indices:
            mask_int = int(mask)
            subset = [b for b in range(n) if (mask_int >> b) & 1]
            result.append(subset)
        return result
    
    result = []
    for mask in best_overall_indices:
        mask_int = int(mask)
        subset = [b for b in range(n) if (mask_int >> b) & 1]
        result.append(subset)
    
    return result
# EVOLVE-BLOCK-END
