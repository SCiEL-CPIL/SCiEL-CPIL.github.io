"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations

P = {'n': 12, 'm': 40}

def construct():
    start_time = time.time()
    deadline = start_time + 58  # Safety margin for 90s limit
    
    n = P['n']
    m = P['m']
    num_masks = 1 << n
    
    # 1. Precompute comparability matrix using bit operations
    # For subset relation: A ⊆ B iff (A | B) == B iff (A & ~B) == 0
    # We want comp[i,j] = 1 if i ⊂ j or j ⊂ i (strict subset)
    
    # Build subset matrix: sub[i,j] = 1 if mask_i ⊆ mask_j
    # Using bit tricks: mask_i ⊆ mask_j iff (mask_i & ~mask_j) == 0
    # We can compute this efficiently with numpy
    
    # Create arrays of masks
    masks = np.arange(num_masks, dtype=np.int32)
    
    # For each pair (i,j), check if i ⊆ j: (masks[i] & ~masks[j]) == 0
    # This is equivalent to: masks[i] | masks[j] == masks[j]
    mask_i = masks[:, None]
    mask_j = masks[None, :]
    
    # i ⊆ j iff (i | j) == j
    i_subset_j = (mask_i | mask_j) == mask_j
    # j ⊆ i iff (i | j) == i  
    j_subset_i = (mask_i | mask_j) == mask_i
    # Strict subset: one is subset of other but not equal
    not_equal = mask_i != mask_j
    comp = ((i_subset_j | j_subset_i) & not_equal).astype(np.int8)
    
    # Precompute degree (total comparability) for each mask
    degree = comp.sum(axis=1).astype(np.int32)
    
    def local_search(selected_indices, comp, in_set, score_in_set, current_score, deadline, best_score, best_indices):
        """Perform single-swap and systematic 2-swap local search."""
        current_set = set(selected_indices)
        
        # Single-swap local search - use numpy for efficiency
        while time.time() < deadline - 3:
            masked_scores = score_in_set.copy()
            masked_scores[in_set] = -1000
            max_idx = int(np.argmax(masked_scores))
            max_val = int(masked_scores[max_idx])
            
            # Find element with minimum score in current set
            in_set_indices = np.where(in_set)[0]
            min_A_idx = in_set_indices[np.argmin(score_in_set[in_set_indices])]
            min_A_val = int(score_in_set[min_A_idx])
            
            if max_val - min_A_val <= 0:
                break
            
            best_delta = max_val - min_A_val
            best_out = min_A_idx
            best_in = max_idx
            
            score_in_set += comp[:, best_in] - comp[:, best_out]
            
            # Update selected_indices
            pos = selected_indices.index(best_out)
            selected_indices[pos] = best_in
            current_set.remove(best_out)
            current_set.add(best_in)
            in_set[best_out] = False
            in_set[best_in] = True
            
            current_score += best_delta
            if current_score > best_score[0]:
                best_score[0] = current_score
                best_indices[0] = list(selected_indices)
        
        # Systematic 2-swap local search
        # For better efficiency, we'll try all pairs of elements to remove,
        # and for each, find the best pair to add
        while time.time() < deadline - 3:
            best_delta2 = 0
            best_out1 = -1
            best_out2 = -1
            best_in1 = -1
            best_in2 = -1
            
            sel = np.array(selected_indices)
            not_in = np.where(~in_set)[0]
            
            if len(not_in) < 2 or len(sel) < 2:
                break
            
            # For each pair (A,B) to remove, compute the "cost" of removal
            # lost(A,B) = score_in_set[A] + score_in_set[B] - comp[A,B]
            
            # We want to find (C,D) not in set that maximizes:
            # gained(C,D) = score_in_set[C] + score_in_set[D] + comp[C,D] - comp[C,A] - comp[C,B] - comp[D,A] - comp[D,B]
            
            # For a fixed (A,B), this is:
            # = (score_in_set[C] - comp[C,A] - comp[C,B]) + (score_in_set[D] - comp[D,A] - comp[D,B]) + comp[C,D]
            
            # Let adj_C = score_in_set[C] - comp[C,A] - comp[C,B]
            # Then we want max over C≠D of adj_C + adj_D + comp[C,D]
            
            # This is still O(|not_in|^2) per (A,B) pair, which is O(m^2 * |not_in|^2)
            # With m=40, |not_in|=4080, this is 40^2 * 4080^2 ≈ 2.6 billion - too slow
            
            # Instead, use a smarter approach: for each candidate C, precompute
            # the best D to pair with it for each (A,B)
            
            # Alternative: iterate over (C,D) pairs first, then find best (A,B)
            # For fixed (C,D), gained - lost = 
            #   score_in_set[C] + score_in_set[D] + comp[C,D] - comp[C,A] - comp[C,B] - comp[D,A] - comp[D,B] - score_in_set[A] - score_in_set[B] + comp[A,B]
            
            # = (score_in_set[C] + score_in_set[D] + comp[C,D]) - (score_in_set[A] + score_in_set[B] - comp[A,B] + comp[C,A] + comp[C,B] + comp[D,A] + comp[D,B])
            
            # Let base_gain = score_in_set[C] + score_in_set[D] + comp[C,D]
            # Let penalty(A,B) = score_in_set[A] + score_in_set[B] - comp[A,B] + comp[C,A] + comp[C,B] + comp[D,A] + comp[D,B]
            
            # We want to minimize penalty over (A,B) in set
            
            # This is O(|not_in|^2 * m^2) which is still too slow
            
            # Use random sampling but with more samples and better strategy
            num_samples = 2000
            for _ in range(num_samples):
                if time.time() > deadline - 3:
                    break
                    
                # Sample two elements to add (prioritize high-degree candidates)
                not_in_masked = score_in_set[not_in].copy()
                # Weight by degree to prioritize promising candidates
                weights = degree[not_in].astype(np.float64) + 1
                weights /= weights.sum()
                
                C_idx = int(np.random.choice(len(not_in), p=weights))
                D_idx = int(np.random.choice(len(not_in), p=weights))
                if C_idx == D_idx:
                    continue
                C = not_in[C_idx]
                D = not_in[D_idx]
                
                # Sample two elements to remove (prioritize low-score elements)
                sel_masked = score_in_set[sel].copy()
                weights_out = (sel_masked.max() - sel_masked + 1).astype(np.float64)
                weights_out /= weights_out.sum()
                
                A_idx = int(np.random.choice(len(sel), p=weights_out))
                B_idx = int(np.random.choice(len(sel), p=weights_out))
                if A_idx == B_idx:
                    continue
                A = sel[A_idx]
                B = sel[B_idx]
                
                # Compute delta
                lost = score_in_set[A] + score_in_set[B] - comp[A, B]
                gained = score_in_set[C] + score_in_set[D] + comp[C, D] - comp[C, A] - comp[C, B] - comp[D, A] - comp[D, B]
                
                delta = gained - lost
                if delta > best_delta2:
                    best_delta2 = delta
                    best_out1 = A
                    best_out2 = B
                    best_in1 = C
                    best_in2 = D
            
            if best_delta2 <= 0:
                break
            
            # Apply 2-swap
            score_in_set += comp[:, best_in1] + comp[:, best_in2] - comp[:, best_out1] - comp[:, best_out2]
            
            # Update selected_indices
            pos1 = selected_indices.index(best_out1)
            pos2 = selected_indices.index(best_out2)
            selected_indices[pos1] = best_in1
            selected_indices[pos2] = best_in2
            current_set.remove(best_out1)
            current_set.remove(best_out2)
            current_set.add(best_in1)
            current_set.add(best_in2)
            in_set[best_out1] = False
            in_set[best_out2] = False
            in_set[best_in1] = True
            in_set[best_in2] = True
            
            current_score += best_delta2
            if current_score > best_score[0]:
                best_score[0] = current_score
                best_indices[0] = list(selected_indices)
        
        return selected_indices, in_set, score_in_set, current_score
    
    # 2. Greedy Initialization with tie-breaking by degree
    selected = np.zeros(num_masks, dtype=bool)
    selected_indices = []
    
    # Use degree as primary criterion, then comparability with selected set
    for _ in range(m):
        if time.time() > deadline - 10:
            break
        
        if len(selected_indices) == 0:
            # Start with highest degree element
            best_idx = int(np.argmax(degree))
        else:
            # Compute marginal gain for each candidate
            gains = comp[:, selected_indices].sum(axis=1).astype(np.int32)
            gains[selected] = -1000
            
            # Break ties by degree (prefer higher degree for future potential)
            # Use a combined score: gain + small bonus for degree
            combined = gains.astype(np.float64) + 0.01 * degree.astype(np.float64)
            best_idx = int(np.argmax(combined))
            
        selected_indices.append(best_idx)
        selected[best_idx] = True
        
    # 3. Local Search
    current_set = set(selected_indices)
    in_set = np.zeros(num_masks, dtype=bool)
    for idx in selected_indices:
        in_set[idx] = True
        
    score_in_set = comp[:, list(current_set)].sum(axis=1).astype(np.int32)
    
    # Compute initial score
    best_score = [0]
    for idx in selected_indices:
        best_score[0] += score_in_set[idx]
    best_score[0] //= 2
    
    current_score = best_score[0]
    best_indices = [list(selected_indices)]
    
    # Run local search
    selected_indices, in_set, score_in_set, current_score = local_search(
        selected_indices, comp, in_set, score_in_set, current_score, deadline, best_score, best_indices
    )
    
    # Random restarts with varied strategies
    np.random.seed(42)
    num_restarts = 12
    for restart in range(num_restarts):
        if time.time() > deadline - 5:
            break
        
        # Different initialization strategies
        strategy = restart % 4
        
        if strategy == 0:
            # Pure random
            selected_indices = list(np.random.choice(num_masks, size=m, replace=False))
        elif strategy == 1:
            # Greedy with random perturbation
            selected = np.zeros(num_masks, dtype=bool)
            selected_indices = []
            
            for _ in range(m):
                if time.time() > deadline - 5:
                    break
                if len(selected_indices) == 0:
                    best_idx = int(np.argmax(degree))
                else:
                    gains = comp[:, selected_indices].sum(axis=1).astype(np.int32)
                    gains[selected] = -1000
                    # Add randomness
                    noise = np.random.normal(0, 2, num_masks).astype(np.int32)
                    gains += noise
                    best_idx = int(np.argmax(gains))
                    
                selected_indices.append(best_idx)
                selected[best_idx] = True
            
            if len(selected_indices) < m:
                continue
        elif strategy == 2:
            # Start from best solution, perturb a few elements
            selected_indices = list(best_indices[0])
            num_perturb = np.random.randint(2, 6)
            for _ in range(num_perturb):
                pos = np.random.randint(m)
                old = selected_indices[pos]
                # Pick a random replacement
                candidates = list(set(range(num_masks)) - set(selected_indices))
                if candidates:
                    new = int(np.random.choice(candidates))
                    selected_indices[pos] = new
        else:
            # Degree-based greedy with different tie-breaking
            selected = np.zeros(num_masks, dtype=bool)
            selected_indices = []
            
            # Shuffle to break ties differently
            shuffled_degree = degree.copy()
            np.random.shuffle(shuffled_degree)
            
            for _ in range(m):
                if time.time() > deadline - 5:
                    break
                if len(selected_indices) == 0:
                    best_idx = int(np.argmax(shuffled_degree))
                else:
                    gains = comp[:, selected_indices].sum(axis=1).astype(np.int32)
                    gains[selected] = -1000
                    combined = gains.astype(np.float64) + 0.01 * shuffled_degree.astype(np.float64)
                    best_idx = int(np.argmax(combined))
                    
                selected_indices.append(best_idx)
                selected[best_idx] = True
            
            if len(selected_indices) < m:
                continue
        
        current_set = set(selected_indices)
        in_set = np.zeros(num_masks, dtype=bool)
        for idx in selected_indices:
            in_set[idx] = True
        
        score_in_set = comp[:, list(current_set)].sum(axis=1).astype(np.int32)
        
        restart_score = 0
        for idx in selected_indices:
            restart_score += score_in_set[idx]
        restart_score //= 2
        
        # Run local search
        selected_indices, in_set, score_in_set, restart_score = local_search(
            selected_indices, comp, in_set, score_in_set, restart_score, deadline, best_score, best_indices
        )
    
    # Final polish: try single-swap again on best solution
    if time.time() < deadline - 2:
        best_idx_list = best_indices[0]
        current_set = set(best_idx_list)
        in_set = np.zeros(num_masks, dtype=bool)
        for idx in best_idx_list:
            in_set[idx] = True
        
        score_in_set = comp[:, list(current_set)].sum(axis=1).astype(np.int32)
        
        polish_score = best_score[0]
        
        # Try single-swap polish with more iterations
        for _ in range(200):
            if time.time() > deadline - 2:
                break
            masked_scores = score_in_set.copy()
            masked_scores[in_set] = -1000
            max_val = int(np.max(masked_scores))
            max_idx = int(np.argmax(masked_scores))
            
            in_set_indices = np.where(in_set)[0]
            if len(in_set_indices) == 0:
                break
            min_A_idx = in_set_indices[np.argmin(score_in_set[in_set_indices])]
            min_A_val = int(score_in_set[min_A_idx])
            
            if max_val - min_A_val <= 0:
                break
            
            best_delta = max_val - min_A_val
            best_out = min_A_idx
            best_in = max_idx
            
            score_in_set += comp[:, best_in] - comp[:, best_out]
            
            pos = best_idx_list.index(best_out)
            best_idx_list[pos] = best_in
            current_set.remove(best_out)
            current_set.add(best_in)
            in_set[best_out] = False
            in_set[best_in] = True
            
            polish_score += best_delta
            if polish_score > best_score[0]:
                best_score[0] = polish_score
                best_indices[0] = list(best_idx_list)
    
    # Convert to list of lists of Python ints
    result = []
    for mask in best_indices[0]:
        mask_int = int(mask)
        subset = [b for b in range(n) if (mask_int >> b) & 1]
        result.append(subset)
        
    return result
# EVOLVE-BLOCK-END
