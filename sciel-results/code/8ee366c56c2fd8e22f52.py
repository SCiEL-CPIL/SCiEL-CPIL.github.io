import time
import numpy as np
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    n = 128
    min_sum = 128
    max_sum = 256
    target_sums = np.arange(min_sum, max_sum + 1)
    num_targets = len(target_sums)
    
    # Precompute pair contributions to speed up objective evaluation
    # pairs_i, pairs_j: indices of elements in A that form the pair
    # pair_target_idx: index in target_sums array (s - 128)
    pairs_i = []
    pairs_j = []
    pair_target_idx = []
    
    for i in range(129):
        for j in range(i, 129):
            s = i + j
            if min_sum <= s <= max_sum:
                idx = s - min_sum
                pairs_i.append(i)
                pairs_j.append(j)
                pair_target_idx.append(idx)
                
    pairs_i = np.array(pairs_i)
    pairs_j = np.array(pairs_j)
    pair_target_idx = np.array(pair_target_idx)
    num_pairs = len(pairs_i)
    
    # Initial set: A = [64, 65, ..., 128]
    # This is a valid cover: 64+64=128, 128+128=256.
    # Size 65.
    # Let's calculate its peak to initialize.
    A_vec = np.zeros(129, dtype=np.int8)
    A_vec[64:129] = 1
    best_A_vec = A_vec.copy()
    
    def compute_counts(A_vec):
        active = A_vec[pairs_i] * A_vec[pairs_j]
        counts = np.bincount(active * pair_target_idx, weights=active, minlength=num_targets)
        return counts

    best_counts = compute_counts(best_A_vec)
    best_peak = np.max(best_counts)
    
    # Simulated Annealing Parameters
    T = 10.0  # Initial temperature
    cooling_rate = 0.9995
    min_T = 0.1
    max_iterations = 50000
    
    # To make objective smooth for SA decisions, we can use a soft-max or just the max.
    # Since we need to maintain coverage, we must penalize uncovered sums heavily.
    # Objective: max(counts) + 100 * sum(counts == 0)
    
    def get_score(counts):
        max_mult = np.max(counts)
        uncovered = np.sum(counts == 0)
        return max_mult + 100.0 * uncovered

    current_score = get_score(best_counts)
    best_score = current_score
    
    rng = random.Random(42)
    
    # Precompute for fast updates? 
    # Updating counts for a single element change (add/remove) requires updating 
    # all pairs involving that element.
    # Element k is in pairs where i=k or j=k.
    # We can precompute for each element k, the list of pair indices and target indices.
    
    elem_pair_indices = [[] for _ in range(129)]
    elem_pair_targets = [[] for _ in range(129)]
    
    for p_idx in range(num_pairs):
        i = pairs_i[p_idx]
        j = pairs_j[p_idx]
        t = pair_target_idx[p_idx]
        
        # If i != j, it contributes to both i's and j's lists?
        # No, the pair (i,j) is a single unit. 
        # If we change A[i], we affect all pairs (i, j) for j>=i and (j, i) for j<i?
        # Our pairs are i<=j.
        # If we toggle A[i]:
        # Pairs (i, j) for j>=i are affected.
        # Pairs (j, i) for j<i are NOT in our list (since j<i, the pair is (j,i) with j as first).
        # Wait, our list has (i,j) with i<=j.
        # So for element i, the affected pairs are:
        # 1. (i, j) where j >= i
        # 2. (j, i) where j < i -> this is stored as (j, i) in the list.
        
        # So for element k, we need to find all pairs (p_idx) such that pairs_i[p_idx] == k OR pairs_j[p_idx] == k.
        
        elem_pair_indices[k].append(p_idx)
        elem_pair_targets[k].append(t)
        
        # Note: if i==j, it's the same pair. We only add once per pair index.
        # The loop above adds p_idx to elem_pair_indices[i] and elem_pair_indices[j].
        # If i==j, it adds p_idx to elem_pair_indices[i] twice?
        # No, the loop iterates p_idx. 
        # If i==j, we append to elem_pair_indices[i] and elem_pair_indices[j] which is same.
        # So we need to be careful not to double count if we just sum.
        # Actually, for a given pair (i,j), if we toggle i, the contribution of THIS pair changes.
        # If we toggle j, the contribution of THIS pair changes.
        # If i==j, toggling i changes the pair. Toggling j (same i) changes the same pair.
        # So for element k, the set of affected pairs is unique.
        # Let's deduplicate just in case, or handle i==j carefully.
        
    # Deduplicate
    for k in range(129):
        if len(elem_pair_indices[k]) > 0:
            # The list might have duplicates if i==j? 
            # In the loop:
            # p_idx is unique.
            # If pairs_i[p_idx] == k, we append.
            # If pairs_j[p_idx] == k, we append.
            # If pairs_i[p_idx] == pairs_j[p_idx] == k, we append p_idx twice.
            # We should use a set or check.
            unique_idx = set(elem_pair_indices[k])
            elem_pair_indices[k] = list(unique_idx)
            # Re-align targets? No, we need the target for each pair index.
            # Let's rebuild properly.
            
    # Rebuild elem_pair_indices and elem_pair_targets correctly
    elem_pair_indices = [[] for _ in range(129)]
    elem_pair_targets = [[] for _ in range(129)]
    
    for p_idx in range(num_pairs):
        i = pairs_i[p_idx]
        j = pairs_j[p_idx]
        t = pair_target_idx[p_idx]
        
        elem_pair_indices[i].append(p_idx)
        elem_pair_targets[i].append(t)
        
        if j != i:
            elem_pair_indices[j].append(p_idx)
            elem_pair_targets[j].append(t)
        # If j == i, we only add once because toggling i (which is j) affects the pair once.
        # Wait, if i==j, the pair is (i,i). Toggling A[i] changes the state of pair (i,i).
        # It is one pair. So it should be in the list of affected pairs for i exactly once.
        # The code above adds to i's list. Since j==i, we don't add to j's list separately (it's the same list).
        # So it is added once. Correct.
        
    # Convert to numpy arrays for speed? Or keep as lists? Lists are fine for small counts.
    
    current_A = best_A_vec.copy()
    current_counts = best_counts.copy()
    
    for it in range(max_iterations):
        if time.time() > deadline:
            break
        if T < min_T:
            break
            
        # Select a random move: Add, Remove, or Swap
        move_type = rng.choice(['add', 'remove', 'swap'])
        
        if move_type == 'add':
            # Pick a random element not in A
            candidates = np.where(current_A == 0)[0]
            if len(candidates) == 0:
                continue
            k = rng.choice(candidates)
            
            # Calculate delta counts
            # Toggling k from 0 to 1 adds 1 to all pairs involving k.
            affected_pairs = elem_pair_indices[k]
            affected_targets = elem_pair_targets[k]
            
            new_counts = current_counts.copy()
            # new_counts[affected_targets] += 1
            np.add.at(new_counts, affected_targets, 1)
            
            new_score = get_score(new_counts)
            delta = new_score - current_score
            
            # Acceptance probability
            if delta < 0 or rng.random() < np.exp(-delta / T):
                current_A[k] = 1
                current_counts = new_counts
                current_score = new_score
                
                if current_score < best_score:
                    best_score = current_score
                    best_A_vec = current_A.copy()
                    best_counts = current_counts.copy()
                    
        elif move_type == 'remove':
            # Pick a random element in A
            candidates = np.where(current_A == 1)[0]
            if len(candidates) == 0:
                continue
            k = rng.choice(candidates)
            
            affected_pairs = elem_pair_indices[k]
            affected_targets = elem_pair_targets[k]
            
            new_counts = current_counts.copy()
            # new_counts[affected_targets] -= 1
            np.subtract.at(new_counts, affected_targets, 1)
            
            new_score = get_score(new_counts)
            delta = new_score - current_score
            
            if delta < 0 or rng.random() < np.exp(-delta / T):
                current_A[k] = 0
                current_counts = new_counts
                current_score = new_score
                
                if current_score < best_score:
                    best_score = current_score
                    best_A_vec = current_A.copy()
                    best_counts = current_counts.copy()
                    
        else: # swap
            # Pick one in A, one not in A
            in_A = np.where(current_A == 1)[0]
            not_in_A = np.where(current_A == 0)[0]
            if len(in_A) == 0 or len(not_in_A) == 0:
                continue
            k_in = rng.choice(in_A)
            k_out = rng.choice(not_in_A)
            
            # Remove k_in, Add k_out
            # Delta = - (pairs of k_in) + (pairs of k_out)
            # Note: if a pair involves both k_in and k_out, it is removed then added?
            # Pair (k_in, k_out) exists if k_in <= k_out or k_out <= k_in.
            # If we remove k_in, pair (k_in, k_out) count goes 1->0.
            # If we add k_out, pair (k_in, k_out) count goes 0->1?
            # No, the pair is defined by the set A.
            # If A contains {k_in, k_out}, the pair (min, max) is present.
            # Before: k_in in A, k_out not in A. Pair (k_in, k_out) NOT present.
            # After: k_in not in A, k_out in A. Pair (k_in, k_out) NOT present.
            # So the pair (k_in, k_out) status doesn't change?
            # Wait.
            # If k_in < k_out:
            # Before: k_in in, k_out out. Pair (k_in, k_out) requires both. So 0.
            # After: k_in out, k_out in. Pair (k_in, k_out) requires both. So 0.
            # So the interaction term cancels out?
            # Let's verify.
            # Counts are sum of indicators.
            # C_new = C_old - Sum(pairs with k_in) + Sum(pairs with k_out)
            # Pairs with k_in: includes (k_in, k_out) if k_in != k_out.
            # Pairs with k_out: includes (k_in, k_out) if k_in != k_out.
            # So -1 for (k_in, k_out) and +1 for (k_in, k_out) cancels.
            # So we can just apply the deltas independently?
            # Yes, because the change in the pair (k_in, k_out) is 0.
            
            affected_in = elem_pair_indices[k_in]
            targets_in = elem_pair_targets[k_in]
            affected_out = elem_pair_indices[k_out]
            targets_out = elem_pair_targets[k_out]
            
            new_counts = current_counts.copy()
            np.subtract.at(new_counts, targets_in, 1)
            np.add.at(new_counts, targets_out, 1)
            
            new_score = get_score(new_counts)
            delta = new_score - current_score
            
            if delta < 0 or rng.random() < np.exp(-delta / T):
                current_A[k_in] = 0
                current_A[k_out] = 1
                current_counts = new_counts
                current_score = new_score
                
                if current_score < best_score:
                    best_score = current_score
                    best_A_vec = current_A.copy()
                    best_counts = current_counts.copy()
                    
        T *= cooling_rate
        
    # Final check and return
    # Ensure best_A_vec is valid
    final_counts = compute_counts(best_A_vec)
    if np.any(final_counts == 0):
        # Fallback to initial valid set
        best_A_vec = np.zeros(129, dtype=np.int8)
        best_A_vec[64:129] = 1
        final_counts = compute_counts(best_A_vec)
        
    # Convert to list of indices
    result_A = [i for i, v in enumerate(best_A_vec) if v == 1]
    
    # If result is empty (should not happen), fallback
    if not result_A:
        result_A = list(range(64, 129))
        
    return result_A
