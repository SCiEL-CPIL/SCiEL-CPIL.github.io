import time
import itertools
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # 1. Generate all words and indices
    # We map word tuple to index 0..q^n-1
    all_words = list(itertools.product(range(q), repeat=n))
    total_words = len(all_words)
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # 2. Precompute coverage masks
    # coverage_mask[i] is an integer where bit j is 1 if word j is within distance 'radius' of word i
    coverage_masks = [0] * total_words
    
    for i, w in enumerate(all_words):
        mask = 0
        # Cover self
        mask |= (1 << i)
        
        if radius == 1:
            # Generate neighbors at distance 1
            for pos in range(n):
                orig_val = w[pos]
                w_list = list(w)
                for new_val in range(q):
                    if new_val != orig_val:
                        w_list[pos] = new_val
                        neighbor = tuple(w_list)
                        j = word_to_idx[neighbor]
                        mask |= (1 << j)
                        w_list[pos] = orig_val # restore
        else:
            # General case: check all (slow, but radius is 1 for this task)
            for j, v in enumerate(all_words):
                if sum(1 for a, b in zip(w, v) if a != b) <= radius:
                    mask |= (1 << j)
        
        coverage_masks[i] = mask
    
    all_ones_mask = (1 << total_words) - 1
    
    # 3. Greedy Set Cover
    # We will run multiple restarts with different random seeds to find a better solution.
    best_chosen = None
    best_size = total_words + 1
    
    # Number of restarts based on time
    # Each greedy pass is fast (729 * 729 ops approx), so we can do many.
    # Let's try 10 restarts.
    num_restarts = 10
    
    for restart in range(num_restarts):
        if time.time() > deadline - 5: # Keep 5s for local search on best
            break
            
        # Randomize the order of candidates for greedy selection
        # Or just pick randomly among top-k?
        # Standard greedy: pick max coverage.
        # To diversify, we can add noise or randomize tie-breaking.
        # Let's use a simple greedy with random tie-breaking.
        
        uncovered = all_ones_mask
        chosen = []
        chosen_set = set()
        
        # Pre-compute list of all indices
        indices = list(range(total_words))
        
        while uncovered:
            if time.time() > deadline - 5:
                break
                
            best_idx = -1
            best_count = 0
            best_indices = []
            
            # Find all indices with max coverage
            for i in indices:
                if i in chosen_set:
                    continue
                    
                common = coverage_masks[i] & uncovered
                if common:
                    count = bin(common).count('1')
                    if count > best_count:
                        best_count = count
                        best_indices = [i]
                    elif count == best_count:
                        best_indices.append(i)
                        
            if best_idx == -1 and best_indices:
                # Pick randomly from best_indices
                best_idx = random.choice(best_indices)
            
            if best_idx == -1:
                break # Should not happen
                
            chosen.append(best_idx)
            chosen_set.add(best_idx)
            uncovered &= ~coverage_masks[best_idx]
            
        # Local Improvement
        # We try to remove or swap codewords to minimize the set size.
        
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            
            # Compute M_c = OR of all masks in chosen except c
            # M_c[i] corresponds to chosen[i]
            # Optimization: Use bit operations.
            # Total OR is all_ones_mask if valid.
            # M_c = total_or ^ coverage_masks[c] ? No, OR is not invertible.
            # We need to recompute or use a clever trick.
            # Since n is small (729), recomputing M_c for each c is O(N^2) bits ops.
            # 729 * 729 bits is fast.
            
            M_c_list = [0] * len(chosen)
            
            # Compute total OR
            total_or = 0
            for c in chosen:
                total_or |= coverage_masks[c]
                
            # If total_or != all_ones_mask, the set is invalid (shouldn't happen from greedy)
            # But local search might break it? No, we only swap if valid.
            
            for idx, c in enumerate(chosen):
                mask_c = 0
                for idx2, c2 in enumerate(chosen):
                    if idx != idx2:
                        mask_c |= coverage_masks[c2]
                M_c_list[idx] = mask_c
            
            # Try Removal
            removed = False
            for idx, c in enumerate(chosen):
                if M_c_list[idx] == all_ones_mask:
                    # Safe to remove
                    chosen.pop(idx)
                    chosen_set.discard(c)
                    improved = True
                    removed = True
                    break
            
            if removed:
                continue
            
            # Try Swap
            # For each c in chosen, try replacing with c' not in chosen
            swapped = False
            for idx, c in enumerate(chosen):
                base_mask = M_c_list[idx]
                missing_mask = all_ones_mask & ~base_mask
                
                if missing_mask == 0:
                    # This means c is redundant, should have been caught in removal
                    # But maybe M_c_list wasn't updated? No, we just computed it.
                    # So if missing is 0, removal should have worked.
                    # However, if we just failed removal, missing_mask != 0 for all.
                    pass
                
                # We need c' such that coverage_masks[c'] covers missing_mask
                # i.e., (missing_mask & ~coverage_masks[c']) == 0
                
                # Iterate through all possible c'
                # To speed up, we can pre-filter?
                # Just iterate all 729.
                for c_prime in range(total_words):
                    if c_prime in chosen_set:
                        continue
                    
                    # Check if c_prime covers all missing bits
                    if (missing_mask & ~coverage_masks[c_prime]) == 0:
                        # Swap c with c_prime
                        chosen[idx] = c_prime
                        chosen_set.discard(c)
                        chosen_set.add(c_prime)
                        improved = True
                        swapped = True
                        break
                
                if swapped:
                    break
                    
            # If no improvement, break
            if not improved:
                break
        
        # Check if this solution is better
        if len(chosen) < best_size:
            best_size = len(chosen)
            best_chosen = list(chosen)
            
    # If no valid solution found (should not happen), fallback to greedy
    if best_chosen is None:
        # Fallback: simple greedy
        uncovered = all_ones_mask
        chosen = []
        chosen_set = set()
        while uncovered:
            best_idx = -1
            best_count = 0
            for i in range(total_words):
                if i in chosen_set:
                    continue
                common = coverage_masks[i] & uncovered
                if common:
                    count = bin(common).count('1')
                    if count > best_count:
                        best_count = count
                        best_idx = i
            if best_idx == -1:
                break
            chosen.append(best_idx)
            chosen_set.add(best_idx)
            uncovered &= ~coverage_masks[best_idx]
        best_chosen = chosen
        
    return [list(all_words[i]) for i in best_chosen]
