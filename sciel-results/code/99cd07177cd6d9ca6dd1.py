import time
import numpy as np
import random

def construct(q, n, radius):
    t_start = time.time()
    deadline = 115.0  # Safety margin before 120s limit
    
    # Helper to convert index to word
    def idx_to_word(idx):
        w = [0] * n
        for i in range(n-1, -1, -1):
            w[i] = idx % q
            idx //= q
        return w

    # Precompute neighbors for radius=1
    space_size = q ** n
    neighbors = np.empty(space_size, dtype=object)
    
    # Powers of q for fast digit access
    q_powers = [1] * (n + 1)
    for i in range(1, n + 1):
        q_powers[i] = q_powers[i-1] * q

    for idx in range(space_size):
        nbs = [idx]
        for pos in range(n):
            digit = (idx // q_powers[n-1-pos]) % q
            for new_digit in range(q):
                if new_digit == digit:
                    continue
                delta = (new_digit - digit) * q_powers[n-1-pos]
                nbs.append(idx + delta)
        neighbors[idx] = np.array(nbs, dtype=np.int32)
        
    # Precompute reverse neighbors (who covers x?)
    # Since Hamming distance is symmetric, neighbors[x] is the list of codewords that cover x.
    # We can reuse the neighbors array for this purpose.
    
    # 1. Generate an initial valid code using greedy set cover
    # We also record the "effectiveness" (number of newly covered words) of each chosen word
    # to build a candidate pool.
    
    uncovered = np.ones(space_size, dtype=bool)
    initial_code = []
    effectiveness = []
    
    # Precompute cover matrix for fast greedy selection
    # cover_masks[i][j] = True if j is in neighbors[i]
    cover_masks = np.zeros((space_size, space_size), dtype=bool)
    for i in range(space_size):
        cover_masks[i, neighbors[i]] = True
        
    while uncovered.any():
        # Find word covering most uncovered words
        counts = cover_masks @ uncovered.astype(np.int8)
        best_idx = np.argmax(counts)
        
        if counts[best_idx] == 0:
            break
            
        initial_code.append(best_idx)
        effectiveness.append(counts[best_idx])
        uncovered[neighbors[best_idx]] = False
        
        if time.time() - t_start > deadline - 20:
            break

    # Ensure we have a valid code
    if not uncovered.all():
        while not uncovered.all():
            uncovered_indices = np.where(uncovered)[0]
            if len(uncovered_indices) == 0:
                break
            w_idx = uncovered_indices[0]
            initial_code.append(w_idx)
            effectiveness.append(1) # Approximate
            uncovered[neighbors[w_idx]] = False
            if time.time() - t_start > deadline - 20:
                break

    # 2. Build a candidate pool
    # The idea: The optimal code is likely composed of words that were "effective" in the greedy phase.
    # We select the top K words based on their greedy effectiveness, plus all words in the initial code.
    # This restricts the search space for the ILS to a manageable subset, allowing for
    # high-quality swaps rather than blind adds/removes.
    
    # Sort words by effectiveness
    # We need to be careful: effectiveness was calculated sequentially.
    # A word chosen later might have lower effectiveness just because it was late.
    # However, words with high effectiveness are generally "central" or "good" coverers.
    
    # Let's define the candidate pool as:
    # 1. All words in initial_code
    # 2. The top M words by effectiveness (M = 2 * len(initial_code))
    
    M = 2 * len(initial_code)
    # Get indices of top M words
    # Note: effectiveness list corresponds to initial_code order.
    # We need to map back to indices.
    
    # Actually, let's just use the indices from initial_code and add some random high-quality words.
    # Better: Re-rank all words by how many neighbors they have? No, all have same degree.
    # Re-rank by "greedy order"? Early greedy picks are usually good.
    
    # Let's construct the pool:
    # Pool = set(initial_code)
    # Plus: The first K words from the greedy sequence that were NOT in the final code? 
    # No, the greedy sequence IS the initial code.
    
    # Let's expand the pool by adding the "next best" words that were not chosen.
    # We can simulate a few more greedy steps to find "good" words that weren't needed.
    
    pool_set = set(initial_code)
    
    # Simulate greedy to find more candidates
    temp_uncovered = np.ones(space_size, dtype=bool)
    temp_code = []
    for _ in range(M):
        counts = cover_masks @ temp_uncovered.astype(np.int8)
        best_idx = np.argmax(counts)
        if counts[best_idx] == 0:
            break
        temp_code.append(best_idx)
        temp_uncovered[neighbors[best_idx]] = False
        
    for idx in temp_code:
        pool_set.add(idx)
        
    # Also add some random words to ensure diversity?
    # For now, let's stick to greedy-derived pool.
    candidate_pool = list(pool_set)
    pool_size = len(candidate_pool)
    
    # 3. Iterated Local Search on the fixed-size problem?
    # No, the size isn't fixed. But we can use a "Fixed-Pool ILS".
    # State: A subset of the candidate_pool.
    # Moves:
    #   - Remove a word from the subset (if valid).
    #   - Add a word from the pool not in the subset (if it helps? No, we want to minimize size).
    #   - Swap: Remove one, add one.
    
    # Let's define the ILS to minimize the size of the subset of the pool that covers the space.
    
    # Initialize with the best subset of the pool found by greedy?
    # The initial_code is a subset of the pool.
    # Let's start with the initial_code.
    
    current_code = initial_code[:]
    current_cov = np.zeros(space_size, dtype=np.int32)
    in_pool = np.zeros(space_size, dtype=bool)
    for c in candidate_pool:
        in_pool[c] = True
        
    in_set = np.zeros(space_size, dtype=bool)
    for c in current_code:
        current_cov[neighbors[c]] += 1
        in_set[c] = True
        
    best_code = current_code[:]
    best_size = len(current_code)
    
    rng = np.random.default_rng(42)
    
    # Helper: Prune the current code (remove redundant words)
    def prune(code, cov, in_set_mask):
        # Remove words that are not the only cover for any neighbor.
        # A word c is removable if min(cov[neighbors[c]]) > 1.
        # We do this repeatedly.
        
        changed = True
        while changed:
            changed = False
            # Find a removable word
            # Optimization: Check words in order of "least criticality"?
            # Just iterate.
            for i in range(len(code) - 1, -1, -1):
                c = code[i]
                # Check if c is critical for any neighbor
                nbs = neighbors[c]
                # If any neighbor has cov == 1, c is critical.
                # If all neighbors have cov > 1, c is removable.
                if np.min(cov[nbs]) > 1:
                    # Remove c
                    code.pop(i)
                    in_set_mask[c] = False
                    cov[nbs] -= 1
                    changed = True
                    # Restart scan? Or continue?
                    # Continuing might miss interactions, but we loop anyway.
                    # Breaking and restarting is safer for correctness of "no redundant" state.
                    break
        return code, cov, in_set_mask

    # Helper: Repair the code if invalid (add words from pool)
    def repair(code, cov, in_set_mask):
        # If any word is uncovered (cov == 0), add a word from the pool that covers it.
        # Greedy repair.
        while True:
            uncovered = np.where(cov == 0)[0]
            if len(uncovered) == 0:
                break
            
            # Find a word in the pool (not in set) that covers the most uncovered words
            # counts = cover_masks @ uncovered.astype(int)
            # Mask out words not in pool?
            # Mask out words already in set?
            
            counts = cover_masks @ uncovered.astype(np.int8)
            
            # Only consider words in the pool
            counts[~in_pool] = -1
            # Only consider words not already in the set
            counts[in_set_mask] = -1
            
            best_idx = np.argmax(counts)
            
            if counts[best_idx] <= 0:
                # No word in pool can cover the uncovered words?
                # This shouldn't happen if pool is large enough, but if it does,
                # we must add a word outside the pool to ensure validity.
                # Fallback: add the uncovered word itself (it covers itself).
                best_idx = uncovered[0]
                # Add to pool? No, just add to code.
                # But if it's not in pool, our "pool-based" logic breaks.
                # Let's just add it.
                in_pool[best_idx] = True # Expand pool if necessary
            
            code.append(best_idx)
            in_set_mask[best_idx] = True
            cov[neighbors[best_idx]] += 1
            
            if time.time() - t_start > deadline - 2:
                break

    # Main ILS Loop
    # Kick: Randomly swap some words or remove/add.
    # Since we want to minimize size, a "kick" could be:
    # 1. Remove K random words (might make invalid).
    # 2. Repair.
    # 3. Prune.
    # This is similar to the previous kick, but restricted to the pool for additions?
    # Actually, repair uses the pool.
    
    # Alternative Kick: "Swap"
    # Pick a word in the code and a word in the pool (not in code).
    # Swap them.
    # If valid, prune.
    
    current_code, current_cov, current_in_set = prune(current_code, current_cov, in_set)
    best_code = current_code[:]
    best_size = len(current_code)
    
    no_improve_count = 0
    max_no_improve = 50
    kick_strength = 2
    
    while time.time() - t_start < deadline - 5:
        # Kick: Remove K random words and repair
        kicked_code = current_code[:]
        kicked_cov = current_cov.copy()
        kicked_in_set = current_in_set.copy()
        
        K = kick_strength
        if len(kicked_code) > K:
            # Pick K random indices to remove
            indices_to_remove = rng.choice(len(kicked_code), size=K, replace=False)
            # Sort descending to remove safely
            indices_to_remove = np.sort(indices_to_remove)[::-1]
            
            for idx in indices_to_remove:
                c = kicked_code[idx]
                kicked_code.pop(idx)
                kicked_in_set[c] = False
                kicked_cov[neighbors[c]] -= 1
        
        # Repair
        kicked_code, kicked_cov, kicked_in_set = repair(kicked_code, kicked_cov, kicked_in_set)
        
        # Prune
        pruned_code, pruned_cov, pruned_in_set = prune(kicked_code, kicked_cov, kicked_in_set)
        
        # Check improvement
        if len(pruned_code) < len(best_code):
            best_code = pruned_code[:]
            best_size = len(pruned_code)
            no_improve_count = 0
            current_code, current_cov, current_in_set = pruned_code, pruned_cov, pruned_in_set
        else:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                kick_strength += 1
                if kick_strength > 15:
                    kick_strength = 2
                no_improve_count = 0
        
        # Time check
        if time.time() - t_start > deadline - 10:
            break

    # Final check and return
    # Ensure best_code is valid
    final_cov = np.zeros(space_size, dtype=np.int32)
    for c in best_code:
        final_cov[neighbors[c]] += 1
        
    if np.any(final_cov == 0):
        # Fallback to initial_code
        return [idx_to_word(c) for c in initial_code]
    else:
        return [idx_to_word(c) for c in best_code]

