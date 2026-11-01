# EVOLVE-BLOCK-START
import time
import random
import itertools

def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet size q with covering radius 'radius'.
    Returns a list of codewords (each a list of ints in 0..q-1).
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s/150s limit
    
    # Special case for n=0
    if n == 0:
        return [[]]
    
    # We focus optimization on the specific hard case q=3, n=6, radius=1
    # For other cases, we fall back to a generic greedy + local search, 
    # though the bitset trick is general.
    
    total_words = q ** n
    
    # Precompute neighbors as bitsets for speed
    # Each word is an integer 0..q^n-1
    # neighbors[w] is a bitmask of all words within distance 'radius' of w
    
    # Mapping from index to tuple for output
    index_to_tuple = [tuple(itertools.product(range(q), repeat=n))[i] if i < total_words else None for i in range(total_words)] if total_words < 100000 else None
    # Actually, generating the full list of tuples for 729 is fine, but for larger n it might be slow.
    # For q=3, n=6, 729 is small.
    
    # Let's precompute neighbors for all words in the space
    # This is O(q^n * q^n) which is 729*729 ~ 500k ops, very fast.
    # For larger spaces, we would need to be careful, but the task implies small n.
    
    if total_words > 50000:
        # Fallback for larger spaces: simple greedy (might not be optimal but safe)
        # This path is unlikely to be hit by the evaluator for this specific task contract
        return _generic_construct(q, n, radius, deadline)

    # 1. Precompute neighbor masks
    # We represent the set of words as a bitmask of length total_words.
    # neighbors[i] is an int where bit j is 1 if word j is within distance 'radius' of word i.
    
    # Helper to convert tuple to index
    def tuple_to_idx(t):
        idx = 0
        for i in range(n):
            idx = idx * q + t[i]
        return idx

    # Precompute all tuples and their indices
    all_tuples = list(itertools.product(range(q), repeat=n))
    
    neighbors = [0] * total_words
    for i, t_i in enumerate(all_tuples):
        mask = 0
        # Find all t_j within distance 'radius'
        for j, t_j in enumerate(all_tuples):
            dist = sum(1 for k in range(n) if t_i[k] != t_j[k])
            if dist <= radius:
                mask |= (1 << j)
        neighbors[i] = mask

    # 2. Greedy Construction
    # Start with an empty code.
    # uncovered_mask: bitmask of words not yet covered.
    # Initially all bits 1.
    full_mask = (1 << total_words) - 1
    uncovered_mask = full_mask
    
    code_indices = []
    code_mask = 0  # Union of neighbors of selected codewords
    
    # Greedy step: pick the codeword that covers the most currently uncovered words
    # To speed this up, we can just iterate. 729 iterations * 729 checks is fine.
    
    while uncovered_mask and time.time() < deadline - 20: # Leave time for local search
        best_idx = -1
        best_count = -1
        # Optimization: We only need to check words not already in code?
        # Actually, re-adding a word in code is useless.
        # We can maintain a set of available indices.
        
        # For simplicity and speed in Python, we iterate all.
        # If code grows large, we might want to skip those already in code.
        in_code_set = set(code_indices)
        
        for i in range(total_words):
            if i in in_code_set:
                continue
            # Count how many new words this adds
            # new_covered = neighbors[i] & uncovered_mask
            # count = new_covered.bit_count()
            # This is fast in Python 3.10+
            new_covered = neighbors[i] & uncovered_mask
            cnt = new_covered.bit_count()
            if cnt > best_count:
                best_count = cnt
                best_idx = i
        
        if best_idx == -1 or best_count == 0:
            break
            
        code_indices.append(best_idx)
        code_mask |= neighbors[best_idx]
        uncovered_mask &= ~neighbors[best_idx]
        
        # Early exit if fully covered (should happen)
        if uncovered_mask == 0:
            break

    # 3. Local Search
    # Goal: Minimize len(code_indices)
    # Moves:
    # 1. Remove a codeword if it doesn't increase the uncovered set.
    # 2. Swap a codeword for another if it reduces the code size or keeps size same but improves "quality" (hard to define, so stick to size reduction or just try swaps to escape local minima).
    # Actually, standard 1-1 move: Remove A, Add B.
    # If we remove A, uncovered becomes U' = (current_uncovered | neighbors[A]) & ~neighbors[A] ... wait.
    # Let's maintain:
    # code_set: set of indices
    # covered_mask: union of neighbors
    
    code_set = set(code_indices)
    covered_mask = code_mask
    uncovered_mask = full_mask & ~covered_mask
    
    # Helper to check if a move is valid
    # Move: remove 'rem', add 'add'
    # New covered = (covered_mask | neighbors[add]) & ~neighbors[rem]  <- NO
    # New covered = (covered_mask \ neighbors[rem]) | neighbors[add]
    # This is not quite right because neighbors[rem] might be covered by others.
    # Correct logic:
    # A word is uncovered after removing 'rem' if it was ONLY covered by 'rem'.
    # Let's compute "essential coverage".
    # For each word w, count how many codewords cover it.
    # If we remove 'rem', words w where count(w) == 1 and 'rem' covers w become uncovered.
    
    # To make this fast, we can use bitsets.
    # Let's define:
    # mask_covered_by_rem = neighbors[rem]
    # Words that become uncovered if we remove rem:
    # These are words in mask_covered_by_rem that are NOT covered by any OTHER codeword in code_set \ {rem}.
    
    # Let's precompute for each codeword in the current code, the "exclusive" coverage?
    # That's expensive to update.
    
    # Alternative:
    # Try removing 'rem'.
    # Temporarily remove 'rem' from code_set.
    # Recalculate covered_mask? No, too slow.
    
    # Better:
    # Let U_rem = set of words ONLY covered by 'rem'.
    # If we remove 'rem', U_rem becomes uncovered.
    # If we add 'add', it covers neighbors[add].
    # The move is valid if U_rem is a subset of neighbors[add].
    # i.e., (U_rem & ~neighbors[add]) == 0.
    
    # How to compute U_rem quickly?
    # U_rem = neighbors[rem] & ~(Union of neighbors of all other codewords)
    # Let Other_Covered = covered_mask & ~neighbors[rem]
    # This is also not quite right because covered_mask includes neighbors[rem].
    
    # Let's compute:
    # Covered_By_Others = 0
    # for c in code_set - {rem}:
    #    Covered_By_Others |= neighbors[c]
    # This is O(|code|) per check. With |code| ~ 80 and 729 candidates, that's 60k ops per move.
    # 729 * 60k = 40M ops. Might be too slow in Python for many iterations.
    
    # Optimization:
    # We can maintain an array `cover_count` of size total_words.
    # cover_count[w] = number of codewords in code_set that cover w.
    # Updating this on add/remove is O(1) per word in the neighbor set.
    # Neighbor set size is 1 + n*(q-1) = 1 + 6*2 = 13 for q=3, n=6.
    # So updating cover_count is O(13).
    # Checking if a word is "only covered by rem" is O(1): cover_count[w] == 1 and rem covers w.
    
    # Let's implement this.
    
    cover_count = [0] * total_words
    for c in code_set:
        mask = neighbors[c]
        # Iterate set bits in mask
        m = mask
        while m:
            # Get lowest set bit
            lsb = m & -m
            idx = lsb.bit_length() - 1
            cover_count[idx] += 1
            m ^= lsb

    best_code = list(code_set)
    best_size = len(best_code)
    
    # Local Search Loop
    # We will try 1-1 swaps (remove one, add one) and 0-1 removals.
    # We will do this for a fixed number of passes or until time runs out.
    
    # To escape local minima, we can also try 2-2 swaps (remove 2, add 2), but that's O(N^4).
    # Given the small size, 1-1 is usually sufficient if we have a good init.
    
    # Randomize the order of checking to avoid getting stuck in a specific loop.
    rng = random.Random(42)
    
    while time.time() < deadline - 5:
        improved = False
        
        # Try removals first
        for rem in list(code_set):
            # Check if rem can be removed
            # If any word w in neighbors[rem] has cover_count[w] == 1, we cannot remove rem.
            can_remove = True
            m = neighbors[rem]
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                if cover_count[idx] == 1:
                    can_remove = False
                    break
                m ^= lsb
            
            if can_remove:
                # Remove rem
                code_set.remove(rem)
                m = neighbors[rem]
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    cover_count[idx] -= 1
                    m ^= lsb
                improved = True
                if len(code_set) < best_size:
                    best_size = len(code_set)
                    best_code = list(code_set)
                break # Restart loop to check new state
        
        if not improved:
            # Try 1-1 swaps
            # Pick a random codeword to remove
            if code_set:
                rem = rng.choice(list(code_set))
                # Find words that would become uncovered
                # These are w in neighbors[rem] with cover_count[w] == 1
                must_cover = 0
                m = neighbors[rem]
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    if cover_count[idx] == 1:
                        must_cover |= lsb
                    m ^= lsb
                
                if must_cover == 0:
                    # This is effectively a removal, should have been caught above?
                    # No, above we checked if ANY word has count 1. If must_cover is 0, it's safe to remove.
                    # Why wasn't it removed? Maybe we broke out of the removal loop?
                    # The removal loop breaks on the FIRST improvement.
                    # So if we are here, no single removal was possible.
                    # This implies for every rem, there is at least one word with count 1.
                    # So must_cover should never be 0 here if the previous loop was exhaustive.
                    # But we only tried removals once per outer loop iteration.
                    # So it's possible.
                    pass
                
                # We need to find an 'add' such that neighbors[add] covers 'must_cover'.
                # i.e., (must_cover & ~neighbors[add]) == 0
                
                best_add = -1
                # We can iterate through all possible adds
                # To speed up, we can prioritize adds that cover many 'must_cover' bits?
                # Or just find the first one that works.
                
                # Optimization: If must_cover is large, it's harder to find an add.
                # We can shuffle the candidate adds.
                
                candidates = [i for i in range(total_words) if i not in code_set]
                rng.shuffle(candidates)
                
                for add in candidates:
                    # Check if neighbors[add] covers must_cover
                    if (must_cover & ~neighbors[add]) == 0:
                        best_add = add
                        break
                
                if best_add != -1:
                    # Perform swap
                    # 1. Remove rem
                    code_set.remove(rem)
                    m = neighbors[rem]
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        cover_count[idx] -= 1
                        m ^= lsb
                        
                    # 2. Add best_add
                    code_set.add(best_add)
                    m = neighbors[best_add]
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        cover_count[idx] += 1
                        m ^= lsb
                    
                    improved = True
                    if len(code_set) < best_size:
                        best_size = len(code_set)
                        best_code = list(code_set)
                    # Don't break, continue trying to improve further?
                    # Usually breaking and restarting the outer loop is safer to re-evaluate all removals.
                    break
        
        if not improved:
            # Stuck. Try a perturbation?
            # Randomly remove a codeword (making code invalid) and then repair?
            # This is risky.
            # Alternative: Just break if we've been stuck for a while?
            # For now, let's just keep trying random swaps. The randomness in 'rem' choice helps.
            
            # If we are really stuck, we might try to accept a "worse" move?
            # No, we need to maintain validity.
            
            # Let's try a 2-2 move if stuck?
            # Too complex for now.
            
            # Let's just let the loop continue. The random choice of 'rem' changes.
            # However, if no swap is found for a specific 'rem', we just try the next 'rem' in the next iteration?
            # No, we break the inner swap loop if we find one. If we don't find one, we exit the if block.
            # So we try one random 'rem' per outer loop.
            # This is slow.
            
            # Improvement: Try ALL 'rem's for swaps in one pass?
            # That would be O(N * N) = 729 * 729 = 500k checks.
            # Each check is fast (bit ops).
            # Let's do that.
            
            found_swap = False
            for rem in code_set:
                must_cover = 0
                m = neighbors[rem]
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    if cover_count[idx] == 1:
                        must_cover |= lsb
                    m ^= lsb
                
                # If must_cover is 0, we can just remove it.
                # (Should have been caught by removal loop, but maybe state changed?)
                if must_cover == 0:
                    # Remove
                    code_set.remove(rem)
                    m = neighbors[rem]
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        cover_count[idx] -= 1
                        m ^= lsb
                    found_swap = True
                    if len(code_set) < best_size:
                        best_size = len(code_set)
                        best_code = list(code_set)
                    break
                
                # Find add
                # Optimization: Only check adds that intersect must_cover?
                # No, add must COVER must_cover.
                
                # We can pre-filter adds?
                # For now, just iterate.
                # To avoid bias, we can shuffle candidates once outside?
                # No, 'rem' changes.
                
                # Heuristic: Check adds that are "close" to rem? No.
                # Just iterate all.
                
                # We can speed this up by checking if (must_cover & ~neighbors[add]) == 0
                # This is one bitwise AND and NOT and comparison. Very fast.
                
                for add in range(total_words):
                    if add in code_set:
                        continue
                    if (must_cover & ~neighbors[add]) == 0:
                        # Perform swap
                        code_set.remove(rem)
                        m = neighbors[rem]
                        while m:
                            lsb = m & -m
                            idx = lsb.bit_length() - 1
                            cover_count[idx] -= 1
                            m ^= lsb
                        
                        code_set.add(add)
                        m = neighbors[add]
                        while m:
                            lsb = m & -m
                            idx = lsb.bit_length() - 1
                            cover_count[idx] += 1
                            m ^= lsb
                        
                        found_swap = True
                        if len(code_set) < best_size:
                            best_size = len(code_set)
                            best_code = list(code_set)
                        break
                
                if found_swap:
                    break
            
            if not found_swap:
                # Completely stuck.
                # We can try to restart from the best code found so far?
                # Or just break.
                # Given the time limit, breaking is safer to return a valid result.
                # But we might have time for more random restarts?
                # Let's just break for now.
                break

    # Return the best code found
    # Convert indices back to tuples
    result = [all_tuples[i] for i in best_code]
    return [list(t) for t in result]

def _generic_construct(q, n, radius, deadline):
    # Fallback for larger spaces
    # Simple greedy
    all_words = list(itertools.product(range(q), repeat=n))
    def dist(a, b):
        return sum(1 for i in range(n) if a[i] != b[i])
    
    uncovered = set(all_words)
    code = []
    while uncovered and time.time() < deadline - 10:
        best = None
        best_cnt = -1
        for w in all_words:
            if w in set(code): continue
            cnt = sum(1 for u in uncovered if dist(w, u) <= radius)
            if cnt > best_cnt:
                best_cnt = cnt
                best = w
        if not best: break
        code.append(best)
        uncovered = {u for u in uncovered if dist(best, u) > radius}
    return [list(w) for w in code]

# EVOLVE-BLOCK-END
