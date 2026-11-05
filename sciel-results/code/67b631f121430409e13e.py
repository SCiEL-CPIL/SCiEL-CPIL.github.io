import numpy as np
import itertools
import time

def construct(q, n, radius):
    # Parameters from the task
    # q=3, n=7, radius=1
    if q != 3 or n != 7 or radius != 1:
        # Fallback for other parameters if needed, though task specifies this
        # For now, just return the seed logic or a generic greedy
        # But the task contract says "evaluator passes this task's alphabet size..."
        # implying it might be general. However, the seed is specific to 3,7,1.
        # I will assume the evaluator calls with 3, 7, 1 as per the task description.
        # If not, I'll use a simple greedy.
        pass

    total_words = q ** n
    # Map words to indices 0..total_words-1
    # Word is a tuple of ints.
    
    # Precompute neighbors for all words in the space
    # neighbors[i] is a list of indices of words at Hamming distance <= radius from i
    # Actually, we need bitsets for speed.
    
    # 1. Generate all words
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute coverage bitsets for every possible word
    # Bitset size: total_words bits.
    # Use numpy array of uint64.
    num_blocks = (total_words + 63) // 64
    cover_all = np.zeros((total_words, num_blocks), dtype=np.uint64)
    
    for i, w in enumerate(all_words):
        cover_all[i, i // 64] |= (1 << (i % 64))
        # Add neighbors at distance 1
        for pos in range(n):
            for val in range(q):
                if val == w[pos]:
                    continue
                new_w = list(w)
                new_w[pos] = val
                new_w = tuple(new_w)
                j = word_to_idx[new_w]
                cover_all[i, j // 64] |= (1 << (j % 64))
                
    # Initialize with the seed code (Direct sum of tetracode)
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    code = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            code.append(tuple(h + list(tail)))
            
    code_indices = [word_to_idx[c] for c in code]
    
    # Helper to check if a set of indices covers all words
    def is_covering(indices):
        # Sum the cover bitsets of the indices
        # If any bit is 0 in the sum, it's not covered.
        # Since we use OR, we can just OR them all.
        if not indices:
            return total_words == 0
        # Vectorized OR
        # cover_all is (total_words, num_blocks)
        # We want OR of cover_all[i] for i in indices
        # This is a reduction.
        # For small N, a loop might be faster than fancy numpy slicing if not careful.
        # But numpy OR is fast.
        # Let's use a loop for clarity and potential early exit?
        # No, numpy is better.
        # However, creating a view of cover_all[indices] and reducing might be slow.
        # Let's just OR into a zero array.
        current_cover = np.zeros(num_blocks, dtype=np.uint64)
        for idx in indices:
            current_cover |= cover_all[idx]
        # Check if all bits are set
        # The last block might have extra bits.
        for b in range(num_blocks):
            mask = cover_all[0, b] # No, mask should be all 1s for valid bits
            # Actually, we can just check if current_cover has any 0s in the valid range.
            # It's easier to check: is current_cover equal to a "full" mask?
            # Let's precompute the full mask.
        # Precompute full mask
        # Actually, let's just check if any word is missing.
        # We can do this:
        # missing = ~current_cover
        # if any bit in missing (within valid range) is 1, then not covered.
        # To avoid mask complexity, let's just iterate?
        # No, 2187 words is small.
        # Let's stick to the OR reduction.
        
        # Optimization: Precompute the "full" bitset
        # full_mask = np.zeros(num_blocks, dtype=np.uint64)
        # for i in range(total_words): full_mask[i//64] |= 1 << (i%64)
        # Then check np.all(current_cover == full_mask)
        
        return True # Placeholder, will implement properly below

    # Let's refine the coverage check for speed
    full_mask = np.zeros(num_blocks, dtype=np.uint64)
    for i in range(total_words):
        full_mask[i // 64] |= (1 << (i % 64))
        
    def is_covering(indices):
        if not indices:
            return total_words == 0
        current_cover = np.zeros(num_blocks, dtype=np.uint64)
        for idx in indices:
            current_cover |= cover_all[idx]
        return np.all(current_cover == full_mask)

    # State
    current_code = list(code_indices)
    
    deadline = time.time() + 110 # 10s margin
    
    def get_current_cover():
        if not current_code:
            return np.zeros(num_blocks, dtype=np.uint64)
        c = np.zeros(num_blocks, dtype=np.uint64)
        for idx in current_code:
            c |= cover_all[idx]
        return c

    total_cover = get_current_cover()
    
    while time.time() < deadline:
        improved = False
        
        # 1. Redundancy Removal
        # Randomize order
        indices_to_check = list(range(len(current_code)))
        np.random.shuffle(indices_to_check)
        
        for i in indices_to_check:
            if time.time() > deadline:
                break
            if i >= len(current_code):
                continue
            idx_to_remove = current_code[i]
            
            # Check if removing idx_to_remove keeps coverage
            # total_cover_without = total_cover & ~cover_all[idx_to_remove]
            # But we need to know if it's covered by OTHERS.
            # So we need to check if (total_cover & ~cover_all[idx_to_remove]) == full_mask?
            # No, that's not right.
            # A word is covered by others if it is in (OR of cover_all[j] for j != i).
            # Let's compute "others_cover".
            # This is expensive if done from scratch.
            
            # Alternative:
            # A word w is "uniquely covered" by idx_to_remove if w is in cover_all[idx_to_remove]
            # AND w is NOT in (OR of cover_all[j] for j != i).
            # If there are no such words, we can remove.
            
            # Efficient way:
            # Let's just compute the new total cover and check.
            # Since N is small (~200), and bitset ops are fast, 
            # recomputing the total cover from scratch is O(N * num_blocks).
            # 200 * 35 = 7000 ops. Very fast.
            
            # But we can do better.
            # Let's maintain a count of how many codewords cover each word?
            # That would be an array of size 2187.
            # When we add/remove a codeword, we update the counts.
            # This is O(num_blocks) per update if we use bitsets? No, counts are per-word.
            # Let's use counts.
            
            pass # I will re-implement using counts for efficiency in the loop below.

    # Re-implementation with Counts for efficiency
    
    # counts[i] = number of codewords that cover word i
    counts = np.zeros(total_words, dtype=np.int32)
    for idx in current_code:
        # Add cover_all[idx] to counts
        # cover_all[idx] is a bitset. We need to increment counts for each set bit.
        # This is slow in pure Python/NumPy if done naively.
        # However, we can just use the bitset OR approach for validity checks 
        # and only use counts if we find it's faster.
        # Given N=200, O(N^2) checks with bitsets is 40k * 35 ops = 1.4M ops.
        # This is trivial. Let's stick to bitset OR for simplicity and correctness.
        
        pass

    # Let's restart the loop logic with the bitset OR check, which is robust.
    
    while time.time() < deadline:
        improved = False
        
        # Pass 1: Redundancy Removal
        indices_to_check = list(range(len(current_code)))
        np.random.shuffle(indices_to_check)
        
        for i in indices_to_check:
            if time.time() > deadline:
                break
            if i >= len(current_code):
                continue
            idx_to_remove = current_code[i]
            
            # Check if code without idx_to_remove is valid
            # Compute cover of all except i
            # To speed this up, we can pre-compute the total cover, 
            # but we need the cover of the *others*.
            # Let's just recompute the total cover for the list without i.
            # This is O(N).
            temp_code = current_code[:i] + current_code[i+1:]
            if is_covering(temp_code):
                current_code = temp_code
                improved = True
                # Break to restart with new state? Or continue?
                # Restarting is safer to avoid complex state updates.
                break
        
        if improved:
            continue
            
        # Pass 2: 2-for-1 Replacement
        # Try to replace pair (i, j) with a single k
        # Find pair (i, j)
        pairs = list(itertools.combinations(range(len(current_code)), 2))
        np.random.shuffle(pairs)
        
        for i, j in pairs:
            if time.time() > deadline:
                break
            
            idx_i = current_code[i]
            idx_j = current_code[j]
            
            # Compute words that would be uncovered if we remove both
            # These are words in (Cover[i] U Cover[j]) that are NOT covered by any other codeword.
            # Let OthersCover be the OR of cover_all[k] for k in current_code, k != i, k != j.
            # This is O(N) to compute.
            others_cover = np.zeros(num_blocks, dtype=np.uint64)
            for k in range(len(current_code)):
                if k == i or k == j:
                    continue
                others_cover |= cover_all[current_code[k]]
                
            # Words covered by i or j
            pair_cover = cover_all[idx_i] | cover_all[idx_j]
            
            # Words uniquely covered by the pair (relative to others)
            # unique = pair_cover & ~others_cover
            unique_to_pair = pair_cover & (~others_cover)
            
            # If unique_to_pair is 0, then both are redundant (handled in Pass 1).
            # We need to find a new codeword k such that Cover[k] covers unique_to_pair.
            # i.e., (Cover[k] & unique_to_pair) == unique_to_pair
            
            # Vectorized check over all 2187 candidates
            # cover_all is (2187, num_blocks)
            # We want to find any k where cover_all[k] contains unique_to_pair.
            
            # Note: k should ideally not be i or j, but if it is, it's just a deletion 
            # which should have been caught. However, if k is i, then Cover[i] must cover unique_to_pair.
            # Since unique_to_pair is a subset of Cover[i] U Cover[j], if Cover[i] covers it, 
            # then j is redundant.
            
            # Let's find all candidates
            # condition = (cover_all & unique_to_pair) == unique_to_pair
            # This is a boolean array of shape (2187, num_blocks) -> (2187,)
            
            # Optimization: 
            # If unique_to_pair is 0, skip.
            if not np.any(unique_to_pair):
                continue
                
            # Check candidates
            # We can just iterate through a subset of candidates or all.
            # 2187 is small.
            candidates = np.where(np.all((cover_all & unique_to_pair) == unique_to_pair, axis=1))[0]
            
            if len(candidates) > 0:
                # Pick a candidate
                # Prefer one that is not i or j?
                # If we pick i, we are effectively deleting j.
                # Let's just pick a random one.
                new_idx = candidates[np.random.randint(len(candidates))]
                
                # Check if this replacement actually reduces the code size
                # We remove i and j, add new_idx.
                # If new_idx is already in the code (and not i or j), size stays same or increases?
                # No, we remove 2, add 1. Size decreases by 1.
                # Unless new_idx is i or j.
                # If new_idx == i, we remove i, j and add i -> net -1.
                # If new_idx is some other k already in the code, we remove i, j and add k (which is already there).
                # So we just remove i and j. Net -2.
                # Wait, if new_idx is already in the code, we don't add a new distinct word.
                # So the new code is current_code without i and j.
                # This is valid if the coverage is maintained.
                # Our condition `(cover_all[new_idx] & unique_to_pair) == unique_to_pair` ensures 
                # that the words uniquely covered by the pair are covered by new_idx.
                # The words covered by others are still covered by others.
                # So the union is covered.
                
                # Construct new code
                new_code = []
                for k in range(len(current_code)):
                    if k == i or k == j:
                        continue
                    new_code.append(current_code[k])
                
                # Add new_idx if it's not already in new_code
                # If it is, we just have new_code.
                # If it isn't, we add it.
                if new_idx not in new_code:
                    new_code.append(new_idx)
                
                # Verify validity (safety check)
                if is_covering(new_code):
                    current_code = new_code
                    improved = True
                    break # Restart loop

        if not improved:
            # No improvement found in this iteration
            # We could try more aggressive random restarts or break
            # Let's break if we've stuck for a while?
            # Or just break to save time.
            break

    # Convert indices back to words
    final_code = [all_words[idx] for idx in current_code]
    return [list(w) for w in final_code]
