import itertools
import time
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    total_space = q ** n
    
    # Precompute word to index and index to word
    # For q=3, n=7, total_space=2187.
    # We can use a simple base-q conversion.
    
    def word_to_idx(w):
        idx = 0
        for v in w:
            idx = idx * q + v
        return idx

    def idx_to_word(idx):
        w = []
        for i in range(n):
            w.append(idx % q)
            idx //= q
        w.reverse()
        return w

    # Precompute all words
    all_words = [idx_to_word(i) for i in range(total_space)]
    
    # Precompute cover masks
    # mask[i] is a bitmask of indices of words within Hamming distance <= radius of word i
    cover_masks = [0] * total_space
    for i in range(total_space):
        w = all_words[i]
        mask = 1 << i
        if radius >= 1:
            for pos in range(n):
                orig_val = w[pos]
                for val in range(q):
                    if val == orig_val:
                        continue
                    # Create new word
                    nw = w[:]
                    nw[pos] = val
                    j = word_to_idx(nw)
                    mask |= (1 << j)
        # For radius > 1, we would need more logic, but task specifies radius 1
        cover_masks[i] = mask
        
    full_mask = (1 << total_space) - 1
    
    # Greedy Set Cover
    def greedy_cover():
        uncovered = full_mask
        code = []
        while uncovered:
            best_idx = -1
            best_gain = 0
            # Optimization: if uncovered is small, we might find it faster? 
            # No, just scan all. 2187 is small.
            for i in range(total_space):
                # Count how many new words this covers
                gain = (cover_masks[i] & uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
                    # If we cover everything remaining, we can break early?
                    # If gain == uncovered.bit_count(), then we cover all remaining.
                    if gain == uncovered.bit_count():
                        break
                # If best_gain is 1, we can't do better? No, we want max gain.
                # But if best_gain is 1, any other with 1 is equivalent in terms of gain.
                # However, future implications might differ. 
                # For now, standard greedy picks max gain.
                
            if best_idx == -1:
                break # Should not happen if uncovered != 0
            
            code.append(best_idx)
            uncovered &= ~cover_masks[best_idx]
            
        return code

    # Local Search: Remove redundant codewords
    def local_search_reduce(code):
        # code is a list of indices
        if not code:
            return code
            
        # Make a copy to modify
        current_code = code[:]
        
        # We iterate until no more removals are possible
        # To avoid getting stuck in a specific order, we can shuffle
        improved = True
        while improved:
            improved = False
            # Shuffle to ensure we don't miss removals due to order dependency
            # (Removal of A might make B redundant)
            random.shuffle(current_code)
            
            n_cw = len(current_code)
            # Precompute masks for current code for faster access?
            # masks_list = [cover_masks[i] for i in current_code]
            
            for i in range(n_cw):
                if time.time() > deadline:
                    return current_code
                
                # Compute union of all masks except current_code[i]
                # This is O(N) big-int ORs.
                union_others = 0
                # Optimization: If N is large, this is the bottleneck.
                # But N ~ 200, so 200 ORs is fast.
                for j in range(n_cw):
                    if i != j:
                        union_others |= cover_masks[current_code[j]]
                        
                # Check if union_others covers the full space
                if union_others == full_mask:
                    # Remove current_code[i]
                    current_code.pop(i)
                    improved = True
                    # Restart the loop with the new list
                    break # Break inner loop, restart outer with new list
            
        return current_code

    # Initial greedy construction
    best_code = greedy_cover()
    
    # Apply local search
    best_code = local_search_reduce(best_code)
    
    # If we have time, we can try a few randomized restarts of the greedy phase
    # to see if we find a better starting point that reduces better.
    # However, greedy is deterministic. We can add random tie-breaking.
    
    def greedy_cover_randomized():
        uncovered = full_mask
        code = []
        while uncovered:
            best_gain = 0
            candidates = []
            for i in range(total_space):
                gain = (cover_masks[i] & uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    candidates = [i]
                elif gain == best_gain and gain > 0:
                    candidates.append(i)
            
            if not candidates:
                break
                
            # Pick one randomly from candidates
            best_idx = random.choice(candidates)
            code.append(best_idx)
            uncovered &= ~cover_masks[best_idx]
            
        return code

    # Run a few restarts if time permits
    # The first run already used some time.
    # Let's check time.
    if time.time() < deadline - 20.0:
        for _ in range(10): # Try up to 10 restarts
            if time.time() > deadline - 5.0:
                break
            cand_code = greedy_cover_randomized()
            cand_code = local_search_reduce(cand_code)
            if len(cand_code) < len(best_code):
                best_code = cand_code
    
    # Convert indices back to words
    return [all_words[i] for i in best_code]

# EVOLVE-BLOCK-END
