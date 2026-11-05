"""Improved program for ternary covering code K_3(6,1)."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
import heapq

def construct(q, n, radius):
    deadline = time.time() + 115.0

    if q != 3 or n != 6 or radius != 1:
        # Fallback for other parameters
        codewords = []
        covered = [False] * (q ** n)
        def tup_to_idx(t):
            idx = 0
            for i in range(n):
                idx = idx * q + t[i]
            return idx
        for word in itertools.product(range(q), repeat=n):
            w_idx = tup_to_idx(word)
            if not covered[w_idx]:
                codewords.append(list(word))
                covered[w_idx] = True
                for pos in range(n):
                    for val in range(q):
                        if val == word[pos]:
                            continue
                        new_word = list(word)
                        new_word[pos] = val
                        covered[tup_to_idx(tuple(new_word))] = True
                if time.time() > deadline - 5:
                    break
        return codewords

    # Specialized for q=3, n=6, radius=1
    all_words = list(itertools.product(range(3), repeat=6))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute neighbor masks as integers (bitsets)
    # Each word covers itself + 12 neighbors = 13 words
    neighbor_masks = [0] * 729
    for i, w in enumerate(all_words):
        mask = 1 << i
        for pos in range(6):
            for val in range(3):
                if val != w[pos]:
                    new_w = list(w)
                    new_w[pos] = val
                    mask |= 1 << word_to_idx[tuple(new_w)]
        neighbor_masks[i] = mask
    
    full_mask = (1 << 729) - 1
    
    def is_valid_cover(code_indices):
        """Check if the given set of codewords covers all 729 words."""
        union_mask = 0
        for idx in code_indices:
            union_mask |= neighbor_masks[idx]
        return union_mask == full_mask
    
    def greedy_construct(order=None):
        """Greedy set cover: repeatedly pick the word covering most uncovered points."""
        if order is None:
            order = list(range(729))
        random.shuffle(order)
        
        uncovered = full_mask
        code = []
        remaining = set(order)
        
        # Precompute for each word, how many uncovered points it covers
        # Use a priority queue: (-count, word_idx)
        # But we need to update counts as we add words
        
        # Simpler approach: iterate, find best each time
        while uncovered != 0:
            if time.time() > deadline - 10:
                break
            
            best_word = None
            best_count = -1
            best_new_uncovered = None
            
            for w in order:
                # Count how many currently uncovered points this word covers
                covers = neighbor_masks[w] & uncovered
                count = covers.bit_count()
                if count > best_count:
                    best_count = count
                    best_word = w
                    best_new_uncovered = uncovered & ~neighbor_masks[w]
                    if best_count == 0:
                        break
            
            if best_word is None or best_count <= 0:
                break
            
            code.append(best_word)
            uncovered = best_new_uncovered
            
            # Remove best_word from consideration (optional, but speeds up)
            # Actually we can keep it; it just won't be picked again if it covers 0 new points
            # But to be safe, let's not remove; the count check handles it
        
        return code
    
    # Try multiple greedy constructions with different random orderings
    best_code = None
    best_size = 729  # worst case
    
    num_restarts = 20
    for restart in range(num_restarts):
        if time.time() > deadline - 15:
            break
        
        code = greedy_construct()
        if len(code) < best_size:
            best_size = len(code)
            best_code = code[:]
    
    # Local search: try to remove redundant codewords
    def reduce_code(code):
        """Remove redundant codewords from the code."""
        if not code:
            return code
        
        code_set = set(code)
        # Count how many times each point is covered
        counts = [0] * 729
        for idx in code_set:
            mask = neighbor_masks[idx]
            # Iterate through set bits
            m = mask
            while m:
                lsb = m & (-m)
                bit_pos = lsb.bit_length() - 1
                counts[bit_pos] += 1
                m ^= lsb
        
        changed = True
        while changed:
            changed = False
            for idx in list(code_set):
                # Check if removing idx leaves any point with count 0
                mask = neighbor_masks[idx]
                can_remove = True
                m = mask
                while m:
                    lsb = m & (-m)
                    bit_pos = lsb.bit_length() - 1
                    if counts[bit_pos] == 1:
                        can_remove = False
                        break
                    m ^= lsb
                if can_remove:
                    code_set.remove(idx)
                    m = mask
                    while m:
                        lsb = m & (-m)
                        bit_pos = lsb.bit_length() - 1
                        counts[bit_pos] -= 1
                        m ^= lsb
                    changed = True
        
        return list(code_set)
    
    if best_code:
        best_code = reduce_code(best_code)
        best_size = len(best_code)
    
    # Try more local search: swap operations
    # For each pair of codewords, try replacing one with another to see if we can then remove more
    # This is expensive, so we do limited iterations
    
    if best_code and time.time() < deadline - 20:
        current_code = best_code[:]
        current_size = len(current_code)
        
        # Try single swaps: replace one codeword with another, then reduce
        tried_swaps = 0
        max_swaps = 500
        
        for _ in range(max_swaps):
            if time.time() > deadline - 15:
                break
            if tried_swaps >= max_swaps:
                break
            
            if len(current_code) < 2:
                break
            
            i = random.randint(0, len(current_code) - 1)
            # Pick a random other word to swap with
            j = random.randint(0, 728)
            
            if current_code[i] == j:
                continue
            
            new_code = current_code[:]
            new_code[i] = j
            new_code_set = set(new_code)
            
            if len(new_code_set) != len(current_code):
                # Duplicate, skip
                continue
            
            # Try to reduce
            reduced = reduce_code(list(new_code_set))
            if len(reduced) < current_size:
                current_code = reduced
                current_size = len(reduced)
            
            tried_swaps += 1
        
        if current_size < best_size:
            best_code = current_code
            best_size = current_size
    
    # Final validity check
    if best_code is None:
        # Fallback: use all words (valid but huge)
        best_code = list(range(729))
    elif not is_valid_cover(best_code):
        # Fallback
        best_code = list(range(729))
    
    return [list(all_words[idx]) for idx in best_code]

# EVOLVE-BLOCK-END
