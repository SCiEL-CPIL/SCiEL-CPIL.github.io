import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    total_space = q ** n
    
    # Precompute all words as a numpy array of shape (total_space, n)
    # Using base-q expansion
    indices = np.arange(total_space)
    all_words = np.zeros((total_space, n), dtype=np.int32)
    for i in range(n):
        all_words[:, i] = (indices // (q ** (n - 1 - i))) % q

    # Precompute cover masks (spheres of radius 1)
    # For each word, we need the indices of all words within Hamming distance <= radius
    # Since radius is 1 (typical for this problem instance), the sphere size is 1 + n * (q - 1)
    # For q=3, n=7, size = 1 + 7*2 = 15
    
    # We will represent the cover of word i as a bitset (Python int) or a list of indices.
    # Given total_space = 3^7 = 2187, a bitset is a 2187-bit integer, which fits in Python's arbitrary precision ints.
    # However, iterating over bits in Python is slow. 
    # Alternative: Store as a set of indices? Or use NumPy boolean masks?
    # NumPy boolean masks of shape (total_space, total_space) would be 2187*2187 bools ~ 4.7 MB. This is fine.
    # But operations on boolean matrices might be slower than bit-parallel ints for set cover?
    # Let's stick to Python ints for bitsets but optimize the iteration.
    # Actually, for 2187 elements, a Python int is very fast for bitwise ops.
    
    cover_masks = [0] * total_space
    
    # Helper to convert word to index
    def word_to_idx(w):
        idx = 0
        for v in w:
            idx = idx * q + v
        return idx

    # Precompute neighbors for each word
    # For each word i, find all j such that dist(i, j) <= radius
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
                    # Instead of slicing and copying, we can compute the index difference
                    # index = sum(w[k] * q^(n-1-k))
                    # Changing w[pos] to val changes index by (val - orig_val) * q^(n-1-pos)
                    diff = (val - orig_val) * (q ** (n - 1 - pos))
                    j = i + diff
                    # Note: This works because the mixed radix representation is linear in this way
                    # provided we handle the base correctly. 
                    # Let's verify: 
                    # i = w0*q^6 + w1*q^5 + ...
                    # j = w0*q^6 + ... + val*q^(n-1-pos) + ...
                    # j - i = (val - w_pos) * q^(n-1-pos)
                    # This is correct.
                    mask |= (1 << j)
        cover_masks[i] = mask

    full_mask = (1 << total_space) - 1

    # Precompute reverse cover: for each word w, list of codewords that cover it
    # This is useful for finding candidates for swaps/moves
    reverse_cover = [[] for _ in range(total_space)]
    for cw_idx in range(total_space):
        m = cover_masks[cw_idx]
        while m:
            low = m & (-m)
            w_idx = low.bit_length() - 1
            reverse_cover[w_idx].append(cw_idx)
            m ^= low

    # Greedy Set Cover
    def greedy_cover():
        uncovered = full_mask
        code = []
        while uncovered:
            best_idx = -1
            best_gain = 0
            for i in range(total_space):
                gain = (cover_masks[i] & uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
                    if gain == uncovered.bit_count():
                        break
            if best_idx == -1:
                break
            code.append(best_idx)
            uncovered &= ~cover_masks[best_idx]
        return code

    # Local Search with Removal, 1-for-1 Swap, and 2-for-1 Move
    def local_search_reduce(code, time_budget):
        if not code:
            return code
        
        local_deadline = start_time + time_budget
        current_code = code[:]
        current_code_set = set(current_code)
        
        # cover_count[w] = number of codewords covering word w
        cover_count = [0] * total_space
        for idx in current_code:
            m = cover_masks[idx]
            while m:
                low = m & (-m)
                w_idx = low.bit_length() - 1
                cover_count[w_idx] += 1
                m ^= low
        
        improved = True
        while improved and time.time() < local_deadline:
            improved = False
            random.shuffle(current_code)
            n_cw = len(current_code)
            
            # Phase 1: Removal (1-for-0)
            for i in range(n_cw):
                if time.time() > local_deadline:
                    return current_code
                idx = current_code[i]
                mask = cover_masks[idx]
                removable = True
                m = mask
                while m:
                    low = m & (-m)
                    w_idx = low.bit_length() - 1
                    if cover_count[w_idx] <= 1:
                        removable = False
                        break
                    m ^= low
                
                if removable:
                    m = mask
                    while m:
                        low = m & (-m)
                        w_idx = low.bit_length() - 1
                        cover_count[w_idx] -= 1
                        m ^= low
                    current_code.pop(i)
                    current_code_set.remove(idx)
                    improved = True
                    break
            
            # Phase 2: 1-for-1 Swap
            if not improved:
                max_unique_to_cover = 3
                for i in range(n_cw):
                    if time.time() > local_deadline:
                        return current_code
                    idx = current_code[i]
                    mask = cover_masks[idx]
                    
                    unique_mask = 0
                    m = mask
                    while m:
                        low = m & (-m)
                        w_idx = low.bit_length() - 1
                        if cover_count[w_idx] == 1:
                            unique_mask |= (1 << w_idx)
                        m ^= low
                    
                    if unique_mask == 0:
                        # Removable, should have been caught in Phase 1, but just in case
                        m = mask
                        while m:
                            low = m & (-m)
                            w_idx = low.bit_length() - 1
                            cover_count[w_idx] -= 1
                            m ^= low
                        current_code.pop(i)
                        current_code_set.remove(idx)
                        improved = True
                        break
                    
                    if unique_mask.bit_count() > max_unique_to_cover:
                        continue
                    
                    words_in_unique = []
                    m = unique_mask
                    while m:
                        low = m & (-m)
                        w_idx = low.bit_length() - 1
                        words_in_unique.append(w_idx)
                        m ^= low
                    
                    words_in_unique.sort(key=lambda w: len(reverse_cover[w]))
                    
                    candidates = set(reverse_cover[words_in_unique[0]])
                    for w in words_in_unique[1:]:
                        if not candidates:
                            break
                        candidates.intersection_update(reverse_cover[w])
                    
                    candidates.discard(idx)
                    candidates -= current_code_set
                    
                    if candidates:
                        new_idx = random.choice(list(candidates))
                        
                        m = cover_masks[idx]
                        while m:
                            low = m & (-m)
                            w_idx = low.bit_length() - 1
                            cover_count[w_idx] -= 1
                            m ^= low
                        
                        m = cover_masks[new_idx]
                        while m:
                            low = m & (-m)
                            w_idx = low.bit_length() - 1
                            cover_count[w_idx] += 1
                            m ^= low
                        
                        current_code[i] = new_idx
                        current_code_set.remove(idx)
                        current_code_set.add(new_idx)
                        
                        improved = True
                        break

            # Phase 3: 2-for-1 Move
            if not improved and n_cw > 1:
                # Try to find two codewords that can be replaced by one
                # Heuristic: Find a pair (c1, c2) such that U(c1) U U(c2) is small and covered by some a
                # To make this fast, we only check pairs where c1 and c2 are "close" or share many neighbors?
                # Or just random pairs?
                # Let's try a limited number of random pairs to avoid O(N^2) explosion
                num_pair_attempts = 50
                for _ in range(num_pair_attempts):
                    if time.time() > local_deadline:
                        return current_code
                    
                    i, j = random.sample(range(n_cw), 2)
                    c1 = current_code[i]
                    c2 = current_code[j]
                    
                    # Calculate unique words for c1 and c2
                    # Unique to c1: covered by c1, but not by any other codeword except c2?
                    # No, "unique" in the context of removing both means:
                    # Words covered by c1 OR c2, but NOT covered by any other codeword.
                    
                    # Let's compute the set of words covered by c1 or c2
                    # And the set of words covered by others.
                    # If (Cover(c1) U Cover(c2)) - Cover(Others) is covered by some a, we can swap.
                    
                    # Efficient check:
                    # 1. Find words that are covered ONLY by c1 or c2 (or both) among the current code.
                    #    i.e., cover_count[w] == 1 and w in Cover(c1), OR cover_count[w] == 2 and w in Cover(c1) & Cover(c2) and no other covers it?
                    #    Actually, if we remove c1 and c2, a word w becomes uncovered if:
                    #    1. w is covered by c1 or c2
                    #    2. w is NOT covered by any other codeword in current_code \ {c1, c2}
                    
                    # Let's identify the "critical" words:
                    critical_words = []
                    
                    m1 = cover_masks[c1]
                    m2 = cover_masks[c2]
                    m_union = m1 | m2
                    
                    # Iterate through words in m_union
                    m_iter = m_union
                    while m_iter:
                        low = m_iter & (-m_iter)
                        w_idx = low.bit_length() - 1
                        # Check if w_idx is covered by any other codeword
                        # cover_count[w_idx] is the total count.
                        # If cover_count[w_idx] == 1, it must be covered by c1 or c2 (since it's in union).
                        # If cover_count[w_idx] == 2, it might be covered by c1 and c2, or c1 and other, or c2 and other.
                        # We need to know if it's covered by ANY other codeword.
                        
                        # This check is expensive if done naively.
                        # Approximation: If cover_count[w_idx] <= 2, it's a candidate for being critical.
                        # If cover_count[w_idx] == 1, it's definitely critical.
                        # If cover_count[w_idx] == 2, it's critical only if the two coverers are c1 and c2.
                        
                        if cover_count[w_idx] == 1:
                            critical_words.append(w_idx)
                        elif cover_count[w_idx] == 2:
                            # Check if c1 and c2 are the only coverers
                            # We can check if w_idx is in Cover(c1) and Cover(c2)
                            # And if no other codeword covers it.
                            # This is hard to check quickly without iterating all codewords.
                            # Skip for now, only handle cover_count == 1.
                            pass
                        
                        m_iter ^= low
                    
                    if not critical_words:
                        # No critical words, so we can remove c1 and c2 and add nothing? 
                        # No, we need to add something if they covered something.
                        # If no critical words, it means all words covered by c1 or c2 are covered by others.
                        # So we can just remove them? That would be a 2-for-0 move.
                        # But we are in 2-for-1 phase.
                        # If we can remove them, we should have done it in Phase 1?
                        # Phase 1 removes one at a time. If c1 is removable, it would be removed.
                        # So if we reach here, c1 and c2 are not individually removable.
                        # But together they might be removable if their unique sets are disjoint and covered by others?
                        # No, if c1 is not removable, it has a unique word w1.
                        # If c2 is not removable, it has a unique word w2.
                        # If w1 != w2, removing both leaves w1 and w2 uncovered.
                        # So we must add a codeword that covers w1 and w2.
                        
                        continue

                    # Try to find a codeword a that covers all critical_words
                    # If critical_words is small, we can intersect their reverse covers.
                    if len(critical_words) > 4:
                        continue # Too many critical words, unlikely to be covered by one codeword efficiently
                    
                    candidates = set(reverse_cover[critical_words[0]])
                    for w in critical_words[1:]:
                        if not candidates:
                            break
                        candidates.intersection_update(reverse_cover[w])
                    
                    candidates.discard(c1)
                    candidates.discard(c2)
                    candidates -= current_code_set # We want a NEW codeword? Or existing?
                    # If we replace c1, c2 with an EXISTING codeword a, we are just removing c1, c2.
                    # That is a 2-for-0 move.
                    # If we replace with a NEW codeword, it's 2-for-1.
                    # Both reduce size by 1 (if new) or 2 (if existing).
                    # Let's allow any candidate not in {c1, c2}.
                    candidates.difference_update({c1, c2})
                    
                    if candidates:
                        new_idx = random.choice(list(candidates))
                        
                        # Remove c1
                        m = cover_masks[c1]
                        while m:
                            low = m & (-m)
                            w_idx = low.bit_length() - 1
                            cover_count[w_idx] -= 1
                            m ^= low
                        current_code.remove(c1)
                        current_code_set.remove(c1)
                        
                        # Remove c2
                        m = cover_masks[c2]
                        while m:
                            low = m & (-m)
                            w_idx = low.bit_length() - 1
                            cover_count[w_idx] -= 1
                            m ^= low
                        current_code.remove(c2)
                        current_code_set.remove(c2)
                        
                        # Add new_idx
                        m = cover_masks[new_idx]
                        while m:
                            low = m & (-m)
                            w_idx = low.bit_length() - 1
                            cover_count[w_idx] += 1
                            m ^= low
                        current_code.append(new_idx)
                        current_code_set.add(new_idx)
                        
                        improved = True
                        break

        return current_code

    # Initial greedy construction
    best_code = greedy_cover()
    
    # Apply local search
    best_code = local_search_reduce(best_code, time_budget=30.0)
    
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
                
            best_idx = random.choice(candidates)
            code.append(best_idx)
            uncovered &= ~cover_masks[best_idx]
            
        return code

    # Run randomized restarts if time permits
    remaining = deadline - time.time()
    if remaining > 15.0:
        num_restarts = min(20, int(remaining / 5.0))
        for _ in range(num_restarts):
            if time.time() > deadline - 5.0:
                break
            cand_code = greedy_cover_randomized()
            cand_code = local_search_reduce(cand_code, time_budget=3.0)
            if len(cand_code) < len(best_code):
                best_code = cand_code
    
    return [all_words[i].tolist() for i in best_code]
