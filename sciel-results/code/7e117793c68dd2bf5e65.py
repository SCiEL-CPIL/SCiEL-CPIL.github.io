import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    total_space = q ** n

    # For q=3, n=7, total_space=2187. Use numpy for vectorized operations.
    # Precompute all words as a 2D array of shape (total_space, n)
    all_words = np.zeros((total_space, n), dtype=np.int32)
    for i in range(total_space):
        idx = i
        for j in range(n):
            all_words[i, j] = idx % q
            idx //= q

    # Precompute cover masks as bitsets.
    # Each word covers itself and all words at Hamming distance <= radius.
    # For radius=1, each word covers 1 + n*(q-1) words.
    # We represent the cover set as a Python int bitmask (2187 bits fits in ~28 ints).
    # Actually, for efficiency, let's use numpy boolean arrays or a list of sets.
    # But bit_count on Python ints is fast. Let's stick to Python ints for masks.
    
    # word_to_idx helper
    def word_to_idx(w):
        idx = 0
        for v in w:
            idx = idx * q + v
        return idx

    # Precompute cover masks
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
                    nw = w.tolist()
                    nw[pos] = val
                    j = word_to_idx(nw)
                    mask |= (1 << j)
        # For radius > 1, we would need more, but task specifies radius 1
        cover_masks[i] = mask

    full_mask = (1 << total_space) - 1

    # Greedy Set Cover with randomization
    def greedy_cover(seed=None):
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random
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
                
            best_idx = rng.choice(candidates)
            code.append(best_idx)
            uncovered &= ~cover_masks[best_idx]
            
        return code

    # Local search: Remove redundant codewords
    def local_search_reduce(code, deadline):
        if not code:
            return code
        
        current_code = code[:]
        
        # Build cover_count array
        cover_count = [0] * total_space
        for idx in current_code:
            m = cover_masks[idx]
            while m:
                low = m & (-m)
                w_idx = low.bit_length() - 1
                cover_count[w_idx] += 1
                m ^= low
        
        improved = True
        while improved:
            improved = False
            random.shuffle(current_code)
            n_cw = len(current_code)
            
            for i in range(n_cw):
                if time.time() > deadline:
                    return current_code
                
                idx = current_code[i]
                mask = cover_masks[idx]
                
                # Check if all words covered by idx are covered by at least one other codeword
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
                    improved = True
                    break
        
        return current_code

    # Local search: Swap codewords to reduce size
    def local_search_swap(code, deadline):
        if len(code) < 2:
            return code
        
        current_code = code[:]
        cover_count = [0] * total_space
        for idx in current_code:
            m = cover_masks[idx]
            while m:
                low = m & (-m)
                w_idx = low.bit_length() - 1
                cover_count[w_idx] += 1
                m ^= low
        
        improved = True
        while improved:
            improved = False
            random.shuffle(current_code)
            n_cw = len(current_code)
            
            for i in range(n_cw):
                if time.time() > deadline:
                    return current_code
                
                idx_i = current_code[i]
                mask_i = cover_masks[idx_i]
                
                # Find words that are only covered by idx_i
                unique_words = []
                m = mask_i
                while m:
                    low = m & (-m)
                    w_idx = low.bit_length() - 1
                    if cover_count[w_idx] == 1:
                        unique_words.append(w_idx)
                    m ^= low
                
                if not unique_words:
                    # This codeword is redundant, remove it
                    m = mask_i
                    while m:
                        low = m & (-m)
                        w_idx = low.bit_length() - 1
                        cover_count[w_idx] -= 1
                        m ^= low
                    current_code.pop(i)
                    improved = True
                    break
                
                # Try to find a swap: replace idx_i with some j such that
                # j covers all unique_words of idx_i and the total size doesn't increase
                # Actually, a swap replaces idx_i with j, so size stays the same.
                # We want to find j such that after removing idx_i and adding j,
                # all words are still covered, and ideally we can then remove another word.
                # Simpler: just try to find j that covers unique_words.
                # But we need to ensure that removing idx_i doesn't uncover words
                # that j doesn't cover. So j must cover all unique_words.
                
                # Let's try a different approach: 1-move local search.
                # Try removing idx_i and adding j.
                # This is expensive. Let's stick to removal for now, and maybe add
                # a simple swap: replace idx_i with j if j covers unique_words
                # and j is not already in code.
                
                # For efficiency, let's just do removal in this function and
                # handle swaps in a separate, more targeted way.
                
                pass
        
        return current_code

    # Initial greedy construction
    best_code = greedy_cover()
    best_code = local_search_reduce(best_code, deadline)
    best_len = len(best_code)
    
    # Run multiple randomized restarts
    if time.time() < deadline - 15.0:
        for trial in range(50):
            if time.time() > deadline - 5.0:
                break
            cand_code = greedy_cover(seed=trial * 12345 + 6789)
            cand_code = local_search_reduce(cand_code, deadline)
            if len(cand_code) < best_len:
                best_len = len(cand_code)
                best_code = cand_code
    
    # Convert indices back to words
    return [all_words[i].tolist() for i in best_code]
