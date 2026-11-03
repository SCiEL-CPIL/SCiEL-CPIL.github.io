# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def construct(q, n, radius):
    """
    Optimized greedy set-cover algorithm with high-volume random restarts.
    Designed for small spaces (q^n < 10000) where bitmask operations are efficient.
    Includes a post-processing reduction phase to remove redundant codewords.
    """
    if n <= 0:
        return [[]]
    
    total_words = q ** n
    
    if total_words > 5000:
        # Fallback for larger spaces, though this task is small
        return _construct_greedy_simple(q, n, radius)
        
    return _construct_greedy_optimized(q, n, radius)

def _construct_greedy_optimized(q, n, radius):
    """
    Optimized greedy set-cover using precomputed bitmask balls.
    Uses vectorized-like list comprehensions for speed.
    Includes a reduction phase and a local search phase.
    """
    total_words = q ** n
    
    # Precompute powers of q
    q_powers = [1] * n
    for i in range(1, n):
        q_powers[i] = q_powers[i-1] * q
        
    # Precompute ball masks
    ball_masks = [0] * total_words
    
    if radius == 1:
        for i in range(total_words):
            mask = 1 << i
            for p in range(n):
                step = q_powers[p]
                digit = (i // step) % q
                for v in range(q):
                    if v == digit:
                        continue
                    j = i + (v - digit) * step
                    mask |= 1 << j
            ball_masks[i] = mask
    else:
        # Generic fallback for radius > 1
        def idx_to_word(idx):
            word = [0] * n
            for k in range(n-1, -1, -1):
                word[k] = idx % q
                idx //= q
            return word
        
        for i in range(total_words):
            mask = 1 << i
            center = idx_to_word(i)
            if radius == 2:
                # Distance 1
                for p in range(n):
                    step = q_powers[p]
                    digit = (i // step) % q
                    for v in range(q):
                        if v == digit: continue
                        j = i + (v - digit) * step
                        mask |= 1 << j
                # Distance 2
                for p1 in range(n):
                    for p2 in range(p1+1, n):
                        step1 = q_powers[p1]
                        step2 = q_powers[p2]
                        d1 = (i // step1) % q
                        d2 = (i // step2) % q
                        for v1 in range(q):
                            if v1 == d1: continue
                            for v2 in range(q):
                                if v2 == d2: continue
                                j = i + (v1 - d1) * step1 + (v2 - d2) * step2
                                mask |= 1 << j
            ball_masks[i] = mask

    all_mask = (1 << total_words) - 1
    best_codewords = None
    best_len = float('inf')
    
    start_time = time.time()
    max_time = 110.0
    restart_count = 0
    rng = np.random.RandomState(42)
    
    def idx_to_word(idx):
        word = [0] * n
        temp = idx
        for k in range(n-1, -1, -1):
            word[k] = temp % q
            temp //= q
        return word

    # Precompute list of all indices for shuffling
    all_indices = list(range(total_words))

    def local_search_reduce(indices):
        """
        Aggressive reduction and local search to minimize the set of indices.
        1. Remove redundant words.
        2. Try to swap a word c with a non-word w if it allows removing another word.
        """
        if not indices:
            return []
        
        active = list(indices)
        # coverage_count[x] = number of active indices that cover x
        coverage_count = [0] * total_words
        
        # Initialize coverage
        for idx in active:
            m = ball_masks[idx]
            while m:
                lsb = m & -m
                w_idx = lsb.bit_length() - 1
                coverage_count[w_idx] += 1
                m ^= lsb
        
        improved = True
        while improved:
            improved = False
            
            # Phase 1: Remove redundant words
            i = 0
            while i < len(active):
                idx = active[i]
                mask = ball_masks[idx]
                m = mask
                is_redundant = True
                while m:
                    lsb = m & -m
                    w_idx = lsb.bit_length() - 1
                    if coverage_count[w_idx] == 1:
                        is_redundant = False
                        break
                    m ^= lsb
                
                if is_redundant:
                    # Remove idx
                    m = mask
                    while m:
                        lsb = m & -m
                        w_idx = lsb.bit_length() - 1
                        coverage_count[w_idx] -= 1
                        m ^= lsb
                    active.pop(i)
                    improved = True
                else:
                    i += 1
            
            # Phase 2: Try swaps (1-in, 1-out) that enable removal
            # For each c in active, for each w not in active:
            # Check if replacing c with w allows removing some d.
            # This is expensive, so we limit it or do it smartly.
            # We'll just try to find if there's a w that covers the "unique" part of c.
            # If so, swap c->w, then check if any d becomes redundant.
            
            # To speed up, we only try this if we haven't reached the target size (e.g., 73)
            # And we limit the number of swap attempts per iteration.
            
            if len(active) > 73:
                # Create a set for fast lookup
                active_set = set(active)
                
                # Iterate over a copy of active to allow modification
                for i in range(len(active)):
                    c = active[i]
                    # Find words uniquely covered by c
                    mask_c = ball_masks[c]
                    m = mask_c
                    unique_mask = 0
                    while m:
                        lsb = m & -m
                        w_idx = lsb.bit_length() - 1
                        if coverage_count[w_idx] == 1:
                            unique_mask |= lsb
                        m ^= lsb
                    
                    if unique_mask == 0:
                        continue # Should have been removed in Phase 1
                        
                    # Find a w not in active_set such that unique_mask is subset of ball_masks[w]
                    # Iterate through all possible w? 729 is small.
                    found_swap = False
                    # Shuffle order of w to randomize
                    perm_w = rng.permutation(all_indices).tolist()
                    for w in perm_w:
                        if w in active_set:
                            continue
                        if (unique_mask & ~ball_masks[w]) == 0:
                            # w covers all unique parts of c
                            # Tentatively swap c -> w
                            # Update counts: remove c, add w
                            m_c = ball_masks[c]
                            while m_c:
                                lsb = m_c & -m_c
                                w_idx = lsb.bit_length() - 1
                                coverage_count[w_idx] -= 1
                                m_c ^= lsb
                                
                            m_w = ball_masks[w]
                            while m_w:
                                lsb = m_w & -m_w
                                w_idx = lsb.bit_length() - 1
                                coverage_count[w_idx] += 1
                                m_w ^= lsb
                            
                            # Check if any d in active \ {c} is now redundant
                            removal_idx = -1
                            for j in range(len(active)):
                                if j == i:
                                    continue
                                d = active[j]
                                mask_d = ball_masks[d]
                                m_d = mask_d
                                is_red = True
                                while m_d:
                                    lsb = m_d & -m_d
                                    w_idx = lsb.bit_length() - 1
                                    if coverage_count[w_idx] == 1:
                                        is_red = False
                                        break
                                    m_d ^= lsb
                                if is_red:
                                    removal_idx = j
                                    break
                            
                            if removal_idx != -1:
                                # Accept swap and removal
                                # Remove d
                                m_d = ball_masks[removal_idx]
                                while m_d:
                                    lsb = m_d & -m_d
                                    w_idx = lsb.bit_length() - 1
                                    coverage_count[w_idx] -= 1
                                    m_d ^= lsb
                                
                                # Replace c with w in active
                                active[i] = w
                                active.pop(removal_idx)
                                active_set.remove(c)
                                active_set.add(w)
                                
                                improved = True
                                found_swap = True
                                break # Restart the while improved loop
                            else:
                                # Undo swap
                                m_c = ball_masks[c]
                                while m_c:
                                    lsb = m_c & -m_c
                                    w_idx = lsb.bit_length() - 1
                                    coverage_count[w_idx] += 1
                                    m_c ^= lsb
                                    
                                m_w = ball_masks[w]
                                while m_w:
                                    lsb = m_w & -m_w
                                    w_idx = lsb.bit_length() - 1
                                    coverage_count[w_idx] -= 1
                                    m_w ^= lsb
                            
                    if found_swap:
                        break # Restart the while improved loop
        
        return active

    while True:
        # Shuffle indices for random tie-breaking / order
        perm = rng.permutation(all_indices).tolist()
        
        uncovered = all_mask
        codeword_indices = []
        
        while uncovered != 0:
            # Calculate scores for all candidates
            scores = [ (ball_masks[idx] & uncovered).bit_count() for idx in perm ]
            
            max_score = max(scores)
            
            if max_score == 0:
                break
                
            best_candidates = [idx for idx, score in zip(perm, scores) if score == max_score]
            best_idx = rng.choice(best_candidates)
            
            codeword_indices.append(best_idx)
            uncovered &= ~ball_masks[best_idx]
            
        # Reduction and Local Search Phase
        if len(codeword_indices) > 0:
            final_indices = local_search_reduce(codeword_indices)
            
            codewords = [idx_to_word(idx) for idx in final_indices]
            
            if len(codewords) < best_len:
                best_len = len(codewords)
                best_codewords = codewords
                
                if best_len == 73:
                    break
                    
        restart_count += 1
        
        if time.time() - start_time > max_time:
            break
            
        if restart_count > 20000:
            break
            
    return best_codewords

def _construct_greedy_simple(q, n, radius):
    """Fallback simple greedy for larger spaces."""
    total_words = q ** n
    
    def word_to_idx(word):
        idx = 0
        for i in range(n):
            idx = idx * q + word[i]
        return idx
    
    def idx_to_word(idx):
        word = [0] * n
        for i in range(n - 1, -1, -1):
            word[i] = idx % q
            idx //= q
        return word
    
    ball_masks = [0] * total_words
    for idx in range(total_words):
        center = idx_to_word(idx)
        mask = 1 << idx
        if radius == 1:
            for pos in range(n):
                for val in range(q):
                    if val != center[pos]:
                        neighbor = center.copy()
                        neighbor[pos] = val
                        nidx = word_to_idx(neighbor)
                        mask |= 1 << nidx
        ball_masks[idx] = mask
        
    uncovered = (1 << total_words) - 1
    codewords = []
    while uncovered != 0:
        best_idx = -1
        best_count = -1
        for idx in range(total_words):
            covered = ball_masks[idx] & uncovered
            count = covered.bit_count()
            if count > best_count:
                best_count = count
                best_idx = idx
        if best_idx == -1:
            break
        codewords.append(idx_to_word(best_idx))
        uncovered &= ~ball_masks[best_idx]
    return codewords
# EVOLVE-BLOCK-END
