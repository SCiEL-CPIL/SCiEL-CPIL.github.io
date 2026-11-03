# EVOLVE-BLOCK-START
import numpy as np
import time

def construct(q, n, radius):
    """
    Greedy set-cover for covering codes: select codewords maximizing uncovered coverage.
    """
    if n <= 0:
        return [[]]
    
    total_words = q ** n
    q_powers = [1] * n
    for i in range(1, n):
        q_powers[i] = q_powers[i-1] * q
    
    # Precompute ball masks for all words
    ball_masks = _precompute_balls(q, n, radius, q_powers)
    
    all_mask = (1 << total_words) - 1
    best_codewords = None
    best_len = float('inf')
    
    start_time = time.time()
    max_time = 115.0
    
    # Multiple restarts with different tie-breaking strategies
    rng = np.random.RandomState(42)
    restart_count = 0
    
    while True:
        # Strategy: random permutation for tie-breaking
        perm = rng.permutation(total_words)
        
        uncovered = all_mask
        codewords = []
        
        while uncovered != 0:
            max_score = -1
            best_idx = -1
            
            for idx in perm:
                score = (ball_masks[idx] & uncovered).bit_count()
                if score > max_score:
                    max_score = score
                    best_idx = idx
            
            if max_score <= 0:
                break
            
            temp = best_idx
            cw = [0] * n
            for k in range(n-1, -1, -1):
                cw[k] = temp % q
                temp //= q
            codewords.append(cw)
            uncovered &= ~ball_masks[best_idx]
        
        if len(codewords) < best_len:
            best_len = len(codewords)
            best_codewords = codewords
            
            # Early termination for known optimum
            if q == 3 and n == 6 and radius == 1 and best_len == 73:
                break
        
        restart_count += 1
        
        if time.time() - start_time > max_time:
            break
        
        if restart_count > 20000:
            break
    
    return best_codewords

def _precompute_balls(q, n, radius, q_powers):
    """
    Precompute bitmask of all words within Hamming distance 'radius' of each word.
    """
    total_words = q ** n
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
    elif radius == 2:
        for i in range(total_words):
            mask = 1 << i
            # Distance 1
            for p in range(n):
                step = q_powers[p]
                digit = (i // step) % q
                for v in range(q):
                    if v == digit:
                        continue
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
                        if v1 == d1:
                            continue
                        for v2 in range(q):
                            if v2 == d2:
                                continue
                            j = i + (v1 - d1) * step1 + (v2 - d2) * step2
                            mask |= 1 << j
            ball_masks[i] = mask
    else:
        # Generic brute-force for radius > 2
        def idx_to_word(idx):
            word = [0] * n
            for k in range(n-1, -1, -1):
                word[k] = idx % q
                idx //= q
            return word
        
        def word_to_idx(word):
            idx = 0
            for k in range(n):
                idx = idx * q + word[k]
            return idx
        
        for i in range(total_words):
            center = idx_to_word(i)
            mask = 0
            for j in range(total_words):
                word_j = idx_to_word(j)
                dist = sum(1 for a, b in zip(center, word_j) if a != b)
                if dist <= radius:
                    mask |= 1 << j
            ball_masks[i] = mask
    
    return ball_masks
# EVOLVE-BLOCK-END
