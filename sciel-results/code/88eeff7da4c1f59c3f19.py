# EVOLVE-BLOCK-START
import numpy as np
import time

def construct(q, n, radius):
    """
    Optimized greedy set-cover algorithm with high-volume random restarts.
    Uses NumPy matrix-vector multiplication for fast score calculation.
    """
    if n <= 0:
        return [[]]
    
    total_words = q ** n
    
    if total_words > 5000:
        # Fallback for larger spaces, though this task is small
        return _construct_greedy_simple(q, n, radius)
        
    return _construct_greedy_fast(q, n, radius)

def _construct_greedy_fast(q, n, radius):
    """
    Fast greedy set-cover using precomputed bitmask balls and NumPy for scoring.
    """
    total_words = q ** n
    
    # Precompute powers of q
    q_powers = [1] * n
    for i in range(1, n):
        q_powers[i] = q_powers[i-1] * q
        
    # Precompute ball masks (int) and indices (list)
    ball_masks = [0] * total_words
    ball_indices = [[] for _ in range(total_words)]
    
    if radius == 1:
        for i in range(total_words):
            mask = 1 << i
            idx_list = [i]
            for p in range(n):
                step = q_powers[p]
                digit = (i // step) % q
                # For each other value v in 0..q-1
                for v in range(q):
                    if v == digit:
                        continue
                    j = i + (v - digit) * step
                    mask |= 1 << j
                    idx_list.append(j)
            ball_masks[i] = mask
            ball_indices[i] = idx_list
    else:
        # Generic fallback for radius > 1
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
            mask = 1 << i
            idx_list = [i]
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
                        idx_list.append(j)
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
                                idx_list.append(j)
            ball_masks[i] = mask
            ball_indices[i] = idx_list

    # Precompute NumPy array for fast dot product
    # ball_arr[i][j] = 1 if j is in ball(i)
    ball_arr = np.zeros((total_words, total_words), dtype=np.int8)
    for i in range(total_words):
        for j in ball_indices[i]:
            ball_arr[i][j] = 1

    all_mask = (1 << total_words) - 1
    best_codewords = None
    best_len = float('inf')
    
    # Determine number of restarts based on time budget
    start_time = time.time()
    max_time = 110.0 # Leave 10s for overhead
    
    restart_count = 0
    rng = np.random.RandomState(42)
    
    while True:
        # Generate random permutation for tie-breaking
        # We can just use a random priority or simply randomize the initial "uncovered" state?
        # Standard greedy is deterministic. Random restarts usually randomize the order of candidates or the initial choice.
        # Here, we can randomize the "scores" slightly or pick a random starting point?
        # Actually, the standard "randomized greedy" picks the max score, but if there are ties, it breaks randomly.
        # Or it adds noise.
        # Let's stick to the permutation idea but use it to break ties in argmax?
        # np.argmax returns the first occurrence. If we want random tie-breaking, we can add small random noise to scores.
        
        uncovered_int = all_mask
        uncovered_arr = np.ones(total_words, dtype=np.int8)
        codewords = []
        
        while uncovered_int != 0:
            # Calculate scores
            # scores[i] = sum(ball_arr[i] * uncovered_arr)
            scores = ball_arr.dot(uncovered_arr)
            
            # Add small random noise to break ties randomly
            # This simulates the random permutation effect on ties
            noise = rng.random(total_words) * 1e-6
            scores += noise
            
            best_idx = np.argmax(scores)
            max_score = scores[best_idx]
            
            if max_score < 0.5: # Effectively 0
                break
                
            # Convert best_idx to word
            temp = best_idx
            cw = [0] * n
            for k in range(n-1, -1, -1):
                cw[k] = temp % q
                temp //= q
            codewords.append(cw)
                
            # Update uncovered
            uncovered_int &= ~ball_masks[best_idx]
            
            # Update uncovered_arr
            # Set indices in ball_indices[best_idx] to 0
            # Using numpy fancy indexing
            uncovered_arr[ball_indices[best_idx]] = 0
            
        if len(codewords) < best_len:
            best_len = len(codewords)
            best_codewords = codewords
            
            # Early termination if we hit known optimal
            if best_len == 73:
                break
                
        restart_count += 1
        
        # Check time
        if time.time() - start_time > max_time:
            break
            
        # Safety cap on restarts
        if restart_count > 10000:
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
