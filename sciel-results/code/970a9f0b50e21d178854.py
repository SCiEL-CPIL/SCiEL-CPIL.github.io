import itertools
import time
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit

    if q != 3 or n != 8 or radius != 1:
        return _general_greedy(q, n, radius, deadline - start_time)

    total_words = 3 ** n
    balls = np.empty((total_words, 17), dtype=np.int32)
    
    def int_to_base3(x, n):
        digits = [0] * n
        for i in range(n - 1, -1, -1):
            digits[i] = x % 3
            x //= 3
        return digits

    def base3_to_int(digits):
        res = 0
        for d in digits:
            res = res * 3 + d
        return res

    for x in range(total_words):
        digits = int_to_base3(x, n)
        ball_indices = [x]
        for j in range(n):
            for v in range(3):
                if v != digits[j]:
                    new_digits = digits[:]
                    new_digits[j] = v
                    ball_indices.append(base3_to_int(new_digits))
        balls[x] = ball_indices

    remaining = np.ones(total_words, dtype=bool)
    codewords = []
    
    # Greedy Set Cover (Hypergraph Edge Coloring Heuristic)
    while remaining.any():
        if time.time() > deadline - 10:
            break
            
        # Vectorized count of uncovered vertices (words) in each hyperedge (ball)
        counts = remaining[balls].sum(axis=1)
        best_candidate = np.argmax(counts)
        best_count = counts[best_candidate]
        
        if best_count == 0:
            break
            
        codewords.append(best_candidate)
        remaining[balls[best_candidate]] = False

    # Redundancy Removal
    cover_count = np.zeros(total_words, dtype=np.int32)
    for c in codewords:
        cover_count[balls[c]] += 1
        
    changed = True
    while changed:
        if time.time() > deadline - 5:
            break
        changed = False
        new_codewords = []
        for c in codewords:
            if cover_count[balls[c]].min() >= 2:
                cover_count[balls[c]] -= 1
                changed = True
            else:
                new_codewords.append(c)
        codewords = new_codewords

    result = [int_to_base3(c, n) for c in codewords]
    
    if remaining.any():
        while remaining.any():
            if time.time() > deadline - 1:
                break
            idx = np.where(remaining)[0][0]
            codewords.append(idx)
            remaining[balls[idx]] = False
        result = [int_to_base3(c, n) for c in codewords]

    return result

def _general_greedy(q, n, radius, time_limit):
    # Simple fallback for non-3,8,1 cases
    # Not optimized, but functional
    total = q ** n
    covered = np.zeros(total, dtype=bool)
    codewords = []
    
    def int_to_word(x):
        w = [0] * n
        for i in range(n - 1, -1, -1):
            w[i] = x % q
            x //= q
        return w

    def word_to_int(w):
        res = 0
        for d in w:
            res = res * q + d
        return res

    # Generate balls on the fly or precompute if small
    # For general case, precomputation might be heavy if q^n is large.
    # But task is specific to 3,8,1.
    # This is just a safety net.
    
    start = time.time()
    while covered.sum() < total:
        if time.time() - start > time_limit - 5:
            break
        best_c = -1
        best_count = -1
        for c in range(total):
            if covered[c]: continue # Optimization: if c is covered, it might still cover others, but usually we pick uncovered ones? No, pick max coverage.
            # Calculate coverage
            w = int_to_word(c)
            count = 0
            # Check self
            if not covered[c]: count += 1
            # Check neighbors
            for j in range(n):
                for v in range(q):
                    if v == w[j]: continue
                    w2 = w[:]
                    w2[j] = v
                    idx = word_to_int(w2)
                    if not covered[idx]: count += 1
            if count > best_count:
                best_count = count
                best_c = c
        
        if best_c == -1: break
        codewords.append(best_c)
        # Update covered
        w = int_to_word(best_c)
        covered[best_c] = True
        for j in range(n):
            for v in range(q):
                if v == w[j]: continue
                w2 = w[:]
                w2[j] = v
                covered[word_to_int(w2)] = True
        
    return [int_to_word(c) for c in codewords]
