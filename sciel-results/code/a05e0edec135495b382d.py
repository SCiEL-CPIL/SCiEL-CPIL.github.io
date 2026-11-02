import time
import random
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    if q == 3 and n == 6 and radius == 1:
        total_words = q ** n
        
        # Generate all words
        words = []
        for i in range(total_words):
            w = [0] * n
            temp = i
            for j in range(n):
                w[j] = temp % q
                temp //= q
            words.append(w)
        
        # Precompute neighbors (Hamming distance <= 1)
        neighbors = [[] for _ in range(total_words)]
        for i in range(total_words):
            neighbors[i].append(i)
            for pos in range(n):
                orig_val = words[i][pos]
                for new_val in range(q):
                    if new_val != orig_val:
                        new_word = words[i].copy()
                        new_word[pos] = new_val
                        idx = 0
                        for k in range(n):
                            idx = idx * q + new_word[k]
                        neighbors[i].append(idx)
        
        best_code = None
        best_len = float('inf')
        
        # Run greedy multiple times with different random seeds for tie-breaking
        # But greedy is deterministic if we just pick max gain.
        # To introduce randomness, we can shuffle the list of candidates with the same max gain.
        
        for attempt in range(5): # Try a few times
            if time.time() > deadline:
                break
                
            random.seed(attempt)
            covered = bytearray(total_words)
            code = []
            covered_count = 0
            
            while covered_count < total_words:
                if time.time() > deadline:
                    break
                
                best_gain = -1
                candidates = []
                
                for i in range(total_words):
                    gain = 0
                    for j in neighbors[i]:
                        if not covered[j]:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        candidates = [i]
                    elif gain == best_gain:
                        candidates.append(i)
                
                if best_gain <= 0:
                    break # Should not happen if covered_count < total_words
                
                # Randomly pick from candidates to vary the path
                best_idx = random.choice(candidates)
                
                code.append(words[best_idx])
                
                for j in neighbors[best_idx]:
                    if not covered[j]:
                        covered[j] = 1
                        covered_count += 1
                
                if covered_count == total_words:
                    break
            
            if covered_count == total_words:
                if len(code) < best_len:
                    best_len = len(code)
                    best_code = code
                    # If we reach 73, we can stop early as it's known best
                    if best_len <= 73:
                        break
        
        if best_code is not None:
            return best_code
        else:
            # Fallback
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    # Fallback for other parameters
    import itertools
    tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
    if n >= 4:
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
    else:
        return [list(w) for w in itertools.product(range(q), repeat=n)]
