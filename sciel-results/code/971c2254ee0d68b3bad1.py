import time
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 145.0

    # Handle the specific known hard case efficiently
    # For other cases, a simpler greedy might suffice, but we focus on q=3, n=7, R=1
    if q == 3 and n == 7 and radius == 1:
        N = 3**n  # 2187
        
        # Precompute power of 3 for index calculation
        pow3 = [1] * n
        for i in range(1, n):
            pow3[i] = pow3[i-1] * 3

        # Precompute covers as bitsets (Python ints)
        # covers[i] has bits set for all words within distance 1 of i
        covers = [0] * N
        for i in range(N):
            mask = 1 << i
            # Calculate neighbors
            # To avoid idx_to_word conversion, we can derive neighbors from index
            # But idx_to_word is easy.
            # Let's just convert i to word, modify, convert back.
            # Or use the property: neighbor at pos p with val v is i + (v - w[p]) * 3**p
            
            # Get word digits
            tmp = i
            digits = [0] * n
            for p in range(n):
                digits[p] = tmp % 3
                tmp //= 3
            
            for p in range(n):
                orig = digits[p]
                for new_val in range(3):
                    if new_val == orig:
                        continue
                    # Calculate new index
                    # j = i - orig * 3**p + new_val * 3**p
                    j = i + (new_val - orig) * pow3[p]
                    mask |= (1 << j)
            covers[i] = mask

        best_code = None
        best_size = float('inf')
        
        # Run randomized greedy restarts
        # We use a standard greedy set cover algorithm with random tie-breaking.
        
        while time.time() < deadline:
            uncovered = (1 << N) - 1
            code = []
            
            while uncovered:
                max_gain = -1
                best_candidates = []
                
                # Scan all candidates
                # Optimization: We can skip candidates that are fully covered?
                # No, a covered word might still cover uncovered neighbors.
                # However, if a word is covered, its coverage is a subset of its neighbors' coverage?
                # Not necessarily.
                
                # Just scan all.
                for i in range(N):
                    # Bitwise AND and popcount
                    gain = (uncovered & covers[i]).bit_count()
                    if gain > max_gain:
                        max_gain = gain
                        best_candidates = [i]
                    elif gain == max_gain:
                        best_candidates.append(i)
                
                if max_gain == 0:
                    break # Safety break, should not happen if logic is correct
                
                # Randomly pick from best candidates to allow different paths
                chosen_idx = random.choice(best_candidates)
                
                # Convert index to word for output
                tmp = chosen_idx
                word = [0] * n
                for p in range(n):
                    word[p] = tmp % 3
                    tmp //= 3
                code.append(word)
                
                # Update uncovered set
                uncovered &= ~covers[chosen_idx]
            
            if len(code) < best_size:
                best_size = len(code)
                best_code = code
                
                # If we found a very good code, we might want to continue to see if we can do better
                # 186 is the target.
                
        return best_code
    else:
        # Generic fallback for other parameters
        # Simple greedy set cover
        from itertools import product
        all_words = list(product(range(q), repeat=n))
        N = len(all_words)
        
        # Map word to index
        idx_map = {w: i for i, w in enumerate(all_words)}
        
        # Precompute covers
        covers = []
        for w in all_words:
            c = set()
            c.add(w)
            for i in range(n):
                for val in range(q):
                    if val == w[i]:
                        continue
                    new_w = list(w)
                    new_w[i] = val
                    c.add(tuple(new_w))
            covers.append(c)
            
        uncovered = set(all_words)
        code = []
        
        while uncovered and time.time() < deadline:
            best_word = None
            best_gain = -1
            for w in all_words:
                gain = len(uncovered & covers[idx_map[w]])
                if gain > best_gain:
                    best_gain = gain
                    best_word = w
            if best_word is None:
                break
            code.append(list(best_word))
            uncovered -= covers[idx_map[best_word]]
            
        return code
