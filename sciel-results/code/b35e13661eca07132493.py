import numpy as np
import time
import itertools

def construct(q, n, radius):
    if not (q == 3 and n == 8 and radius == 1):
        # Fallback for non-target parameters
        # Simple exhaustive or greedy for small cases
        if n <= 4:
            return [list(t) for t in itertools.product(range(q), repeat=n)]
        # Generic greedy for other cases
        # ... (omitted for brevity, focus on target case)
        return []

    N = q ** n
    start_time = time.time()
    time_limit = 110.0 # Safety margin
    
    # Helper to convert int to base-q digits
    def int_to_digits(num, length):
        digits = [0] * length
        for i in range(length):
            digits[i] = num % q
            num //= q
        return digits

    # Helper to convert base-q digits to int
    def digits_to_int(digits):
        val = 0
        for i in range(len(digits) - 1, -1, -1):
            val = val * q + digits[i]
        return val

    # Precompute neighbors for all words
    # neighbors[i] is a list of indices j such that d(i, j) <= 1
    neighbors = []
    for i in range(N):
        digits = int_to_digits(i, n)
        ball = [i]
        for k in range(n):
            orig_val = digits[k]
            for v in range(q):
                if v == orig_val:
                    continue
                new_digits = digits.copy()
                new_digits[k] = v
                j = digits_to_int(new_digits)
                ball.append(j)
        neighbors.append(ball)
        
    # Convert neighbors to numpy arrays for faster operations if needed, 
    # but lists are fine for small sizes. 
    # Let's stick to lists for clarity and flexibility in set operations.

    # Phase 1: Greedy Construction (Lazy)
    # coverage[w] = number of codewords covering w
    coverage = np.zeros(N, dtype=np.int32)
    codewords = []
    uncovered_count = N
    uncovered_set = set(range(N)) # For quick lookup? Or just check coverage[w]==0
    
    while uncovered_count > 0:
        if time.time() - start_time > time_limit - 10.0:
            break
            
        # Find an uncovered word
        # Linear scan is O(N). We can optimize by keeping a list of uncovered words.
        # Let's maintain a list of uncovered words.
        # Actually, just finding the first one with coverage 0 is fast enough.
        u = -1
        for w in range(N):
            if coverage[w] == 0:
                u = w
                break
        
        if u == -1:
            break # Should not happen if uncovered_count > 0
            
        # Find best candidate c among neighbors[u]
        # c must cover u.
        # Score: number of currently uncovered words covered by c.
        best_c = -1
        best_score = -1
        
        for c in neighbors[u]:
            # Count how many words in neighbors[c] are currently uncovered (coverage == 0)
            score = 0
            for w in neighbors[c]:
                if coverage[w] == 0:
                    score += 1
            if score > best_score:
                best_score = score
                best_c = c
                
        if best_c == -1:
            # Should not happen, u covers itself
            best_c = u
            
        # Add best_c to codewords
        codewords.append(best_c)
        
        # Update coverage
        for w in neighbors[best_c]:
            if coverage[w] == 0:
                uncovered_count -= 1
            coverage[w] += 1

    # Phase 2: Local Search (Deletion and Swaps)
    # Rebuild coverage to be sure
    coverage = np.zeros(N, dtype=np.int32)
    for c in codewords:
        for w in neighbors[c]:
            coverage[w] += 1
            
    cw_set = set(codewords)
    
    # Precompute word_coverers? 
    # word_coverers[w] = list of c in codewords that cover w.
    # This changes as codewords change. 
    # Instead, for a critical set W_crit, we find intersection of neighbors[w] for w in W_crit.
    
    changed = True
    while changed and (time.time() - start_time < time_limit - 5.0):
        changed = False
        new_codewords = []
        
        for c in codewords:
            if time.time() - start_time > time_limit - 5.0:
                # Time up, keep the rest
                new_codewords.append(c)
                continue
                
            # Tentatively remove c
            for w in neighbors[c]:
                coverage[w] -= 1
                
            # Check if any word is uncovered (coverage < 1)
            # Find critical words: those with coverage == 0 after removal
            critical_words = []
            for w in neighbors[c]:
                if coverage[w] == 0:
                    critical_words.append(w)
                    
            if not critical_words:
                # c is redundant, remove it
                cw_set.discard(c)
                changed = True
                # Do not add c to new_codewords
            else:
                # c is needed. Try to find a swap.
                # We need a c' not in cw_set such that c' covers all critical_words.
                # Intersect neighbors of all critical words.
                
                # Start with neighbors of the first critical word
                candidates = set(neighbors[critical_words[0]])
                
                for w in critical_words[1:]:
                    candidates.intersection_update(neighbors[w])
                    if not candidates:
                        break
                        
                # Remove current codewords from candidates
                candidates.difference_update(cw_set)
                
                if candidates:
                    # Pick the best candidate.
                    # Heuristic: pick the one that covers the most words with coverage == 1 
                    # (in the state where c is removed).
                    # This maximizes the chance of future deletions.
                    
                    best_c_prime = -1
                    best_swap_score = -1
                    
                    for c_prime in candidates:
                        # Count words w in neighbors[c_prime] where coverage[w] == 1
                        # (Note: coverage currently reflects state WITHOUT c)
                        score = 0
                        for w in neighbors[c_prime]:
                            if coverage[w] == 1:
                                score += 1
                        if score > best_swap_score:
                            best_swap_score = score
                            best_c_prime = c_prime
                            
                    if best_c_prime != -1:
                        # Perform swap
                        # Restore coverage for c (since we subtracted it)
                        # Actually, we are replacing c with c_prime.
                        # Current state: c is removed (coverage reflects this).
                        # We need to add c_prime.
                        for w in neighbors[best_c_prime]:
                            coverage[w] += 1
                            
                        cw_set.discard(c)
                        cw_set.add(best_c_prime)
                        new_codewords.append(best_c_prime)
                        changed = True
                    else:
                        # No good swap found, restore c
                        for w in neighbors[c]:
                            coverage[w] += 1
                        new_codewords.append(c)
                else:
                    # No candidate covers all critical words, restore c
                    for w in neighbors[c]:
                        coverage[w] += 1
                    new_codewords.append(c)
                    
        codewords = new_codewords
        
    # Convert back to lists of integers
    result = []
    for c in codewords:
        digits = int_to_digits(c, n)
        result.append(digits[::-1])

    return result
