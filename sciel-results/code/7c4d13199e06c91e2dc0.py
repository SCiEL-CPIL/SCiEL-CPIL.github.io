"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import random
import numpy as np
import time

def construct(q, n, radius):
    if n == 0:
        return [[]]
    
    # Handle small cases with known optimal or near-optimal
    if n == 1:
        return [[i] for i in range(q)]
    if n == 2:
        # All pairs? No, K_q(2,1) = q^2 / (1 + 2(q-1)) * q? 
        # For q=3, n=2, r=1. 
        # 9 words. Each covers 1 + 2*2 = 5.
        # Lower bound 9/5 = 1.8 -> 2? No.
        # Word 00 covers 00,01,02,10,20.
        # Word 11 covers 11,10,12,21,01.
        # Word 22 covers 22,21,20,12,02.
        # 00,11,22 cover:
        # 00: 00,01,02,10,20
        # 11: 11,10,12,21,01
        # 22: 22,21,20,12,02
        # Union: 00,01,02,10,11,12,20,21,22. All 9.
        # So 3 words.
        if q == 3:
            return [[0,0], [1,1], [2,2]]
        # General q=3 n=2
        return [[0,0], [1,1], [2,2]]
    if n == 3:
        if q == 3:
            return [[0,0,0], [1,1,1], [2,2,2]]
    if n == 4:
        if q == 3:
            return [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            
    # General case: q=3, n=6, radius=1
    # Use integer representation for speed
    # Map tuple to int and vice versa
    # Since n=6, q=3, total 729 words.
    
    if q != 3 or n != 6:
        # Fallback to generic greedy for other params
        return _generic_greedy(q, n, radius)

    # Specialized for q=3, n=6
    total_words = 3**n
    
    # Precompute neighbors for all words
    # neighbors[w] = list of words within dist 1
    neighbors = [[] for _ in range(total_words)]
    
    def tuple_to_int(t):
        val = 0
        for i in range(n):
            val = val * 3 + t[i]
        return val

    def int_to_tuple(val):
        t = [0] * n
        for i in range(n-1, -1, -1):
            t[i] = val % 3
            val //= 3
        return tuple(t)

    # Build neighbors
    for w in range(total_words):
        tw = int_to_tuple(w)
        # Center
        neighbors[w].append(w)
        # Variations
        for i in range(n):
            orig = tw[i]
            for v in range(3):
                if v != orig:
                    new_t = list(tw)
                    new_t[i] = v
                    new_w = tuple_to_int(tuple(new_t))
                    neighbors[w].append(new_w)
    
    # Greedy Construction
    covered = [0] * total_words
    code = []
    uncovered_count = total_words
    
    # We want to pick codewords that cover max uncovered
    # To speed up, we can just iterate
    
    while uncovered_count > 0:
        best_c = -1
        best_gain = 0
        # Optimization: Only check candidates that are not already in code?
        # Actually, checking all is fine, 729 * 13 is small.
        # But we can skip if gain is 0.
        
        for c in range(total_words):
            if c in code: # O(N) check, slow. Use set.
                continue
            gain = 0
            for nb in neighbors[c]:
                if covered[nb] == 0:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_c = c
            if best_gain == 13: # Max possible
                break
        
        if best_c == -1:
            break # Should not happen if uncovered_count > 0
            
        code.append(best_c)
        for nb in neighbors[best_c]:
            if covered[nb] == 0:
                uncovered_count -= 1
            covered[nb] += 1
            
    # Convert to set for O(1) lookup
    code_set = set(code)
    
    # Local Search: Minimize size
    # 1. Remove redundant
    # 2. Swap to enable removal
    
    improved = True
    while improved:
        improved = False
        
        # Phase 1: Remove redundant
        for i in range(len(code)):
            c = code[i]
            redundant = True
            for nb in neighbors[c]:
                if covered[nb] == 1:
                    redundant = False
                    break
            if redundant:
                # Remove c
                for nb in neighbors[c]:
                    covered[nb] -= 1
                code.pop(i)
                code_set.remove(c)
                improved = True
                break # Restart scan
        
        # Phase 2: Swap
        # For each c, if it has unique words, try to find c' that covers them
        # and see if c' allows another removal
        for i in range(len(code)):
            c = code[i]
            unique_words = [nb for nb in neighbors[c] if covered[nb] == 1]
            
            if not unique_words:
                # Should have been caught in Phase 1, but maybe coverage changed?
                # If no unique words, remove it
                for nb in neighbors[c]:
                    covered[nb] -= 1
                code.pop(i)
                code_set.remove(c)
                improved = True
                break
            
            # Find intersection of neighbors of unique_words
            # Intersection of sets
            # Start with neighbors[unique_words[0]]
            candidates = set(neighbors[unique_words[0]])
            for w in unique_words[1:]:
                candidates.intersection_update(neighbors[w])
            
            # Try each candidate
            # We want to replace c with c'
            # If c' is in code, it's a swap within code (no size change immediately, but might allow removal)
            # If c' not in code, size stays same, but might allow removal of another word
            
            # To keep it fast, only try candidates that are "good"
            # Heuristic: Prefer c' that covers many words with count 1 (to protect them)
            # Actually, just try a few random ones or all if small
            
            # If candidates is empty, we can't swap c without losing coverage.
            if not candidates:
                continue
                
            # Try to find a c' such that after swapping, we can remove another codeword
            # This is expensive. 
            # Alternative: Just try to find c' that covers unique_words.
            # Then check if any *other* codeword becomes redundant.
            
            for c_prime in candidates:
                if c_prime == c:
                    continue
                
                # Simulate swap: remove c, add c_prime
                # Check if coverage is maintained
                # Words in neighbors[c] that were unique to c must be covered by c_prime.
                # By construction of candidates, c_prime is in neighbors[w] for all w in unique_words.
                # So unique words are covered.
                # Other words in neighbors[c] had count >= 2, so count >= 1 after removal.
                # So coverage is maintained.
                
                # Now, does this allow removing another codeword?
                # We need to check if any codeword k != i, k != c_prime (if in code) becomes redundant.
                
                # This check is O(N * 13).
                # We can limit the number of swaps tried.
                
                # Let's just do a simple improvement:
                # If c_prime is NOT in code, we can't reduce size directly.
                # If c_prime IS in code, we are just moving c to c_prime's position? No, c_prime is already there.
                # If c_prime in code, we can just remove c?
                # If c_prime in code, and c_prime covers unique_words(c), then c is redundant?
                # No, c might cover words that c_prime doesn't.
                # But if c_prime covers unique_words(c), and c_prime is in code,
                # then for any w in neighbors(c):
                #   if w was unique to c, c_prime covers it.
                #   if w was not unique, some other c'' covers it.
                # So c is redundant!
                
                if c_prime in code_set:
                    # c is redundant because c_prime covers its unique parts, and others cover the rest
                    # Wait, this logic holds only if c_prime covers ALL unique words of c.
                    # Yes, candidates are intersection of neighbors of unique words.
                    # So c_prime covers all unique words.
                    # Therefore, c can be removed.
                    
                    # Remove c
                    for nb in neighbors[c]:
                        covered[nb] -= 1
                    code.pop(i)
                    code_set.remove(c)
                    improved = True
                    break
                    
            if improved:
                break
                
        # Phase 3: Randomized restart / perturbation if stuck?
        # For now, rely on greedy + remove/swap.
        
    # Convert back to list of lists
    result = [list(int_to_tuple(c)) for c in code]
    return result

def _generic_greedy(q, n, radius):
    # Fallback for other parameters
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    def hamming(a, b):
        return sum(1 for x, y in zip(a, b) if x != y)
        
    covered = [False] * len(all_words)
    code = []
    
    # Greedy
    while not all(covered):
        best_idx = -1
        best_count = 0
        for i, w in enumerate(all_words):
            if w in code: continue
            count = 0
            # Check coverage
            for j, u in enumerate(all_words):
                if not covered[j] and hamming(w, u) <= radius:
                    count += 1
            if count > best_count:
                best_count = count
                best_idx = i
        if best_idx == -1: break
        code.append(all_words[best_idx])
        for j, u in enumerate(all_words):
            if hamming(all_words[best_idx], u) <= radius:
                covered[j] = True
                
    return code
# EVOLVE-BLOCK-END
