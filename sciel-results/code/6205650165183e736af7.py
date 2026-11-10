import numpy as np
import time
import sys
import random
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110  # 110 seconds to be safe
    
    # Fallback for non-specific cases
    if q != 3 or n != 8 or radius != 1:
        if n >= 4 and q == 3:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            return [list(w) for w in itertools.product(range(q), repeat=n)]

    total_words = 3 ** n  # 6561
    
    # Precompute spheres (neighbors within distance 1) for all words
    # spheres[i] is a list of word indices covered by codeword i
    spheres = [[] for _ in range(total_words)]
    
    # Helper to convert index to word
    def index_to_word(idx):
        w = [0] * n
        for i in range(n):
            w[i] = idx % 3
            idx //= 3
        return w
        
    # Helper to convert word to index
    def word_to_index(w):
        idx = 0
        for i in range(n):
            idx += w[i] * (3 ** i)
        return idx

    # Precompute neighbors
    for i in range(total_words):
        w = index_to_word(i)
        spheres[i].append(i)
        for pos in range(n):
            orig_val = w[pos]
            for delta in [1, 2]:
                new_val = (orig_val + delta) % 3
                w[pos] = new_val
                j = word_to_index(w)
                spheres[i].append(j)
            w[pos] = orig_val

    # State:
    # code: list of codeword indices
    # cover_count: array of size total_words, cover_count[i] = number of codewords covering i
    
    # 1. Greedy Initialization
    # Start with empty code
    code = []
    cover_count = [0] * total_words
    is_uncovered = [True] * total_words
    
    # Use a faster greedy: pick the word that covers the most uncovered words
    # To speed this up, we can maintain a list of uncovered words and sample from it,
    # or just scan all candidates but break early if we find a perfect cover (size 17).
    # Since the space is small (6561), scanning all is ~110k ops.
    # 400 iterations * 110k = 44M ops. Should be okay in 110s.
    
    while any(is_uncovered):
        if time.time() > deadline:
            break
            
        best_c = -1
        max_covered = -1
        
        # Scan all candidates
        for c in range(total_words):
            # Count uncovered neighbors
            count = 0
            for x in spheres[c]:
                if is_uncovered[x]:
                    count += 1
            
            if count > max_covered:
                max_covered = count
                best_c = c
                if count == 17: # Perfect cover
                    break
        
        if best_c == -1:
            break
            
        code.append(best_c)
        
        # Update cover_count and is_uncovered
        for x in spheres[best_c]:
            cover_count[x] += 1
            if cover_count[x] == 1:
                is_uncovered[x] = False

    # 2. Local Search Optimization
    code_set = set(code)
    
    # Re-verify cover_count is correct? It should be.
    
    best_size = len(code_set)
    best_code = list(code_set)
    
    # Helper to check redundancy
    def is_redundant(c):
        for x in spheres[c]:
            if cover_count[x] <= 1:
                return False
        return True

    # Helper to remove codeword
    def remove_codeword(c):
        code_set.remove(c)
        for x in spheres[c]:
            cover_count[x] -= 1
            if cover_count[x] == 0:
                is_uncovered[x] = True

    # Helper to add codeword
    def add_codeword(c):
        code_set.add(c)
        for x in spheres[c]:
            cover_count[x] += 1
            if cover_count[x] == 1:
                is_uncovered[x] = False

    # Local Search Loop
    while time.time() < deadline:
        # Phase 1: Remove redundancies
        changed = True
        while changed:
            changed = False
            for c in list(code_set):
                if is_redundant(c):
                    remove_codeword(c)
                    changed = True
                    if len(code_set) < best_size:
                        best_size = len(code_set)
                        best_code = list(code_set)
                    if time.time() > deadline:
                        break
            if time.time() > deadline:
                break
        
        if time.time() > deadline:
            break
        
        # Phase 2: Perturbation (1-swap)
        # Pick a random codeword to remove
        # Try to replace it with another codeword that covers the most newly uncovered words
        # Or just remove and re-greedy fill?
        # Let's try: Remove c, find best c' to add, then check if size decreased or same?
        # Actually, standard approach:
        # 1. Remove c
        # 2. Find set of uncovered U
        # 3. Find c' that maximizes coverage of U
        # 4. Add c'
        # 5. If len(code) <= best_size, keep. Else revert?
        # Since we track best_size, we can just do the swap and let the next iteration's redundancy check clean up.
        
        if len(code_set) > 0:
            c = random.choice(list(code_set))
            
            # Save state for potential revert? 
            # Or just do the swap and if it gets worse, the next redundancy phase might not fix it immediately
            # if the new word is redundant? No, if it's redundant we remove it.
            # If the swap increases size, we might be stuck in a local minimum.
            # Let's just do the swap.
            
            remove_codeword(c)
            
            # Find best c'
            best_c_prime = -1
            max_uncovered_covered = -1
            
            # Optimization: Only consider words that cover at least one uncovered word
            # But we need to find the one covering the MOST.
            # Scan all candidates.
            for c_prime in range(total_words):
                if c_prime == c:
                    continue
                count = 0
                for x in spheres[c_prime]:
                    if is_uncovered[x]:
                        count += 1
                if count > max_uncovered_covered:
                    max_uncovered_covered = count
                    best_c_prime = c_prime
                    if count == 17:
                        break
            
            if best_c_prime != -1:
                add_codeword(best_c_prime)
            else:
                # Revert
                add_codeword(c)

    # Return the best code found
    result = [index_to_word(i) for i in best_code]
    return result
