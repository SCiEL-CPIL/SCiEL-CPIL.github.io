import numpy as np
import time
import sys
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110  # 110 seconds to be safe
    
    # Fallback for non-specific cases
    if q != 3 or n != 8 or radius != 1:
        if n >= 4 and q == 3:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            import itertools
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            import itertools
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
    uncovered = set(range(total_words))
    
    # To speed up greedy, we can pick the word that covers the most uncovered words.
    # Since all spheres are size 17, initially all are equal.
    # We can just pick random words or sequential.
    # A better greedy:
    # While uncovered is not empty:
    #   Pick c that maximizes len(spheres[c] & uncovered)
    #   Add c to code
    #   Update cover_count and uncovered
    
    # Optimization: Instead of sets, use lists and boolean arrays.
    is_uncovered = [True] * total_words
    
    while any(is_uncovered):
        if time.time() > deadline:
            break
            
        best_c = -1
        max_covered = -1
        
        # Sample a subset of candidates to speed up? Or scan all?
        # Scanning 6561 words * 17 checks = ~110k ops.
        # If we do this 500 times, that's 55M ops. Might be too slow in Python.
        # Let's use a heuristic: Pick random candidates.
        
        # Actually, for the first few words, random is fine.
        # Let's just pick random words until we have a reasonable set, then refine?
        # No, greedy is better for a good start.
        
        # To make it fast, we can pre-compute a random order of words and pick the first one that covers an uncovered word?
        # Or just pick a random word that covers at least one uncovered word.
        
        # Let's try a simple random greedy:
        # Pick a random uncovered word u.
        # Pick a random word c from spheres_inv[u] (words that cover u).
        # Add c.
        
        # Precompute inverse spheres: inv_spheres[u] = list of c such that u in spheres[c]
        # Since spheres are symmetric (distance 1 is symmetric), inv_spheres[u] == spheres[u].
        
        # So:
        while any(is_uncovered):
             # Pick a random uncovered word
             # To avoid O(N) scan for random uncovered, we can maintain a list of uncovered indices
             # But updating it is expensive.
             
             # Let's just pick a random index and check if it's uncovered.
             # If not, try again.
             # Expected trials: 2 on average if half covered.
             
             u = random.randint(0, total_words - 1)
             if not is_uncovered[u]:
                 continue
                 
             # Pick a random codeword c that covers u
             # c must be in spheres[u]
             c = random.choice(spheres[u])
             
             # Add c to code
             code.append(c)
             
             # Update cover_count and is_uncovered
             for x in spheres[c]:
                 cover_count[x] += 1
                 if cover_count[x] == 1:
                     is_uncovered[x] = False
                     
             if time.time() > deadline:
                 break

    # 2. Local Search Optimization
    # Remove redundant codewords
    def is_redundant(c):
        for x in spheres[c]:
            if cover_count[x] <= 1:
                return False
        return True

    def remove_codeword(c):
        for x in spheres[c]:
            cover_count[x] -= 1
            if cover_count[x] == 0:
                is_uncovered[x] = True
        code.remove(c) # O(N) removal, might be slow if code is large.
                        # Use a set or swap with last element.

    # Let's use a list and swap-remove for O(1)
    # But we need to map index in list to codeword?
    # No, we just need the set of codewords.
    # Let's keep `code` as a list, but for removal, we can use a set for O(1) membership and removal,
    # and iterate over the set for redundancy check.
    
    code_set = set(code)
    
    # Re-verify cover_count is correct
    # It should be.
    
    best_size = len(code_set)
    best_code = list(code_set)
    
    while time.time() < deadline:
        # Local Search: Remove redundancies
        changed = True
        while changed:
            changed = False
            # Iterate over a copy of the set to allow modification
            for c in list(code_set):
                if is_redundant(c):
                    # Remove c
                    code_set.remove(c)
                    for x in spheres[c]:
                        cover_count[x] -= 1
                        if cover_count[x] == 0:
                            is_uncovered[x] = True
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
        
        # If we didn't reduce size, we are in a local minimum.
        # Perturbation:
        # 1. Pick a random codeword c to remove.
        # 2. Remove it.
        # 3. Find the set of uncovered words U.
        # 4. Find a word c' that covers the most words in U.
        # 5. Add c'.
        # 6. (Optional) If size increased, try to reduce back in next Local Search.
        
        if len(code_set) > 0:
            c = random.choice(list(code_set))
            
            # Remove c
            code_set.remove(c)
            for x in spheres[c]:
                cover_count[x] -= 1
                if cover_count[x] == 0:
                    is_uncovered[x] = True
            
            # Find uncovered words
            # We need to find c' that maximizes coverage of uncovered words.
            # Iterate over all possible c' (6561 words).
            # For each c', count how many x in spheres[c'] have is_uncovered[x] == True.
            
            best_c_prime = -1
            max_uncovered_covered = -1
            
            # To speed this up, we can sample a subset of c' if it's too slow.
            # 6561 * 17 = ~110k ops.
            # If we do this 1000 times, that's 110M ops.
            # In Python, this might take ~10-20s.
            # So we can do ~5-10 perturbations per second?
            # 110s / 10 = 1100 perturbations.
            # This seems feasible.
            
            for c_prime in range(total_words):
                count = 0
                for x in spheres[c_prime]:
                    if is_uncovered[x]:
                        count += 1
                if count > max_uncovered_covered:
                    max_uncovered_covered = count
                    best_c_prime = c_prime
            
            # Add best_c_prime
            code_set.add(best_c_prime)
            for x in spheres[best_c_prime]:
                cover_count[x] += 1
                if cover_count[x] == 1:
                    is_uncovered[x] = False

    # Return the best code found
    # Convert indices to words
    result = [index_to_word(i) for i in best_code]
    return result
