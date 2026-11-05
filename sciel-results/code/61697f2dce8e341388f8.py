import itertools
import time
import random

def construct(q, n, radius):
    # This task is specifically for q=3, n=8, radius=1
    if q != 3 or n != 8 or radius != 1:
        # Fallback for other parameters, though the task contract implies fixed values
        # We'll just return a naive construction or empty if not matching
        return []
        
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # 1. Generate initial code: Direct product of tetracode and all tails
    # Tetracode: 9 words of length 4
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    
    initial_codewords = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            initial_codewords.append(h + list(tail))
            
    # Convert to tuples for hashing and easier manipulation
    initial_codewords = [tuple(c) for c in initial_codewords]
    
    # Map words to indices
    def word_to_idx(w):
        idx = 0
        for i, val in enumerate(reversed(w)):
            idx = idx * 3 + val
        return idx
        
    def idx_to_word(idx):
        w = [0] * n
        for i in range(n):
            w[i] = idx % 3
            idx //= 3
        return tuple(w)
        
    # Precompute spheres for all 6561 words
    # Sphere of a word w: w itself + all words at distance 1
    # There are 1 + n * (q-1) = 1 + 8 * 2 = 17 words in the sphere
    total_words = q ** n
    spheres = []
    for idx in range(total_words):
        w = idx_to_word(idx)
        sphere = [idx]
        for i in range(n):
            for val in range(q):
                if val == w[i]:
                    continue
                new_w = list(w)
                new_w[i] = val
                new_idx = word_to_idx(new_w)
                sphere.append(new_idx)
        spheres.append(sphere)
        
    # Initial coverage counts
    cover_count = [0] * total_words
    for c in initial_codewords:
        c_idx = word_to_idx(c)
        for w_idx in spheres[c_idx]:
            cover_count[w_idx] += 1
            
    best_code = initial_codewords[:]
    best_len = len(best_code)
    
    # Greedy removal loop
    # We will try multiple restarts with different random orders
    # Each restart:
    #   - Start with initial_codewords
    #   - Randomize order
    #   - Iteratively remove redundant codewords
    
    max_restarts = 20
    restarts_done = 0
    
    while restarts_done < max_restarts:
        if time.time() > deadline:
            break
            
        current_code = initial_codewords[:]
        random.shuffle(current_code)
        
        # We need to maintain cover_count for the current set
        # It's faster to recompute cover_count from scratch for each restart if the set is large,
        # but since we start from the same initial set, we can just copy the initial counts.
        cur_cover_count = cover_count[:]
        
        removed_any = True
        # To speed up, we can iterate through the list and try to remove.
        # If we remove one, we update counts.
        # We repeat until a full pass yields no removals.
        
        while removed_any:
            if time.time() > deadline:
                break
            removed_any = False
            # Iterate in reverse to allow in-place removal? 
            # Or just build a new list.
            # Since N is small (~500-700), building a new list is fine.
            
            # To avoid bias from fixed order, we shuffle again if we want? 
            # No, the initial shuffle is usually enough for one pass.
            # But removing one might make another redundant.
            # A single pass over the shuffled list might miss some.
            # We loop "while removed_any" to catch cascading redundancies.
            
            new_code = []
            for c in current_code:
                c_idx = word_to_idx(c)
                # Check if c is redundant
                can_remove = True
                for w_idx in spheres[c_idx]:
                    if cur_cover_count[w_idx] == 1:
                        can_remove = False
                        break
                
                if can_remove:
                    # Remove c
                    for w_idx in spheres[c_idx]:
                        cur_cover_count[w_idx] -= 1
                    removed_any = True
                else:
                    new_code.append(c)
            
            current_code = new_code
            
        if len(current_code) < best_len:
            best_len = len(current_code)
            best_code = current_code[:]
            
        restarts_done += 1
        
    return [list(c) for c in best_code]
