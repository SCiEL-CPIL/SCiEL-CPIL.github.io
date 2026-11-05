import time
import random
import sys

def construct(q, n, radius):
    # Parameters for the specific task: q=3, n=6, radius=1
    # The evaluator passes these, so we use them.
    if q != 3 or n != 6 or radius != 1:
        # Fallback for other parameters if ever called (though task is fixed)
        # Simple greedy fallback
        pass
    
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Total number of words in the space
    total_words = q ** n
    
    # Precompute the Hamming balls for each word
    # balls[i] is a bitset (integer) where bit j is 1 if word j is within radius of word i
    balls = [0] * total_words
    
    # Helper to convert integer index to word tuple
    def idx_to_word(idx):
        word = []
        for _ in range(n):
            word.append(idx % q)
            idx //= q
        return word
    
    # Helper to convert word tuple to integer index
    def word_to_idx(word):
        idx = 0
        for i in range(n):
            idx = idx * q + word[i]
        return idx

    # Precompute neighbors for each word
    # Radius 1 means the word itself plus words differing in exactly 1 position
    for i in range(total_words):
        w = idx_to_word(i)
        mask = 1 << i
        balls[i] = mask
        for pos in range(n):
            original_val = w[pos]
            for new_val in range(q):
                if new_val == original_val:
                    continue
                w[pos] = new_val
                j = word_to_idx(w)
                mask |= (1 << j)
            w[pos] = original_val
        balls[i] = mask

    # Greedy construction function
    def greedy_randomized(seed):
        random.seed(seed)
        uncovered = (1 << total_words) - 1
        selected = []
        
        while uncovered:
            best_cov = -1
            best_candidates = []
            
            # Find the codeword that covers the most uncovered words
            for c in range(total_words):
                # Optimization: if c is already selected, skip? 
                # Actually, re-selecting the same codeword is useless. 
                # We can maintain a set of selected or just check.
                # Since N is small, we can just iterate all.
                # But we should not select the same one twice.
                # Let's keep a boolean array or set.
                pass
            
            # Better: iterate through all, but ignore those already in 'selected'
            # To make it faster, we can just pick from all, and if it's already selected, its contribution to 'uncovered' is 0.
            # So we can just iterate all 729.
            
            for c in range(total_words):
                cov = (balls[c] & uncovered).bit_count()
                if cov > best_cov:
                    best_cov = cov
                    best_candidates = [c]
                elif cov == best_cov:
                    best_candidates.append(c)
            
            if best_cov == 0:
                # Should not happen if space is connected
                break
                
            # Pick randomly from best candidates to introduce diversity
            chosen = random.choice(best_candidates)
            selected.append(chosen)
            uncovered &= ~balls[chosen]
            
        return selected

    best_solution = None
    best_len = float('inf')
    run_count = 0
    
    while time.time() < deadline:
        seed = run_count
        sol = greedy_randomized(seed)
        run_count += 1
        
        if len(sol) < best_len:
            best_len = len(sol)
            best_solution = sol
            
        # If we found a very good solution (e.g., 73), we might still search for 72 or 71
        # But 73 is the known best. 71 is lower bound.
        # We can stop early if we hit the theoretical lower bound? 
        # Lower bound is 71. If we find 71, we can stop.
        if best_len <= 71:
            break

    # Convert indices back to words
    if best_solution is None:
        # Fallback: return a valid (though maybe not minimal) code
        # The seed program's logic
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        import itertools
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    return [idx_to_word(idx) for idx in best_solution]
