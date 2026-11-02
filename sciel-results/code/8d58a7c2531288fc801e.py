import time
import random
import sys

# Precompute balls for q=3, n=7, radius=1
# The task specifies q, n, radius are passed to construct.
# However, for performance, we should precompute if possible, or cache.
# Since the evaluator might call construct multiple times or with different params?
# "the evaluator passes this task's alphabet size q, length n and covering radius"
# It implies the signature is construct(q, n, radius).
# We should handle general q, n, radius if possible, but optimize for 3, 7, 1.
# Given the specific task context (K_3(7,1)), we can specialize if needed, 
# but it's safer to write a general solver that is fast enough for 3,7,1.

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # Total words
    total_words = q ** n
    
    # If the space is small, we can brute force or use simple greedy.
    # For q=3, n=7, total_words=2187.
    
    # 1. Precompute Hamming balls of radius 'radius' for each word.
    # For q=3, n=7, radius=1, ball size is 1 + n*(q-1) = 1 + 7*2 = 15.
    # We represent balls as lists of indices for fast counting/updates,
    # and as bitmasks for fast coverage checks in greedy.
    
    # Generating neighbors
    # word is a tuple of n digits in 0..q-1
    # We map word to int index: sum(d_i * q^(n-1-i))
    
    def word_to_idx(word):
        idx = 0
        for d in word:
            idx = idx * q + d
        return idx

    def idx_to_word(idx):
        word = [0] * n
        for i in range(n-1, -1, -1):
            word[i] = idx % q
            idx //= q
        return word

    # Precompute neighbors for all words
    # This is O(total_words * n * q)
    balls = []
    ball_masks = []
    
    # For q=3, n=7, this is 2187 * 7 * 3 ~ 45k operations. Fast.
    for idx in range(total_words):
        w = idx_to_word(idx)
        neighbors = [idx]
        # Add distance 1 neighbors
        for pos in range(n):
            original = w[pos]
            for new_val in range(q):
                if new_val == original:
                    continue
                w[pos] = new_val
                n_idx = word_to_idx(w)
                neighbors.append(n_idx)
            w[pos] = original
        
        balls.append(neighbors)
        
        # Build bitmask
        mask = 0
        for nb in neighbors:
            mask |= (1 << nb)
        ball_masks.append(mask)
        
    all_mask = (1 << total_words) - 1
    
    best_solution = None
    best_size = float('inf')
    
    # Greedy Set Cover with Reduction
    # We will run multiple iterations with random tie-breaking to escape local minima.
    
    def greedy_cover(seed):
        random.seed(seed)
        covered_mask = 0
        code = []
        # coverage_counts for reduction step
        # We can maintain this incrementally
        
        while covered_mask != all_mask:
            best_gain = -1
            best_candidates = []
            
            # Iterate all candidates
            # Optimization: If we have many candidates, we might want to prune,
            # but 2187 is small.
            for c in range(total_words):
                # Calculate newly covered words
                # gain = bit_count(ball_masks[c] & ~covered_mask)
                gain = (ball_masks[c] & ~covered_mask).bit_count()
                
                if gain > best_gain:
                    best_gain = gain
                    best_candidates = [c]
                elif gain == best_gain:
                    best_candidates.append(c)
            
            if not best_candidates:
                break # Should not happen if all_mask is reachable
                
            # Pick one randomly from best candidates to vary the path
            chosen = random.choice(best_candidates)
            
            code.append(chosen)
            covered_mask |= ball_masks[chosen]
            
            # Time check
            if time.time() > deadline:
                return None
        
        return code

    def reduce_cover(code):
        """
        Remove redundant codewords from the cover.
        A codeword c is redundant if every word in its ball is covered by at least 
        one other codeword in the set.
        """
        if not code:
            return code
            
        # Count coverage for each word
        # cover_count[w] = number of codewords in code whose ball contains w
        cover_count = [0] * total_words
        
        for c in code:
            for w in balls[c]:
                cover_count[w] += 1
        
        changed = True
        while changed:
            changed = False
            # Try to remove each codeword
            # Iterate in reverse to allow index removal
            for i in range(len(code) - 1, -1, -1):
                c = code[i]
                # Check if c is redundant
                is_redundant = True
                for w in balls[c]:
                    if cover_count[w] <= 1:
                        is_redundant = False
                        break
                
                if is_redundant:
                    # Remove c
                    for w in balls[c]:
                        cover_count[w] -= 1
                    code.pop(i)
                    changed = True
        
        return code

    # Main loop: Try different seeds
    # The greedy is deterministic if we don't randomize ties.
    # Randomizing ties allows finding different covers.
    
    # First, try a deterministic greedy (seed 0)
    initial_code = greedy_cover(0)
    if initial_code:
        reduced_code = reduce_cover(initial_code)
        best_solution = reduced_code
        best_size = len(reduced_code)
    
    # Now try random seeds
    seed = 1
    while time.time() < deadline:
        code = greedy_cover(seed)
        if code is None:
            break
        reduced_code = reduce_cover(code)
        if len(reduced_code) < best_size:
            best_solution = reduced_code
            best_size = len(reduced_code)
            # If we hit the known lower bound or best known, we can stop early?
            # Lower bound is 156, Best known 186.
            if best_size <= 186:
                # We might still find 185, 184...
                # But 186 is a strong target.
                pass
        seed += 1
        
    if best_solution is None:
        # Fallback to seed program logic if greedy failed (shouldn't happen)
        # Direct sum of tetracode
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        import itertools
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    # Convert indices back to lists of integers
    result = []
    for idx in best_solution:
        result.append(idx_to_word(idx))
        
    return result
