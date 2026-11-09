import time
import random
import numpy as np

def construct(q, n, radius):
    # Parameters for the specific task: q=3, n=6, radius=1
    # The evaluator passes these, but we assume the task contract values.
    # However, we should handle general q, n if possible, but the task is specific.
    # The prompt says: "The evaluator passes this task's alphabet size q, length n and covering radius."
    # So we must use the arguments.
    
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    # Total number of words in the space
    total_words = q ** n
    
    # If the space is small enough, we can use bitsets.
    # For q=3, n=6, total_words = 729.
    # Python integers can handle this easily.
    
    # Helper to convert a word (list of ints) to an index
    def word_to_idx(word):
        idx = 0
        base = 1
        # word is a list of length n, values 0..q-1
        # Let's use base-q representation
        for i in range(n):
            idx += word[i] * base
            base *= q
        return idx

    def idx_to_word(idx):
        word = [0] * n
        for i in range(n - 1, -1, -1):
            word[i] = idx % q
            idx //= q
        return word

    # Precompute balls for all words
    # ball[i] is an integer with bits set for all words within distance 'radius' of word i
    balls = [0] * total_words
    
    # Generate all words
    all_words = [idx_to_word(i) for i in range(total_words)]
    
    # Compute Hamming distance
    def hamming_dist(w1, w2):
        return sum(1 for a, b in zip(w1, w2) if a != b)
    
    # Precompute balls
    for i in range(total_words):
        wi = all_words[i]
        mask = 0
        for j in range(total_words):
            if hamming_dist(wi, all_words[j]) <= radius:
                mask |= (1 << j)
        balls[i] = mask
        
    # Initial uncovered mask: all bits set
    full_mask = (1 << total_words) - 1
    
    best_code = []
    best_size = total_words + 1
    
    # Fallback: A simple valid code (all words)
    # But we want to minimize.
    
    # Strategy: Randomized Greedy Set Cover
    # We will run multiple iterations.
    
    # To make it faster, we can pre-sort candidates? No, coverage changes.
    
    for restart in range(10000):
        if time.time() > deadline:
            break
            
        # Randomize candidate order to explore different greedy paths
        candidates = list(range(total_words))
        random.shuffle(candidates)
        
        uncovered = full_mask
        current_code = []
        
        while uncovered:
            # Find candidate with max new coverage
            best_idx = -1
            best_gain = 0
            
            # Iterate through candidates
            # Optimization: If we have a good estimate, we can break early? No, need max.
            for c in candidates:
                # Calculate newly covered words
                new_covered = balls[c] & uncovered
                gain = new_covered.bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = c
                    if gain == 0:
                        # No more coverage possible from this candidate, but maybe others?
                        # If best_gain is 0, we are stuck, but uncovered is not 0?
                        # This shouldn't happen if balls are correct.
                        pass
            
            if best_idx == -1:
                # Should not happen if balls cover the space
                break
                
            current_code.append(best_idx)
            uncovered &= ~balls[best_idx]
            
        if len(current_code) < best_size:
            best_size = len(current_code)
            best_code = current_code.copy()
            
            # If we found the theoretical lower bound or a very good code, we can stop early?
            # Lower bound is 71. If we find 71, 72, or 73, it's excellent.
            if best_size <= 73:
                # We can continue to try to find 71 or 72, but 73 is known best.
                # Let's run a few more to see if we can beat 73.
                pass

    # Convert best_code indices back to words
    if not best_code:
        # Fallback to the seed program's logic if greedy failed or timed out immediately
        # The seed program logic:
        if q == 3 and n == 6:
             tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
             import itertools
             return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
             # General fallback: all words (valid but huge)
             return all_words

    return [idx_to_word(idx) for idx in best_code]
