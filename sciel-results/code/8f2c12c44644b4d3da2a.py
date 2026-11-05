import random
import time

def construct(q, n, radius):
    # Only implemented for q=3, n=8, radius=1 as per task
    # But should be somewhat generic or at least handle the specific case robustly.
    # The task says "evaluator passes this task's alphabet size q, length n and covering radius".
    # So we must handle q=3, n=8, radius=1.
    
    if q != 3 or n != 8 or radius != 1:
        # Fallback to a generic simple construction if parameters differ
        # Just return a trivial cover or empty if not the target case
        # For now, assume the target case.
        pass
        
    total_words = q ** n
    # Precompute balls
    # Map word (tuple) to int
    # int_val = sum(w[i] * q**(n-1-i) for i in range(n))
    
    balls = [0] * total_words
    # Precompute powers
    powers = [1] * n
    for i in range(1, n):
        powers[i] = powers[i-1] * q
        
    # Function to convert tuple to int
    def to_int(t):
        val = 0
        for i in range(n):
            val = val * q + t[i]
        return val

    # Function to convert int to tuple
    def to_int_tuple(val):
        t = [0] * n
        for i in range(n-1, -1, -1):
            t[i] = val % q
            val //= q
        return t

    # Precompute balls as bitmasks
    for i in range(total_words):
        t = to_int_tuple(i)
        mask = 1 << i
        # Add neighbors
        for pos in range(n):
            for delta in [1, -1]:
                new_val = (t[pos] + delta) % q
                new_t = t[:]
                new_t[pos] = new_val
                new_i = to_int(new_t)
                mask |= (1 << new_i)
        balls[i] = mask

    all_mask = (1 << total_words) - 1
    
    def greedy_cover(randomize=False):
        covered = 0
        code = []
        # We need to find the best word to add at each step.
        # To speed up, we can just iterate all words.
        # Optimization: If randomize, we might shuffle the order of evaluation or tie-breaking.
        
        for _ in range(total_words): # Max iterations
            best_idx = -1
            best_count = 0
            # If randomizing, we might want to break ties randomly.
            # We can collect all indices with max count and pick one.
            # Or just pick the first one found with max count.
            
            # To make it faster, we can skip words that are already in the code?
            # No, a word in the code might still cover new words? No, its ball is already covered.
            # So we only need to consider words NOT in the code.
            # But checking "in code" is O(M).
            # Let's just iterate all 6561. It's fast.
            
            if randomize:
                # Shuffle indices? No, that's expensive.
                # Instead, if we find a count, we can keep a list of candidates with that count.
                pass
            
            for i in range(total_words):
                # If i is in code, its contribution is 0.
                # We can maintain a set of indices in code.
                # But for simplicity, let's just calculate.
                new_cover = balls[i] & ~covered
                count = new_cover.bit_count()
                if count > best_count:
                    best_count = count
                    best_idx = i
                    # If randomize, we might want to keep looking for other maxes?
                    # For now, let's just take the first max.
                    # To add randomness, we can randomize the order of i?
                    # That would require a list of range(total_words) and shuffle.
                    # 6561 log 6561 is fine.
            
            if best_idx == -1 or best_count == 0:
                break
                
            code.append(best_idx)
            covered |= balls[best_idx]
            
            if covered == all_mask:
                break
                
        # Redundancy removal
        # Maintain cover counts
        # This is O(M * 17)
        # We'll do a pass.
        
        # Actually, a simpler redundancy check:
        # For each c in code:
        #   temp_covered = covered
        #   # We can't easily subtract.
        #   # Let's use the count array approach.
        
        # Let's re-implement the greedy with counts for easier redundancy check?
        # Or just do the redundancy check at the end.
        
        # Redundancy Check:
        # cover_count = [0] * total_words
        # for c in code:
        #    for w in ball_indices[c]: cover_count[w] += 1
        # 
        # for c in code:
        #    redundant = True
        #    for w in ball_indices[c]:
        #        if cover_count[w] == 1:
        #             redundant = False
        #             break
        #    if redundant:
        #        for w in ball_indices[c]:
        #             cover_count[w] -= 1
        #        code.remove(c)
        #    else:
        #        # keep c
        
        # To do this, we need `ball_indices`. We have `balls` as bitmasks.
        # We can extract indices from bitmask.
        
        # Let's optimize the greedy loop by maintaining a list of active candidates?
        # For now, the simple O(N^2) greedy is likely fast enough for N=6561.
        
        return code

    # Run multiple restarts
    best_code = None
    start_time = time.time()
    max_time = 110 # Safety margin
    
    # First run deterministic
    code = greedy_cover(False)
    best_code = code
    
    # Then random restarts
    while time.time() - start_time < max_time:
        # Randomize the order of checking?
        # We can modify greedy_cover to accept a shuffled list of indices.
        # Let's create a helper that takes a permutation.
        
        # For simplicity, let's just add a small random delay or just run it.
        # The current greedy_cover is deterministic.
        # Let's add a parameter to greedy_cover to shuffle the candidate order.
        
        # I will rewrite greedy_cover to be more flexible.
        break # Placeholder, need to implement loop properly

    return [to_int_tuple(c) for c in best_code]
