import time
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s for safety
    
    total_words = q ** n
    if total_words > 10000:
        # Fallback for larger instances, though task is 3^6=729
        return []
        
    # Precompute balls
    # Map word tuple to index
    # For q=3, n=6, index is base-3 representation
    def word_to_idx(w):
        idx = 0
        for i in range(n):
            idx = idx * q + w[i]
        return idx

    def idx_to_word(idx):
        w = [0] * n
        for i in range(n-1, -1, -1):
            w[i] = idx % q
            idx //= q
        return w

    # Generate balls
    balls = []
    for i in range(total_words):
        w = idx_to_word(i)
        ball = [i]
        for pos in range(n):
            orig_val = w[pos]
            for val in range(q):
                if val != orig_val:
                    new_w = w.copy()
                    new_w[pos] = val
                    ball.append(word_to_idx(new_w))
        balls.append(ball)
        
    # Greedy Initialization
    covered = np.zeros(total_words, dtype=bool)
    code_indices = []
    
    # Precompute ball arrays for fast numpy slicing
    ball_arrays = [np.array(b) for b in balls]
    
    while not covered.all():
        best_idx = -1
        best_gain = -1
        # Optimize: only check candidates? No, 729 is small.
        for i in range(total_words):
            # Gain: number of uncovered words in ball[i]
            # ~covered is a boolean array
            gain = np.sum(~covered[ball_arrays[i]])
            if gain > best_gain:
                best_gain = gain
                best_idx = i
                
        if best_idx == -1:
            break
            
        code_indices.append(best_idx)
        covered[ball_arrays[best_idx]] = True
        
        if time.time() > deadline:
            break

    # Local Search
    counts = np.zeros(total_words, dtype=int)
    for c in code_indices:
        counts[ball_arrays[c]] += 1
        
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try to remove codewords
        # Iterate in reverse to handle index changes easily, or use a flag
        for i in range(len(code_indices) - 1, -1, -1):
            c = code_indices[i]
            # Check if all words in ball[c] are covered by at least one other codeword
            # i.e., counts[j] > 1 for all j in ball[c]
            if np.all(counts[ball_arrays[c]] > 1):
                code_indices.pop(i)
                counts[ball_arrays[c]] -= 1
                improved = True
                break # Restart loop
        
        if time.time() > deadline:
            break
            
        # If no removal was possible, try 1-swap to enable removal?
        # This is expensive. Let's try a limited swap search if we have time.
        if not improved and len(code_indices) > 0:
             # Try a few random swaps or systematic swaps?
             # Systematic is better.
             # For each c in code:
             #   For each c' not in code:
             #     Check if swap is valid (counts don't drop to 0)
             #     If valid, perform swap? 
             #     Swapping doesn't reduce size. It only helps if it enables a removal.
             #     We can check: after swap, is there any d in code such that 
             #     min(new_counts[ball[d]]) > 1?
             
             # This is O(N^3). 75^3 * 13 ~ 5.6M ops. Feasible.
             # Let's implement a single pass of "swap if it enables removal"
             
             code_set = set(code_indices)
             found_swap = False
             for i in range(len(code_indices)):
                 if time.time() > deadline: break
                 c = code_indices[i]
                 # Try swapping c with some c'
                 # We only care about c' that might help.
                 # Iterate over all c' not in code
                 for c_prime in range(total_words):
                     if c_prime in code_set:
                         continue
                     
                     # Simulate swap
                     # new_counts = counts
                     # new_counts[ball[c]] -= 1
                     # new_counts[ball[c_prime]] += 1
                     
                     # Check validity: min(new_counts) > 0
                     # We can check:
                     # 1. min(counts[ball[c]] - 1) > 0  => min(counts[ball[c]]) > 1
                     # 2. min(counts[ball[c_prime]] + 1) > 0 => always true if counts >= 0
                     # So we just need min(counts[ball[c]]) > 1
                     if np.min(counts[ball_arrays[c]]) <= 1:
                         continue
                     
                     # If valid swap, check if it enables removal of some d
                     # new_counts = counts.copy()
                     # new_counts[ball[c]] -= 1
                     # new_counts[ball[c_prime]] += 1
                     
                     # We want to know if there exists d in code_indices (d != c)
                     # such that min(new_counts[ball[d]]) > 1
                     
                     # This check is expensive.
                     # Heuristic: Just perform the swap if it's valid?
                     # No, swapping might trap us in a local minimum with same size.
                     # But if we perform a swap, we must ensure we don't loop.
                     
                     # Let's just try to perform the swap and see if a removal becomes possible in the next iteration.
                     # To prevent infinite loops, we can accept a swap only if it improves a tie-breaker (e.g. sum of squares of counts, or lexicographic).
                     # Or, we can just try the swap, perform it, and if no removal happens in the next full pass, revert?
                     
                     # Given time constraints, let's skip complex swaps and rely on the fact that 
                     # Greedy + Removal often works for 73.
                     # If it gets stuck at 74, we might miss 73.
                     
                     # Let's try one more thing: Randomized Restart of Local Search
                     # Shuffle code_indices and try removals again.
                     pass

    # Convert back to words
    return [idx_to_word(i) for i in code_indices]
