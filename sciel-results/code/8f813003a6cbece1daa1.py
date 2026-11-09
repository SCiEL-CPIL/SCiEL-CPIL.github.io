import time
import numpy as np
import itertools
import random

def construct(q, n, radius):
    # Task: q=3, n=7, radius=1
    # Total words: 3^7 = 2187
    # Ball size: 1 + 7*2 = 15
    
    t_start = time.time()
    deadline = 110.0 # Safety margin
    
    # Precompute all words as indices 0..2186
    # Map (d1, d2, ..., d7) to index
    # index = d1*3^6 + d2*3^5 + ... + d7*3^0
    
    # Generate all words
    all_words = list(itertools.product(range(q), repeat=n))
    N = len(all_words)
    
    # Precompute balls: balls[i] is a list of indices of words within distance radius of word i
    # For radius 1, this is the word itself + all words with 1 digit changed
    # We can precompute the neighbor offsets
    # But since n is small (7) and q is small (3), we can just compute it directly or use a lookup
    
    # Faster: Precompute the ball for each index
    balls = np.empty((N, 1 + n*(q-1)), dtype=np.int32)
    
    # Helper to get index from tuple
    def tuple_to_idx(t):
        idx = 0
        for i, val in enumerate(t):
            idx = idx * q + val
        return idx

    # Precompute balls
    for i, w in enumerate(all_words):
        ball = [i]
        w_list = list(w)
        for pos in range(n):
            old_val = w_list[pos]
            for new_val in range(q):
                if new_val != old_val:
                    w_list[pos] = new_val
                    ball.append(tuple_to_idx(tuple(w_list)))
            w_list[pos] = old_val
        balls[i] = ball
        
    # Shuffle balls to break ties randomly? No, we will randomize selection among max gains.
    
    best_code_len = N
    best_code = None
    
    # Run multiple restarts
    while time.time() - t_start < deadline:
        # Greedy Set Cover with Random Tie-Breaking
        covered = np.zeros(N, dtype=bool)
        counts = np.zeros(N, dtype=np.int32)
        code = []
        
        while not covered.all():
            # Calculate gain for all candidates
            # gain[i] = number of uncovered words in ball[i]
            # ~covered[balls] creates a boolean matrix (N, 15)
            # summing along axis 1 gives (N,)
            gains = np.sum(~covered[balls], axis=1)
            
            max_gain = np.max(gains)
            if max_gain == 0:
                # Should not happen if covered.all() is false
                break
                
            # Find all candidates with max gain
            candidates = np.where(gains == max_gain)[0]
            
            # Pick one randomly
            idx = candidates[random.randint(0, len(candidates) - 1)]
            
            code.append(idx)
            
            # Update covered and counts
            ball = balls[idx]
            covered[ball] = True
            counts[ball] += 1
            
        # Local Search: Redundancy Removal
        # Repeat until no more redundant codewords
        changed = True
        while changed:
            changed = False
            if not code:
                break
                
            code_arr = np.array(code, dtype=np.int32)
            # For each codeword in code, check if min count in its ball is > 1
            # If so, it's redundant
            code_balls = balls[code_arr] # shape (K, 15)
            mins = np.min(counts[code_balls], axis=1) # shape (K)
            redundant_mask = mins > 1
            
            if np.any(redundant_mask):
                # Remove all redundant ones in this pass
                redundant_indices = code_arr[redundant_mask]
                
                # Update counts: decrement for all words in their balls
                # We need to subtract 1 from counts for each word in each redundant ball
                # This can be done with np.add.at or simple loop since K is small
                for idx in redundant_indices:
                    counts[balls[idx]] -= 1
                    
                # Remove from code list
                # Create a set for fast lookup
                redundant_set = set(redundant_indices.tolist())
                code = [c for c in code if c not in redundant_set]
                
                changed = True
                
        # Check if this is better
        if len(code) < best_code_len:
            best_code_len = len(code)
            best_code = code.copy()
            
        # Optional: If we found a very good code, we can try to improve it further
        # but the restart loop handles exploration.
        
    # Convert indices back to list of lists of integers
    if best_code is None:
        # Fallback: return all words (should not happen)
        best_code = list(range(N))
        
    result = []
    for idx in best_code:
        # Convert index back to tuple of digits
        w = []
        temp_idx = idx
        for _ in range(n):
            w.append(temp_idx % q)
            temp_idx //= q
        w.reverse()
        result.append(w)
        
    return result
