import numpy as np
import time
from itertools import combinations
import random

def construct():
    N = 17
    MIN_DIST = 6
    start_time = time.time()
    time_limit = 58  # Leave margin
    
    # Precompute masks
    masks_by_dist = [[] for _ in range(MIN_DIST)]
    for d in range(MIN_DIST):
        for combo in combinations(range(N), d):
            mask = 0
            for i in combo:
                mask |= (1 << i)
            masks_by_dist[d].append(mask)
    
    all_masks = []
    for d in range(MIN_DIST):
        all_masks.extend(masks_by_dist[d])
        
    # blocked_count[i] is the number of codewords within distance < MIN_DIST of i
    blocked_count = np.zeros(1 << N, dtype=np.int32)
    
    code = []
    
    # Helper to add word and update counts
    def add_word(w):
        code.append(w)
        for mask in all_masks:
            blocked_count[w ^ mask] += 1
            
    # Helper to remove word and update counts
    def remove_word(w):
        code.remove(w) # O(N), acceptable for small N
        for mask in all_masks:
            blocked_count[w ^ mask] -= 1

    # Phase 1: Greedy Construction
    # We can speed this up by just iterating. 
    # 2^17 = 131072. 
    # Python loop overhead is ~0.1s.
    for w in range(1 << N):
        if time.time() - start_time > time_limit:
            break
        if blocked_count[w] == 0:
            add_word(w)
            
    # Phase 2: Local Search (1-for-2 swaps)
    random.seed(42)
    
    while time.time() - start_time < time_limit:
        found_swap = False
        indices = list(range(len(code)))
        random.shuffle(indices)
        
        for idx in indices:
            if time.time() - start_time > time_limit:
                break
            
            c = code[idx]
            
            # Temporarily remove c
            for mask in all_masks:
                blocked_count[c ^ mask] -= 1
            
            # Find words that are now free (count == 0)
            # These are candidates for new words.
            # They must be in the neighborhood of c.
            newly_free = []
            for mask in all_masks:
                n = c ^ mask
                if blocked_count[n] == 0:
                    newly_free.append(n)
            
            # Try to find two compatible words in newly_free
            # Use vectorized check
            success = False
            if len(newly_free) >= 2:
                # Convert to numpy array for faster processing if needed, 
                # but list comprehension with bit_count might be faster for small K
                # Let's try to find a pair.
                # We can check all pairs.
                # To speed up, we can break early.
                
                # Optimization: If we just need ANY pair, we can check
                # w1 against all w2 > w1.
                
                for i in range(len(newly_free)):
                    w1 = newly_free[i]
                    # Check against w2 in newly_free[i+1:]
                    # Vectorized check
                    # Create array of w2s
                    w2s = np.array(newly_free[i+1:], dtype=np.int32)
                    if len(w2s) == 0:
                        continue
                    dists = np.array([np.bit_count(w1 ^ w2) for w2 in w2s])
                    # Find indices where dist >= MIN_DIST
                    valid_indices = np.where(dists >= MIN_DIST)[0]
                    if len(valid_indices) > 0:
                        # Found a valid w2
                        w2 = w2s[valid_indices[0]]
                        
                        # Execute swap
                        code.remove(c)
                        
                        code.append(w1)
                        for mask in all_masks:
                            blocked_count[w1 ^ mask] += 1
                            
                        code.append(w2)
                        for mask in all_masks:
                            blocked_count[w2 ^ mask] += 1
                        
                        success = True
                        found_swap = True
                        break
            
            if not success:
                # Restore c
                for mask in all_masks:
                    blocked_count[c ^ mask] += 1
            else:
                # Restart the loop to re-evaluate with new code
                break
                
        if not found_swap:
            # No improvement found in this pass
            # Try a different strategy? 
            # For now, break.
            # We can also try 1-for-1 swaps (moving a codeword) to unblock better positions,
            # but 1-for-2 is the one that increases size.
            break

    return code
