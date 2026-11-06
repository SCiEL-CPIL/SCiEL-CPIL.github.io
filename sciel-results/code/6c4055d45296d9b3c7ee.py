import numpy as np
import time
from itertools import combinations
import random

def construct():
    N = 17
    MIN_DIST = 6
    start_time = time.time()
    time_limit = 58  # Leave margin for 60s limit
    
    # Precompute masks for distances 0 to 5
    # Distance 0 is the word itself (blocked)
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
    all_masks = np.array(all_masks, dtype=np.uint32)
    
    # blocked_count[i] is the number of codewords within distance < MIN_DIST of i
    blocked_count = np.zeros(1 << N, dtype=np.int32)
    
    code = []
    
    def add_word(w):
        code.append(w)
        # Vectorized update
        blocked_count[w ^ all_masks] += 1
        
    def remove_word(w):
        code.remove(w)
        # Vectorized update
        blocked_count[w ^ all_masks] -= 1

    # Phase 1: Greedy Construction
    # This yields the 256-word linear code
    for w in range(1 << N):
        if time.time() - start_time > time_limit:
            break
        if blocked_count[w] == 0:
            add_word(w)
            
    best_code = list(code)
    best_size = len(code)
    
    # Phase 2: Local Search
    # We try to increase the size by replacing 1 word with 2.
    
    random.seed(42)
    
    no_improvement_count = 0
    max_no_improvement = 20 # Lower threshold to restart more often
    
    # Helper to find a valid pair in newly_free
    def find_valid_pair(newly_free_arr):
        if len(newly_free_arr) < 2:
            return -1, -1
        
        # Iterate through candidates for w1
        for i in range(len(newly_free_arr) - 1):
            w1 = newly_free_arr[i]
            # Compute distances from w1 to all other candidates j > i
            diffs = w1 ^ newly_free_arr[i+1:]
            dists = np.bit_count(diffs)
            
            # Find indices where distance >= MIN_DIST
            valid_indices = np.where(dists >= MIN_DIST)[0]
            
            if len(valid_indices) > 0:
                w2_idx = i + 1 + valid_indices[0]
                return i, w2_idx
        return -1, -1

    while time.time() - start_time < time_limit:
        improved = False
        
        # Shuffle codewords to try different removals
        indices = list(range(len(code)))
        random.shuffle(indices)
        
        for idx in indices:
            if time.time() - start_time > time_limit:
                break
            
            c = code[idx]
            
            # Temporarily remove c
            blocked_count[c ^ all_masks] -= 1
            
            # Find words that are now free (count == 0)
            # They must be in the neighborhood of c
            # We check all masks
            candidate_masked = c ^ all_masks
            newly_free_indices = np.where(blocked_count[candidate_masked] == 0)[0]
            
            # Map back to actual words
            # newly_free_indices are indices into all_masks
            # Word = c ^ all_masks[newly_free_indices]
            # Note: c ^ 0 = c, but c is removed, so blocked_count[c] should be 0?
            # When we removed c, we decremented blocked_count[c] by 1 (for mask 0).
            # If c was the only blocker for c, blocked_count[c] becomes 0.
            # But c is not in the code anymore.
            # We should exclude c itself if it appears?
            # Actually, if c is free, it's a valid candidate to re-add (swap) or part of a pair.
            
            newly_free_words = c ^ all_masks[newly_free_indices]
            
            if len(newly_free_words) >= 2:
                w1_idx, w2_idx = find_valid_pair(newly_free_words)
                
                if w1_idx != -1:
                    w1 = newly_free_words[w1_idx]
                    w2 = newly_free_words[w2_idx]
                    
                    # Perform the swap
                    # c is already removed from blocked_count
                    # We need to remove c from the list
                    code.remove(c)
                    
                    # Add w1
                    code.append(w1)
                    blocked_count[w1 ^ all_masks] += 1
                    
                    # Add w2
                    code.append(w2)
                    blocked_count[w2 ^ all_masks] += 1
                    
                    improved = True
                    if len(code) > best_size:
                        best_size = len(code)
                        best_code = list(code)
                    
                    break
                else:
                    # No pair found, restore c
                    blocked_count[c ^ all_masks] += 1
            else:
                # Not enough candidates, restore c
                blocked_count[c ^ all_masks] += 1
            
        if not improved:
            no_improvement_count += 1
            if no_improvement_count > max_no_improvement:
                # Perturbation: Swap a random codeword with a random free word
                # This keeps size same but changes structure
                c_idx = random.randint(0, len(code) - 1)
                c = code[c_idx]
                
                swapped = False
                # Try a few times to find a valid swap
                for _ in range(50):
                    # Pick a random word w
                    w = random.randint(0, (1 << N) - 1)
                    # Check if w is free
                    if blocked_count[w] == 0:
                        # Check if w is compatible with c (it is, since it's free)
                        # And compatible with all others (it is, since it's free)
                        # So we can swap
                        blocked_count[c ^ all_masks] -= 1
                        code.remove(c)
                        
                        code.append(w)
                        blocked_count[w ^ all_masks] += 1
                        
                        swapped = True
                        break
                
                if not swapped:
                    # If we can't even swap, we are likely stuck in a very tight configuration
                    # Try a different random seed or break
                    break
                else:
                    no_improvement_count = 0

    return best_code
