import numpy as np
import time
import random
from itertools import combinations

def construct():
    N = 17
    MIN_DIST = 6
    start_time = time.time()
    time_limit = 58  # Safe margin for 60s limit
    
    # Precompute masks for distances 0 to 5
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
    
    blocked_count = np.zeros(1 << N, dtype=np.int32)
    
    code = []
    code_set = set() # For faster removal/checks if needed, but list is fine for iteration
    
    def add_word(w):
        code.append(w)
        code_set.add(w)
        for mask in all_masks:
            blocked_count[w ^ mask] += 1
            
    def remove_word(w):
        code.remove(w)
        code_set.remove(w)
        for mask in all_masks:
            blocked_count[w ^ mask] -= 1

    # Phase 1: Greedy Construction
    for w in range(1 << N):
        if time.time() - start_time > time_limit:
            break
        if blocked_count[w] == 0:
            add_word(w)
            
    best_code = list(code)
    best_size = len(code)
    
    random.seed(42)
    
    # Phase 2: Local Search
    # We alternate between trying to grow (1-for-2) and shuffling (1-for-1) to escape local minima.
    # A 1-for-1 swap changes the code structure without changing size.
    # A 1-for-2 swap increases size.
    
    # To speed up the 1-for-2 check, we can precompute the candidate list more efficiently?
    # The main bottleneck is finding a pair (u, v) in candidates with d(u,v) >= 6.
    
    while time.time() - start_time < time_limit:
        # Randomly select a codeword to remove
        idx = random.randint(0, len(code) - 1)
        c = code[idx]
        
        # Remove c
        remove_word(c)
        
        # Find newly free words
        # These are words w such that blocked_count[w] == 0
        # We only care about words in the neighborhood of c, because only those changed status.
        # Wait, if a word was free before, it would have been added. So any free word now must have been blocked by c.
        # So we just check the neighborhood of c.
        neighbors = c ^ all_masks
        counts = blocked_count[neighbors]
        free_indices = np.where(counts == 0)[0]
        
        if len(free_indices) == 0:
            # No new words are free? This shouldn't happen if c was blocking itself and no one else.
            # c itself is in neighbors (mask 0). If no other codeword blocks c, c is free.
            # So len(free_indices) >= 1 always (c itself).
            # Unless c was blocked by others too.
            # If c was blocked by others, then after removing c, c might still be blocked.
            # But we removed c, so c's own contribution is gone.
            # If c is still blocked, it means some other codeword is within distance < 6 of c.
            # In that case, no new words are necessarily free?
            # Actually, if c was the *only* blocker for some w, w becomes free.
            # If c was blocked by others, it doesn't prevent other words from becoming free.
            # So we just look at free_indices.
            
            # If free_indices is empty, it means no word in the neighborhood is free.
            # This can happen if c was "surrounded" by other codewords.
            # In this case, we can't add anything. We must revert.
            add_word(c)
            continue
            
        candidates = neighbors[free_indices]
        
        # Try to find a pair (u, v) in candidates such that d(u, v) >= MIN_DIST
        # We can do this efficiently by iterating.
        # Optimization: If len(candidates) is small, it's fast. If large, it's slow.
        # Let's limit the search or use a smarter check.
        
        found_pair = False
        w1 = -1
        w2 = -1
        
        # Shuffle candidates to randomize the search order?
        # random.shuffle on a numpy array is not in-place efficiently? 
        # Let's just pick a random subset or iterate.
        
        # To avoid O(M^2) in worst case, we can try a few random pairs first.
        # But we need to be sure.
        
        # Let's try to find a pair.
        # If we find one, we add both.
        # If we don't, we might do a 1-for-1 swap.
        
        # Efficient pair search:
        # For each u in candidates, check if any v in candidates (v != u) is compatible.
        # We can break early.
        
        # To speed this up, we can limit the number of checks if M is large.
        # But M is usually small (< 100) in tight packings.
        
        # Let's iterate.
        M = len(candidates)
        for i in range(M):
            if time.time() - start_time > time_limit:
                break
            u = candidates[i]
            # Check against all j > i
            if i == M - 1:
                break
            v_list = candidates[i+1:]
            diffs = u ^ v_list
            dists = np.bit_count(diffs)
            valid_indices = np.where(dists >= MIN_DIST)[0]
            if len(valid_indices) > 0:
                w2 = v_list[valid_indices[0]]
                w1 = u
                found_pair = True
                break
        
        if found_pair:
            # Add w1 and w2
            add_word(w1)
            add_word(w2)
            
            if len(code) > best_size:
                best_size = len(code)
                best_code = list(code)
            
            # Continue loop to try to grow further
        else:
            # No pair found.
            # Option 1: Revert (remove c, add c back).
            # Option 2: 1-for-1 swap. Pick a random candidate w and add it.
            
            # Let's do a 1-for-1 swap with some probability to escape local minima.
            # Or always do it? If we always do it, we might wander too much.
            # Let's do it if we are stuck for a while?
            # Simpler: Always do a 1-for-1 swap if we can't grow.
            # This ensures we explore the space of maximal codes.
            
            # Pick a random candidate
            # Exclude c? If we pick c, it's a revert.
            # Let's pick a random candidate from free_indices.
            # If we pick c, we revert.
            # If we pick w != c, we swap.
            
            swap_idx = random.randint(0, len(free_indices) - 1)
            w_new = neighbors[free_indices[swap_idx]]
            
            if w_new == c:
                # Revert
                add_word(c)
            else:
                # Swap: add w_new
                # c is already removed
                add_word(w_new)
                # Size remains same, but structure changes.
                # We don't update best_size here, as size is same.
                # But we might find a better structure later.

    return best_code
