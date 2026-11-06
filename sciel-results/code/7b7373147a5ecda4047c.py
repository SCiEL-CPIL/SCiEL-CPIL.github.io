import numpy as np
import time
import random
from itertools import combinations

def construct():
    N = 17
    MIN_DIST = 6
    start_time = time.time()
    time_limit = 55  # Safety margin
    
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
    
    # blocked_count[i] is the number of codewords within distance < MIN_DIST of i
    blocked_count = np.zeros(1 << N, dtype=np.int32)
    
    code = []
    
    def add_word(w):
        code.append(w)
        blocked_count[w ^ all_masks] += 1
        
    def remove_word(w):
        code.remove(w)
        blocked_count[w ^ all_masks] -= 1

    # Phase 1: Greedy Construction
    # We start with the 0 word, then greedily add the smallest valid word.
    # This typically yields the 256-word linear code structure if the order is right,
    # or a similar size code.
    for w in range(1 << N):
        if time.time() - start_time > time_limit:
            break
        if blocked_count[w] == 0:
            add_word(w)
            
    best_code = list(code)
    best_size = len(code)
    
    # Phase 2: Local Search
    # We try to improve the size.
    random.seed(42)
    
    no_improvement_count = 0
    max_no_improvement = 5000 # High threshold for local search steps
    
    def try_improve():
        nonlocal code, blocked_count, best_code, best_size, no_improvement_count
        
        # Randomly select a codeword to remove
        if not code:
            return False
            
        c_idx = random.randint(0, len(code) - 1)
        c = code[c_idx]
        
        # Temporarily remove c
        remove_word(c)
        
        # Find all words that are now free (blocked_count == 0)
        # These are candidates for re-adding
        # Note: c itself is free now.
        
        # We want to add k words from the free set such that they are mutually compatible.
        # Let's try to add 2 words.
        
        # Get list of free words
        # Efficiently: 
        # free_mask = (blocked_count == 0)
        # But we only care about words that were blocked by c? 
        # Actually, any word with blocked_count == 0 is valid to add.
        # However, most words are still blocked by other codewords.
        # The words that became free are those that were blocked ONLY by c (or c and others that are also gone? No, only c is gone).
        # So, words w such that blocked_count[w] was 1 (due to c) and is now 0.
        
        # To find these efficiently:
        # The set of words that were blocked by c is { c ^ m for m in all_masks }.
        # Among these, some might have been blocked by other codewords too.
        # We need the subset where blocked_count is now 0.
        
        candidate_indices = np.where(blocked_count[c ^ all_masks] == 0)[0]
        if len(candidate_indices) < 2:
            # Can't even find 2 free words? Then we can't do a 1->2 swap.
            # Maybe try 1->1 swap (keep size same, change structure)
            # But for now, restore c
            add_word(c)
            return False
            
        candidate_words = c ^ all_masks[candidate_indices]
        
        # We need to find a pair (w1, w2) in candidate_words such that dist(w1, w2) >= 6.
        # Also, they must be distinct from each other (implied by dist >= 6 > 0).
        
        # Optimization: 
        # If we just pick the first two valid, we might get stuck.
        # Let's try a few random pairs from the candidates.
        
        # If the candidate list is small, check all pairs.
        # If large, sample.
        
        found_pair = False
        w1 = -1
        w2 = -1
        
        # Heuristic: Check pairs involving the first candidate
        # Or just random sampling
        num_candidates = len(candidate_words)
        
        # Try up to 20 random pairs
        for _ in range(20):
            if num_candidates < 2:
                break
            i1 = random.randint(0, num_candidates - 1)
            i2 = random.randint(0, num_candidates - 1)
            if i1 == i2:
                continue
            w1 = candidate_words[i1]
            w2 = candidate_words[i2]
            if np.bit_count(w1 ^ w2) >= MIN_DIST:
                found_pair = True
                break
        
        # If random failed, try systematic (first few)
        if not found_pair and num_candidates >= 2:
            w1 = candidate_words[0]
            for i in range(1, min(50, num_candidates)):
                w2 = candidate_words[i]
                if np.bit_count(w1 ^ w2) >= MIN_DIST:
                    found_pair = True
                    break
        
        if found_pair:
            # Add w1 and w2
            add_word(w1)
            add_word(w2)
            
            # Update best
            if len(code) > best_size:
                best_size = len(code)
                best_code = list(code)
                no_improvement_count = 0
                return True
            else:
                # Size didn't change? (Shouldn't happen if we removed 1 and added 2, size increases by 1)
                # Wait, we removed c (size -1) and added w1, w2 (size +2). Net +1.
                # So size always increases if we find a valid pair.
                no_improvement_count = 0
                return True
        else:
            # Restore c
            add_word(c)
            no_improvement_count += 1
            return False

    # Main Loop
    while time.time() - start_time < time_limit:
        if no_improvement_count > max_no_improvement:
            # Perturbation: Randomly swap a codeword with another valid word
            # This keeps size constant but changes the code
            if code:
                c_idx = random.randint(0, len(code) - 1)
                c = code[c_idx]
                remove_word(c)
                
                # Find a random free word
                free_indices = np.where(blocked_count == 0)[0]
                if len(free_indices) > 0:
                    new_w = free_indices[random.randint(0, len(free_indices)-1)]
                    add_word(new_w)
                    no_improvement_count = 0
                else:
                    add_word(c)
                    # If no free words, we are very tight. Break?
                    break
            else:
                break
        else:
            try_improve()
            
    return best_code
