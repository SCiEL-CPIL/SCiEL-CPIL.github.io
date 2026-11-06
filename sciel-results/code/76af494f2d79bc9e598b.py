import time
import random
import math

def _score_mask(M):
    """
    Calculate the number of distinct d > 0 such that there exists a
    with a, a+d, a+2d in the set represented by bitmask M.
    M is an integer where bit i is 1 if i is in the set.
    Indices 1..512 are used.
    """
    count = 0
    # d can range from 1 to 255 (since 1 + 2*255 = 511 <= 512)
    # If a=2, 2+2*255 = 512. So d=255 is possible.
    for d in range(1, 256):
        # Check if any a satisfies the condition
        # M & (M >> d) & (M >> 2*d)
        # If non-zero, there is at least one such a.
        if (M & (M >> d) & (M >> 2 * d)) != 0:
            count += 1
    return count

def construct():
    start_time = time.time()
    # Use 115 seconds to be safe, leaving 5s for cleanup/return
    deadline = start_time + 115.0
    
    N = 32
    MAXV = 512
    
    # Initialize with a random set
    cur_elems = random.sample(range(1, MAXV + 1), N)
    cur_mask = 0
    for e in cur_elems:
        cur_mask |= (1 << e)
    
    cur_score = _score_mask(cur_mask)
    best_mask = cur_mask
    best_score = cur_score
    
    # Simulated Annealing Parameters
    T = 10.0
    T_min = 0.1
    cooling = 0.9995
    
    # Phase 1: Simulated Annealing
    # Run until deadline - 15 seconds (leaving time for LS)
    sa_deadline = deadline - 15.0
    
    while time.time() < sa_deadline and T > T_min:
        # Pick a random element to remove
        # Expected ~16 tries
        while True:
            pos_remove = random.randint(1, MAXV)
            if (cur_mask >> pos_remove) & 1:
                break
        
        # Pick a random element to add
        # Expected ~1.07 tries
        while True:
            pos_add = random.randint(1, MAXV)
            if not ((cur_mask >> pos_add) & 1):
                break
        
        if pos_remove == pos_add:
            continue
        
        # Apply move
        new_mask = cur_mask
        new_mask &= ~(1 << pos_remove)
        new_mask |= (1 << pos_add)
        
        new_score = _score_mask(new_mask)
        delta = new_score - cur_score
        
        # Acceptance criterion
        if delta > 0 or random.random() < math.exp(delta / T):
            cur_mask = new_mask
            cur_score = new_score
            if cur_score > best_score:
                best_score = cur_score
                best_mask = cur_mask
        
        T *= cooling
        
    # Phase 2: Local Search Polish
    # Use remaining time to polish the best solution found.
    # We perform iterative best-improvement local search.
    # In each iteration, we check all possible single-element swaps.
    # If an improvement is found, we apply it and restart the scan.
    # If no improvement is found, we stop.
    
    # Extract elements from best_mask for efficient iteration
    best_elems = [i for i in range(1, MAXV + 1) if (best_mask >> i) & 1]
    
    improved = True
    # Reserve 2 seconds for final conversion and return
    ls_deadline = deadline - 2.0
    
    while improved and time.time() < ls_deadline:
        improved = False
        best_found_in_round = False
        best_new_mask = best_mask
        best_new_score = best_score
        best_new_elems = best_elems
        
        # Iterate over all possible swaps
        for i in range(N):
            if time.time() > ls_deadline:
                break
            
            rem = best_elems[i]
            rem_bit = 1 << rem
            base_mask = best_mask & ~rem_bit
            
            # Try adding any element not currently in the set
            # To speed this up, we can iterate 1..512 and skip set bits
            for add in range(1, MAXV + 1):
                if (best_mask >> add) & 1:
                    continue
                
                add_bit = 1 << add
                cand_mask = base_mask | add_bit
                cand_score = _score_mask(cand_mask)
                
                if cand_score > best_new_score:
                    best_new_score = cand_score
                    best_new_mask = cand_mask
                    # Update the element list for the new best
                    # We can't just update best_elems[i] directly because 
                    # we are scanning the *current* best_elems.
                    # But since we are looking for the *global* best neighbor,
                    # we just track the best candidate found in this round.
                    best_new_elems = best_elems[:]
                    best_new_elems[i] = add
                    improved = True
                    # We don't break here because we want to find the BEST neighbor,
                    # not just the first improving one.
                    # However, checking all 15k neighbors takes time.
                    # If we find a huge improvement, we might want to stop early?
                    # No, let's finish the round to be sure.
        
        # If we found an improvement, update the state and restart
        if improved:
            best_mask = best_new_mask
            best_score = best_new_score
            best_elems = best_new_elems

    # Convert best_mask to list of integers
    result = [i for i in range(1, MAXV + 1) if (best_mask >> i) & 1]
    return result
