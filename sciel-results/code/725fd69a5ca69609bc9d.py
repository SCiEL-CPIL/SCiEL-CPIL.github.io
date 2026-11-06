import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    N = 32
    MAX_VAL = 512
    
    # We represent the set S as a boolean numpy array of length 513 (1-indexed)
    # s[i] = True if i is in S
    
    # Pre-compute bitmasks for differences? 
    # Actually, to compute the set of differences d such that exists a: a, a+d, a+2d in S.
    # Let S be a set.
    # d is a valid difference if (S & (S >> d) & (S >> 2d)) has any 1s.
    # Using numpy, we can check all d at once?
    # It might be faster to just compute the score for a candidate set using vectorized operations.
    
    def get_score_vectorized(s):
        """
        s: numpy boolean array of length 513
        Returns: number of distinct d > 0 such that 3-AP exists.
        """
        # We can iterate d from 1 to 255 (max d is (512-1)/2 = 255.5, so 255)
        # But wait, d can be up to 255.
        # Let's try to vectorize the check for all d.
        
        # For each d, we need to check if there is any a such that s[a] and s[a+d] and s[a+2d] are True.
        # This is equivalent to checking if the intersection of S, S shifted by d, and S shifted by 2d is non-empty.
        
        # Since N is small (32), maybe a simpler O(N^2) or O(N^3) is fast enough?
        # 32^3 = 32768 operations. In Python this is ~1ms. 
        # But we want to do many iterations.
        # Let's try a hybrid: maintain the set as a Python set for fast membership, 
        # but compute score from scratch only when needed? No, that's what the old one did poorly.
        
        # Let's stick to the bitset idea but maybe just compute the score using a pre-computed list of APs?
        # Total number of APs in 1..512:
        # Sum_{d=1}^{255} (512 - 2d) ~ 512*255 - 2*255*256/2 = 130560 - 65280 = 65280.
        # So there are ~65k APs.
        # If we pre-compute these APs as a list of tuples (a, a+d, a+2d), 
        # and for each AP, check if all 3 are in S.
        # That's 65k checks per score evaluation.
        # If we do 1000 evaluations/sec, that's 65M checks/sec. Python might be too slow for that (100ns per check -> 6.5s per 1000 evals).
        
        # Better: Use bitsets.
        # Represent S as a single Python integer (bitmask).
        # Bit i is 1 if i is in S.
        # Then for a difference d, we can check:
        # mask & (mask >> d) & (mask >> 2d)
        # If this is non-zero, d is valid.
        
        # We can do this for all d in a loop. 255 iterations.
        # Each iteration is a few bitwise ops on a 512-bit integer. Very fast.
        
        return_val = 0
        # d max is 255
        for d in range(1, 256):
            # mask is an int
            # We need to ensure we don't pick up bits outside 1..512? 
            # The mask will only have bits 1..512 set if we initialize it that way.
            # When shifting right, bits fall off.
            # (mask >> d) & (mask >> 2d) & mask
            if (s_mask & (s_mask >> d) & (s_mask >> 2*d)) != 0:
                return_val += 1
        return return_val

    # Initial set: random or interval?
    # Interval 1..32 has score 4.8 (from prompt).
    # Let's start with a random set of 32 numbers, or maybe a structured one.
    # Random restarts are good for simulated annealing.
    
    # Let's use a bitmask representation for speed.
    # s_mask is an int where bit i is 1 if i is in S.
    
    # Helper to get set from mask
    def mask_to_set(mask):
        return [i for i in range(1, 513) if (mask >> i) & 1]
        
    # Helper to get mask from set
    def set_to_mask(s):
        m = 0
        for x in s:
            m |= (1 << x)
        return m

    # Initial state
    # Let's try starting with 1..32
    init_set = list(range(1, 33))
    current_mask = set_to_mask(init_set)
    current_score = get_score_vectorized(current_mask)
    
    best_mask = current_mask
    best_score = current_score
    
    # Simulated Annealing parameters
    T = 10.0
    T_min = 0.01
    alpha = 0.999
    
    # We will perform moves: pick a random element in S, remove it, pick a random element not in S, add it.
    # We need to update the mask.
    
    # To make moves faster, we can pre-compute the list of elements in the set?
    # Or just manipulate the mask directly.
    # Picking a random element in S:
    # We can maintain a list of active elements.
    
    active_elements = list(init_set)
    
    while time.time() < deadline:
        # Pick random element to remove
        idx = random.randint(0, len(active_elements) - 1)
        elem_to_remove = active_elements[idx]
        
        # Pick random element to add
        # We need an element not in S.
        # Randomly pick from 1..512, check if in S.
        # Or pick from a pre-computed list of all numbers and filter?
        # Since 32/512 is small, random sampling is efficient.
        new_elem = random.randint(1, 512)
        if new_elem in active_elements: # O(N) check, N=32. Fine.
             # Try again or just skip?
             # Let's just pick again if collision, or use a different strategy.
             # Actually, 32/512 ~ 6% collision.
             # Let's just loop until we get a unique one.
             while new_elem in active_elements:
                 new_elem = random.randint(1, 512)
                 
        # Apply move
        # Remove elem_to_remove
        active_elements[idx] = new_elem
        # Note: order doesn't matter for the set, but for the list we replaced it.
        # We need to update the mask.
        
        old_mask = current_mask
        # Clear bit for elem_to_remove
        current_mask &= ~(1 << elem_to_remove)
        # Set bit for new_elem
        current_mask |= (1 << new_elem)
        
        new_score = get_score_vectorized(current_mask)
        
        delta = new_score - current_score
        
        # Acceptance criterion
        if delta > 0 or (T > T_min and random.random() < np.exp(delta / T)):
            # Accept
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_mask = current_mask
        else:
            # Reject: revert
            current_mask = old_mask
            active_elements[idx] = elem_to_remove
            
        # Cool down
        T *= alpha
        
    return sorted(mask_to_set(best_mask))

