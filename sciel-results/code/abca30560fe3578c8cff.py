"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
_MAX = 512
_N = 32
_DEADLINE = 110.0  # safety margin before 150s kill

def _init_state(elements):
    """Initialize mask, cnt array, and score for a given set of elements."""
    mask = 0
    for x in elements:
        mask |= (1 << x)
    
    cnt = [0] * 256
    score = 0
    # Calculate initial cnt and score
    # For each d, count triplets
    for d in range(1, 256):
        c = 0
        # Check all possible starts x
        # x + 2d <= 512 => x <= 512 - 2d
        for x in range(1, _MAX - 2*d + 1):
            if (mask >> x) & 1 and (mask >> (x+d)) & 1 and (mask >> (x+2*d)) & 1:
                c += 1
        cnt[d] = c
        if c > 0:
            score += 1
    return mask, cnt, score

def _update_score(mask, cnt, score, r, a):
    """
    Update cnt and score for swapping r (remove) with a (add).
    Returns new score. Modifies cnt in place.
    Assumes r is in mask, a is not in mask.
    """
    # We need to check for each d if the count changes
    # and if that change crosses the 0 threshold.
    
    # Pre-compute bits for r and a? No, we use mask.
    # But for 'a', we need to treat it as present, and 'r' as absent.
    # Let's create a modified mask view? No, just logic.
    
    for d in range(1, 256):
        remove_count = 0
        add_count = 0
        
        # Check removal of r
        # Triplet starts x such that r is in (x, x+d, x+2d)
        # x = r
        if r + 2*d <= _MAX:
            if (mask >> (r+d)) & 1 and (mask >> (r+2*d)) & 1:
                remove_count += 1
        # x = r - d
        if r - d >= 1:
            if (mask >> (r+d)) & 1: # r is in S
                remove_count += 1
        # x = r - 2d
        if r - 2*d >= 1:
            if (mask >> (r-d)) & 1: # r is in S
                remove_count += 1
                
        # Check addition of a in S_new
        # S_new = S - {r} + {a}
        # x = a
        if a + 2*d <= _MAX:
            # Need a+d in S_new and a+2d in S_new
            # a+d is in S_new if (a+d != r and (mask >> (a+d)) & 1) or (a+d == a) -> impossible d>0
            # Actually if a+d == r, then a+d is NOT in S_new.
            # If a+d != r, it's in S_new iff it's in S.
            
            bit_a_d = (a+d)
            bit_a_2d = (a+2*d)
            
            ok_1 = False
            if bit_a_d == r:
                ok_1 = False
            else:
                ok_1 = bool((mask >> bit_a_d) & 1)
                
            ok_2 = False
            if bit_a_2d == r:
                ok_2 = False
            else:
                ok_2 = bool((mask >> bit_a_2d) & 1)
                
            if ok_1 and ok_2:
                add_count += 1
                
        # x = a - d
        if a - d >= 1:
            # Triplet (a-d, a, a+d)
            # Need a-d in S_new and a+d in S_new
            bit_a_minus_d = a - d
            bit_a_d = a + d
            
            ok_1 = False
            if bit_a_minus_d == r:
                ok_1 = False
            else:
                ok_1 = bool((mask >> bit_a_minus_d) & 1)
                
            ok_2 = False
            if bit_a_d == r:
                ok_2 = False
            else:
                ok_2 = bool((mask >> bit_a_d) & 1)
                
            if ok_1 and ok_2:
                add_count += 1
                
        # x = a - 2d
        if a - 2*d >= 1:
            # Triplet (a-2d, a-d, a)
            # Need a-2d in S_new and a-d in S_new
            bit_a_minus_2d = a - 2*d
            bit_a_minus_d = a - d
            
            ok_1 = False
            if bit_a_minus_2d == r:
                ok_1 = False
            else:
                ok_1 = bool((mask >> bit_a_minus_2d) & 1)
                
            ok_2 = False
            if bit_a_minus_d == r:
                ok_2 = False
            else:
                ok_2 = bool((mask >> bit_a_minus_d) & 1)
                
            if ok_1 and ok_2:
                add_count += 1

        delta = add_count - remove_count
        if delta != 0:
            old_cnt = cnt[d]
            new_cnt = old_cnt + delta
            if old_cnt > 0 and new_cnt == 0:
                score -= 1
            elif old_cnt == 0 and new_cnt > 0:
                score += 1
            cnt[d] = new_cnt
            
    return score

def construct():
    start_time = time.time()
    deadline = start_time + _DEADLINE
    rng = random.Random(42)
    
    # Initialize with a random set of 32 distinct integers in 1..512
    initial_elements = rng.sample(range(1, _MAX + 1), _N)
    current_elements = set(initial_elements)
    mask, cnt, current_score = _init_state(initial_elements)
    
    best_mask = mask
    best_score = current_score
    best_elements = list(initial_elements)
    
    # Simulated Annealing
    T = 1.0
    cooling = 0.9999
    min_T = 0.001
    
    while time.time() < deadline:
        if T < min_T:
            T = min_T
            
        # Generate a random neighbor: swap one element in with one element out
        rem = rng.choice(list(current_elements))
        
        while True:
            add = rng.randint(1, _MAX)
            if add not in current_elements:
                break
                
        # Apply move to mask and sets for state tracking
        # But we need to revert if rejected.
        # We update cnt and score tentatively.
        
        # To revert, we need to swap back.
        # Since _update_score is not easily reversible without saving state,
        # we can just recalculate if rejected? No, that's slow.
        # Better: save old cnt? No, too big.
        # Alternative: _update_score returns new score. 
        # If rejected, we must restore cnt and mask.
        # Saving cnt (256 ints) per step is cheap.
        
        old_cnt = cnt[:]
        old_score = current_score
        
        new_score = _update_score(mask, cnt, current_score, rem, add)
        
        # Update mask and elements
        mask ^= (1 << rem)
        mask |= (1 << add)
        current_elements.remove(rem)
        current_elements.add(add)
        current_score = new_score
        
        if new_score > best_score:
            best_score = new_score
            best_mask = mask
            best_elements = list(current_elements)
            
        if new_score > old_score or rng.random() < np.exp((new_score - old_score) / T):
            # Accept
            pass
        else:
            # Revert
            # Swap back
            mask ^= (1 << add)
            mask |= (1 << rem)
            current_elements.remove(add)
            current_elements.add(rem)
            current_score = old_score
            cnt = old_cnt
            
        T *= cooling
        
    return sorted(best_elements)
# EVOLVE-BLOCK-END
