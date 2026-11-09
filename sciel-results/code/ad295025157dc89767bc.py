import time
import random
import sys

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    N = 32
    MAX_VAL = 512
    
    def score_mask(mask):
        """
        Calculate the number of distinct d > 0 such that a, a+d, a+2d are in the set.
        mask: int where bit i is 1 if i is in the set.
        """
        count = 0
        # Max d such that a+2d <= 512. If min a >= 1, max a+2d <= 512 => 2d <= 511 => d <= 255.
        for d in range(1, 256):
            # (mask & (mask >> d) & (mask >> (2 * d))) != 0
            # This checks if there exists an 'a' such that bits a, a+d, a+2d are set.
            # Note: mask >> d shifts bits down. 
            # If bit i is set in mask, bit i-d is set in mask>>d.
            # We want i, i+d, i+2d to be set in mask.
            # Let's verify: 
            # If a is in mask, bit a is 1.
            # If a+d is in mask, bit a+d is 1. In mask>>d, bit (a+d)-d = a is 1.
            # If a+2d is in mask, bit a+2d is 1. In mask>>(2d), bit (a+2d)-2d = a is 1.
            # So ANDing mask, mask>>d, mask>>2d at bit 'a' will be 1 if all three exist.
            if (mask & (mask >> d) & (mask >> (2 * d))) != 0:
                count += 1
        return count

    def mask_to_list(mask):
        res = []
        for i in range(1, 513):
            if (mask >> i) & 1:
                res.append(i)
        return res

    def list_to_mask(lst):
        m = 0
        for x in lst:
            m |= (1 << x)
        return m

    # Strategy: Cantor-like construction using Base 4 digits.
    # A set defined by digits in {0, 1} in base 4 (i.e., 00, 01, 10, 11 -> 0, 1, 4, 5)
    # creates a fractal structure. 
    # We want 32 elements. 2^5 = 32. So we can use 5 digits.
    # Values: sum_{i=0}^{4} d_i * 4^i where d_i in {0, 1}.
    # Max value: 1*4^0 + 1*4^1 + 1*4^2 + 1*4^3 + 1*4^4 = 1 + 4 + 16 + 64 + 256 = 341.
    # This fits in 1..512.
    
    # To maximize progressions, we might want to scale or shift this set.
    # Or use a different base.
    # Base 3 digits {0,1} gives 3^5 = 243 max. 32 elements.
    # Base 5 digits {0,1} gives 5^5 = 3125 (too big).
    # Base 4 is good.
    
    # The "Cantor" set C_5 = { sum_{i=0}^4 d_i 4^i | d_i in {0,1} }
    # This set has the property that if a, b, c are in C, and a+c=2b, it's likely they are in C?
    # Actually, Cantor sets are "additive" in a specific way.
    
    # Let's generate the base Cantor set.
    base_cantor = []
    for mask_d in range(32):
        val = 0
        for i in range(5):
            if (mask_d >> i) & 1:
                val += 4 ** i
        base_cantor.append(val)
        
    # base_cantor is a list of 32 distinct integers in 1..341.
    # We can transform this set: S = { a + k * x | x in base_cantor }
    # where a is an offset and k is a scale factor.
    # We need all elements to be in 1..512.
    # Min val in base_cantor is 0 (if we include 0, but task says 1..512, so we should shift).
    # Actually, my loop generates 0..341.
    # Let's shift base_cantor by 1 so min is 1.
    base_cantor = [x + 1 for x in base_cantor]
    
    # Now search for best (offset, scale)
    # S = { offset + scale * x | x in base_cantor }
    # Constraints:
    # min(S) >= 1 => offset + scale * min(base_cantor) >= 1
    # max(S) <= 512 => offset + scale * max(base_cantor) <= 512
    
    min_base = min(base_cantor)
    max_base = max(base_cantor)
    
    best_score = 0
    best_set = None
    
    # Grid search for offset and scale
    # scale can be 1, 2, ...
    # max_base is 341.
    # If scale=1, offset can be 0..170 (1+341+170 = 512)
    # If scale=2, max is 682 > 512, so scale=2 only works if offset is small?
    # 2*341 = 682. 682 > 512. So scale=2 is impossible unless we shift the base set down?
    # No, base_cantor is fixed.
    # So scale is likely 1.
    # Wait, if scale=1, we just shift. Shifting doesn't change the set of differences d.
    # So any shift of the Cantor set has the same score.
    
    # To get different scores, we need to modify the structure.
    # Perhaps use a "Mixed" Radix or randomize the digit choices slightly?
    # Or use a different base?
    
    # Let's try Base 3.
    # 3^5 = 243.
    # base_cantor_3 = { sum d_i 3^i | d_i in {0,1} }
    base_cantor_3 = []
    for mask_d in range(32):
        val = 0
        for i in range(5):
            if (mask_d >> i) & 1:
                val += 3 ** i
        base_cantor_3.append(val + 1) # Shift to 1..243
        
    # Base 2?
    # 2^5 = 32.
    # base_cantor_2 = { sum d_i 2^i | d_i in {0,1} } = {0..31}
    # This is just an interval. Interval [1, 32] has many progressions.
    # Score of [1, 32]:
    # d=1: 1,2,3... yes.
    # d=2: 1,3,5... yes.
    # ...
    # d=15: 1,16,31 yes.
    # d=16: 1,17,33 no (33>32).
    # So d=1..15. Score 15.
    
    # Score of Cantor Base 4?
    # Let's calculate score of base_cantor_4 (shifted by 1).
    mask_4 = list_to_mask(base_cantor)
    score_4 = score_mask(mask_4)
    
    mask_3 = list_to_mask(base_cantor_3)
    score_3 = score_mask(mask_3)
    
    # Interval 1..32
    mask_2 = list_to_mask(list(range(1, 33)))
    score_2 = score_mask(mask_2)
    
    # It turns out intervals often have high scores for small N.
    # But we can do better with "thick" sets.
    # A set like {1..16} U {33..48} ?
    
    # Let's use a hybrid approach:
    # 1. Start with the best of the constructive candidates (Cantor 3, Cantor 4, Interval).
    # 2. Use a fast local search (hill climbing) to refine the specific elements.
    #    The local search is much faster if we start from a good structure.
    
    candidates = [
        base_cantor,
        base_cantor_3,
        list(range(1, 33)),
        list(range(1, 17)) + list(range(33, 49)) # Two blocks
    ]
    
    best_initial_score = 0
    best_initial_set = None
    
    for c in candidates:
        # Ensure valid range 1..512
        if min(c) < 1 or max(c) > 512:
            continue
        m = list_to_mask(c)
        s = score_mask(m)
        if s > best_initial_score:
            best_initial_score = s
            best_initial_set = c
            
    if best_initial_set is None:
        best_initial_set = list(range(1, 33))
        best_initial_score = score_mask(list_to_mask(best_initial_set))
        
    current_set = list(best_initial_set)
    current_score = best_initial_score
    best_score = current_score
    best_mask = list_to_mask(current_set)
    
    # Local Search Refinement
    # Use a fast bitmask-based hill climbing.
    # Move: Swap one element with another.
    # To speed up, we can precompute the score contribution? No, bitmask AND is fast.
    
    # We will run hill climbing until time limit.
    # Since we start from a good spot, we might converge quickly.
    
    # To avoid getting stuck, we can allow a few "random restarts" if stagnation is detected,
    # but given the time limit, simple hill climbing from the best constructive start is a strong structural change.
    
    # Optimization: Maintain the list and mask.
    # On swap:
    # 1. Remove old element from mask.
    # 2. Add new element to mask.
    # 3. Calculate score.
    # 4. If score improves, keep. Else revert.
    
    # We can also try "2-swap" moves (swap 2 elements) for better escape.
    
    while time.time() < deadline:
        improved = False
        
        # Try single swaps
        # Pick a random element to remove
        idx_out = random.randint(0, 31)
        val_out = current_set[idx_out]
        
        # Pick a random element to add (not in set)
        # To make this efficient, we can pick from a list of "holes"
        # But generating holes every time is O(512).
        # Alternative: pick random 1..512, check if in set.
        # Set lookup is O(1) if we use a Python set or bitmask check.
        # Bitmask check: (mask >> val) & 1
        
        cur_mask = list_to_mask(current_set)
        
        val_in = random.randint(1, 512)
        while (cur_mask >> val_in) & 1:
            val_in = random.randint(1, 512)
        
        # Apply swap
        new_set = current_set.copy()
        new_set[idx_out] = val_in
        
        new_mask = list_to_mask(new_set)
        new_score = score_mask(new_mask)
        
        if new_score > current_score:
            current_score = new_score
            current_set = new_set
            improved = True
            if current_score > best_score:
                best_score = current_score
                best_mask = new_mask
        
        # Occasionally try a 2-swap to escape local optima
        if random.random() < 0.1:
            idx_out1 = random.randint(0, 31)
            idx_out2 = random.randint(0, 31)
            if idx_out1 == idx_out2:
                continue
            val_out1 = current_set[idx_out1]
            val_out2 = current_set[idx_out2]
            
            val_in1 = random.randint(1, 512)
            while (cur_mask >> val_in1) & 1:
                val_in1 = random.randint(1, 512)
            
            # Update mask temporarily for second check? 
            # Simpler: just build new set and mask.
            new_set = current_set.copy()
            new_set[idx_out1] = val_in1
            # Check if val_in1 conflicts with val_out2? No, we are replacing.
            # But we need to ensure val_in2 is not in the NEW set (which has val_in1 added).
            # Let's just pick val_in2 from the original complement, and check if it equals val_in1?
            # Actually, if val_in1 == val_in2, it's a 1-swap effectively.
            # If val_in1 is already in the set? We checked against cur_mask.
            # So val_in1 is not in current_set.
            # val_in2 must not be in current_set, and val_in2 != val_in1.
            
            val_in2 = random.randint(1, 512)
            while (cur_mask >> val_in2) & 1 or val_in2 == val_in1:
                val_in2 = random.randint(1, 512)
                
            new_set[idx_out2] = val_in2
            
            new_mask = list_to_mask(new_set)
            new_score = score_mask(new_mask)
            
            if new_score > current_score:
                current_score = new_score
                current_set = new_set
                improved = True
                if current_score > best_score:
                    best_score = current_score
                    best_mask = new_mask
    
    return mask_to_list(best_mask)
# EVOLVE-BLOCK-END
