import time
import random

def construct():
    n = 512
    deadline = time.time() + 55.0  # 55s safety margin
    
    # 1. Precompute all 4-APs
    # A 4-AP is (a, a+d, a+2d, a+3d) with 1 <= a and a+3d <= n
    # We use 0-indexed for bitmask: 0..511
    # Elements are 1..512, so index i corresponds to element i+1
    aps = []
    # d can range from 1 to (n-1)//3
    max_d = (n - 1) // 3
    for d in range(1, max_d + 1):
        for a in range(1, n - 3 * d + 1):
            # a, a+d, a+2d, a+3d
            # Convert to 0-indexed
            i0 = a - 1
            i1 = a + d - 1
            i2 = a + 2 * d - 1
            i3 = a + 3 * d - 1
            aps.append((i0, i1, i2, i3))
            
    num_aps = len(aps)
    
    # Precompute for each index i, which APs contain it
    # aps_containing[i] = list of indices in aps
    aps_containing = [[] for _ in range(n)]
    for idx, ap in enumerate(aps):
        for i in ap:
            aps_containing[i].append(idx)

    # Helper: check if adding index i to mask creates a 4-AP
    # mask is an integer where bit j is 1 if j is in the set
    def is_valid_add(i, mask):
        # Check all APs containing i
        for ap_idx in aps_containing[i]:
            i0, i1, i2, i3 = aps[ap_idx]
            # Check if the other 3 are in mask
            # We can do this by checking bits
            # If i is i0, check i1, i2, i3
            # Mask check: (mask >> j) & 1
            # Faster: (mask & (1 << j)) != 0
            # Let's precompute bit masks for each index? 
            # Or just use bit shifts.
            # Since n=512, mask fits in a Python int easily.
            
            # We need to check if the other 3 bits are set.
            # Let's construct a bitmask for the other 3.
            # This is slightly slow if done dynamically.
            # Alternative: 
            # For each AP, store the 4 bits as a tuple of masks?
            # Or just check individually.
            
            # Optimization: 
            # For a specific AP (a,b,c,d), if i is in it, we know which 3 to check.
            # We can precompute for each (ap_idx, position_in_ap) the mask of the other 3.
            # But aps_containing just gives ap_idx.
            # We need to know which position i is in ap.
            
            # Let's refine aps_containing to store (ap_idx, pos)
            # Or just re-iterate the AP tuple.
            pass
    
    # Let's re-structure for speed.
    # Precompute for each index i, a list of (mask_of_other_3, ) for each AP containing i?
    # No, we just need to check if (mask & other_3_mask) == other_3_mask.
    
    # Re-do precomputation
    # For each i, store list of masks of the other 3 elements for each AP containing i.
    other_masks = [[] for _ in range(n)]
    for idx, ap in enumerate(aps):
        i0, i1, i2, i3 = ap
        # For i0, others are i1, i2, i3
        m0 = (1 << i1) | (1 << i2) | (1 << i3)
        other_masks[i0].append(m0)
        
        # For i1, others are i0, i2, i3
        m1 = (1 << i0) | (1 << i2) | (1 << i3)
        other_masks[i1].append(m1)
        
        # For i2, others are i0, i1, i3
        m2 = (1 << i0) | (1 << i1) | (1 << i3)
        other_masks[i2].append(m2)
        
        # For i3, others are i0, i1, i2
        m3 = (1 << i0) | (1 << i1) | (1 << i2)
        other_masks[i3].append(m3)

    def can_add(i, mask):
        for om in other_masks[i]:
            if (mask & om) == om:
                return False
        return True

    # 2. Greedy Initialization
    mask = 0
    for i in range(n):
        if can_add(i, mask):
            mask |= (1 << i)
            
    best_mask = mask
    best_size = bin(mask).count('1')
    
    # 3. Local Search
    # We will try to add elements. If stuck, remove one and try again.
    # To speed up "try to add", we can iterate through all i not in mask.
    
    # Optimization: Keep a list of available indices? 
    # Just iterate 0..511.
    
    while time.time() < deadline:
        added = False
        # Try to add any valid element
        # Randomize order to escape symmetry? 
        # Or just iterate. 
        # Iterating in fixed order might get stuck in specific local optima.
        # Let's shuffle the candidate list each "stuck" phase?
        # Or just iterate.
        
        for i in range(n):
            if not (mask & (1 << i)):
                if can_add(i, mask):
                    mask |= (1 << i)
                    added = True
                    if bin(mask).count('1') > best_size:
                        best_size = bin(mask).count('1')
                        best_mask = mask
        
        if not added:
            # Stuck. Remove a random element to allow new additions
            # Pick a random set bit
            set_bits = []
            temp_mask = mask
            while temp_mask:
                lsb = temp_mask & -temp_mask
                idx = lsb.bit_length() - 1
                set_bits.append(idx)
                temp_mask ^= lsb
            
            if set_bits:
                rem_idx = random.choice(set_bits)
                mask &= ~(1 << rem_idx)
            else:
                # Empty set? Restart?
                mask = 0
                for i in range(n):
                    if can_add(i, mask):
                        mask |= (1 << i)
        
    # Return best found
    # Convert mask to sorted list of 1-based integers
    result = []
    m = best_mask
    while m:
        lsb = m & -m
        idx = lsb.bit_length() - 1
        result.append(idx + 1)
        m ^= lsb
        
    return sorted(result)
