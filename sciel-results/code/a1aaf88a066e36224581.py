import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 75  # Leave 15s margin for 90s limit
    
    MAX_VAL = 256
    MAX_SUM = MAX_VAL * 30  # Max possible sum for 30 elements
    
    # Precompute multiples bitsets
    # multiples_bitset[s] has bits set at s, 2s, 3s, ...
    multiples_bitset = [0] * (MAX_SUM + 1)
    for s in range(1, MAX_SUM + 1):
        mask = 0
        for m in range(s, MAX_SUM + 1, s):
            mask |= (1 << m)
        multiples_bitset[s] = mask
    
    def _try_add(current_sums, candidate):
        """
        Try to add candidate to the set represented by current_sums bitset.
        Returns new_sums bitset if valid, else None.
        """
        # current_sums includes bit 0 (empty set)
        # New sums are current_sums OR (current_sums << candidate)
        
        shifted = current_sums << candidate
        new_sums = current_sums | shifted
        
        # Check for duplicates in non-empty sums
        # If any bit is set in both current_sums and shifted, 
        # it means a sum s in current_sums equals a sum s' + candidate.
        # This implies duplicate sum values.
        # Note: bit 0 is in current_sums. shifted has bit 'candidate'.
        # If candidate is already a sum in current_sums, then bit 'candidate' is in current_sums.
        # So current_sums & shifted will have bit 'candidate' set.
        # This correctly detects if candidate is already representable.
        if current_sums & shifted:
            return None
            
        # Check divisibility
        # We need to check all non-zero sums in new_sums
        # For each sum s in new_sums, if new_sums contains any multiple of s (other than s), it's invalid.
        # We iterate through set bits in new_sums.
        # To speed up, we can check only the sums that are "small" or just all of them.
        # Since max sum is ~7680, and number of sums is at most 2^30 (but usually much less due to distinctness constraint),
        # iterating through set bits is feasible.
        
        # Optimization: Only check sums s where s <= MAX_SUM/2? No, large sums can divide larger sums.
        # But if s > MAX_SUM/2, the only multiple is 2s > MAX_SUM, so no conflict with other sums in range.
        # So we only need to check s <= MAX_SUM // 2.
        
        temp = new_sums
        while temp:
            # Get lowest set bit
            lsb = temp & -temp
            s = lsb.bit_length() - 1
            if s == 0:
                temp ^= lsb
                continue
            if s > MAX_SUM // 2:
                # No multiples of s can fit in MAX_SUM except s itself
                # So no conflict possible from this s
                temp ^= lsb
                continue
                
            # Check if new_sums has any multiple of s other than s
            # multiples_bitset[s] has bits at s, 2s, 3s...
            # We want to see if (new_sums & multiples_bitset[s]) has any bit set other than s
            # mask_out_s = multiples_bitset[s] ^ lsb
            # if new_sums & mask_out_s: return None
            
            if new_sums & (multiples_bitset[s] ^ lsb):
                return None
            temp ^= lsb
            
        return new_sums

    def _greedy_build(candidates, current_sums, current_set, deadline):
        """
        Greedily add candidates to current_set.
        """
        for c in candidates:
            if time.time() > deadline:
                break
            if len(current_set) >= 30:
                break
            if c in current_set:
                continue
            new_sums = _try_add(current_sums, c)
            if new_sums is not None:
                current_set.add(c)
                current_sums = new_sums
        return current_set, current_sums

    best_set = set()
    best_len = 0
    
    # Strategy 1: Greedy with descending order (large numbers first)
    # Large numbers are less likely to divide each other or create small sums that divide others.
    cands_desc = list(range(256, 0, -1))
    s, sums = _greedy_build(cands_desc, 1, set(), deadline) # 1 is bit 0 set
    if len(s) > best_len:
        best_set = s
        best_len = len(s)
    
    # Strategy 2: Randomized Greedy
    # Use multiple random restarts
    for i in range(500):
        if time.time() > deadline:
            break
        cands = list(range(1, 257))
        random.shuffle(cands)
        s, sums = _greedy_build(cands, 1, set(), deadline)
        if len(s) > best_len:
            best_set = s
            best_len = len(s)
            if best_len >= 30:
                break

    # Strategy 3: Local Search / Refinement
    # Try to add more elements to the best set found
    if time.time() < deadline - 5:
        current_set = set(best_set)
        # Reconstruct sums bitset for current_set
        current_sums = 1
        for x in current_set:
            current_sums = current_sums | (current_sums << x)
        
        # Try to add any remaining number
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for c in range(1, 257):
                if c in current_set:
                    continue
                new_sums = _try_add(current_sums, c)
                if new_sums is not None:
                    current_set.add(c)
                    current_sums = new_sums
                    improved = True
                    if len(current_set) > best_len:
                        best_set = current_set
                        best_len = len(best_set)
                    break # Restart loop with new state
    
    # Strategy 4: Try specific heuristic starting points
    # Sometimes starting with a specific pattern helps
    heuristics = [
        list(range(128, 257)), # Large numbers
        list(range(1, 257, 2)), # Even numbers
        list(range(1, 257, 3)), # Multiples of 3
    ]
    for h in heuristics:
        if time.time() > deadline:
            break
        cands = [c for c in h if c not in best_set]
        random.shuffle(cands)
        # Start from best_set
        current_set = set(best_set)
        current_sums = 1
        for x in current_set:
            current_sums = current_sums | (current_sums << x)
        
        s, sums = _greedy_build(cands, current_sums, current_set, deadline)
        if len(s) > best_len:
            best_set = s
            best_len = len(best_len)
            
    return [int(x) for x in best_set]
