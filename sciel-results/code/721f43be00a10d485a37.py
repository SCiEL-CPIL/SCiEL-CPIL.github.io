import time

def construct():
    start_time = time.time()
    limit = 85  # 5s safety margin from 90s limit
    
    N = 12
    period = 27720
    
    candidate_m = 0
    
    # L must be at least 12 to fit 12 distinct integers
    for L in range(12, 200):
        # Build lookup table for this L
        # table[i][r] = bitmask of deltas d in [1, L] such that d % i == r
        table = {}
        for i in range(1, N + 1):
            table[i] = {}
            for r in range(i):
                mask = 0
                start = r if r != 0 else i
                d = start
                while d <= L:
                    mask |= (1 << (d - 1))
                    d += i
                table[i][r] = mask
        
        exists_bad_m = False
        current_bad_m = None
        
        for m in range(period):
            # Time check
            if time.time() - start_time > limit:
                return candidate_m
            
            # Build candidate masks for each index, sorted by popcount (most constrained first)
            cands_masks = []
            for i in range(1, N + 1):
                r = (-m) % i
                mask = table[i][r]
                if mask == 0:
                    # No candidates for this index, matching impossible
                    break
                cands_masks.append(mask)
            
            if len(cands_masks) < N:
                # Matching impossible
                exists_bad_m = True
                current_bad_m = m
                break
            
            # Sort by number of set bits (most constrained first)
            cands_masks.sort(key=lambda x: bin(x).count('1'))
            
            # Check if matching exists using iterative backtracking
            # We need to assign one bit from each mask such that all bits are distinct
            # Use DFS with bitmask tracking used bits
            
            # Quick check: total available bits must be >= 12
            all_bits = 0
            for mask in cands_masks:
                all_bits |= mask
            if bin(all_bits).count('1') < N:
                exists_bad_m = True
                current_bad_m = m
                break
            
            # Iterative backtracking with explicit stack
            # Each stack frame: (idx, used_mask, bit_pos)
            # bit_pos indicates which bit in the available set we're trying
            # We'll use a list as stack
            
            # Precompute bit lists for each mask
            bit_lists = []
            for mask in cands_masks:
                bits = []
                mm = mask
                while mm:
                    b = mm & (-mm)
                    bits.append(b)
                    mm ^= b
                bit_lists.append(bits)
            
            # Stack: list of (idx, used_mask, next_bit_index)
            # For idx, we try bit_lists[idx][next_bit_index] if not in used_mask
            stack = [(0, 0, 0)]
            found = False
            
            while stack:
                idx, used, bi = stack.pop()
                
                if idx == N:
                    found = True
                    break
                
                bl = bit_lists[idx]
                progressed = False
                for j in range(bi, len(bl)):
                    bit = bl[j]
                    if not (used & bit):
                        if idx + 1 == N:
                            found = True
                            break
                        # Push: try this bit, then continue with remaining
                        # First push "continue" (j+1) for current idx
                        if j + 1 < len(bl):
                            stack.append((idx, used, j + 1))
                        # Then push next idx
                        stack.append((idx + 1, used | bit, 0))
                        progressed = True
                        break
                
                if found:
                    break
                if not progressed:
                    continue
            
            if not found:
                exists_bad_m = True
                current_bad_m = m
                break
        
        if not exists_bad_m:
            # All m have matching for this L
            # The max f is L
            # Return the m that failed for L-1
            return candidate_m
        else:
            # Some m failed, so max f > L
            candidate_m = current_bad_m

    # Fallback
    return candidate_m
