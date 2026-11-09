import math

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    L = 30030 # LCM of divisors
    
    # Precompute bitsets for each divisor
    # bit r is set if r % d == 0
    div_bitsets = []
    for d in divisors:
        mask = 0
        for r in range(0, L, d):
            mask |= (1 << r)
        div_bitsets.append(mask)
    
    # Function to check if SDR exists for given available masks
    def check_sdr(avails):
        # avails: list of 10 bitmasks
        # Sort indices by popcount to prune early
        order = sorted(range(n), key=lambda i: avails[i].bit_count())
        
        def backtrack(idx, used_mask):
            if idx == n:
                return True
            i = order[idx]
            options = avails[i] & ~used_mask
            while options:
                bit = options & -options
                if backtrack(idx + 1, used_mask | bit):
                    return True
                options ^= bit
            return False
        
        return backtrack(0, 0)

    # Iterate W from 1 upwards (W=0 impossible for distinct)
    # Max W is L-1
    for W in range(1, L):
        # Iterate start A
        for A in range(0, L):
            # Construct window mask
            # Residues in [A, A+W] modulo L
            if A + W < L:
                window_mask = ((1 << (W + 1)) - 1) << A
            else:
                # Wrap around
                # Part 1: A to L-1
                part1 = ((1 << (L - A)) - 1) << A
                # Part 2: 0 to A+W-L
                part2 = (1 << (A + W + 1 - L)) - 1
                window_mask = part1 | part2
            
            # Check if window has at least n bits
            if window_mask.bit_count() < n:
                continue
                
            # Calculate available masks for each divisor
            avails = [div_bitsets[i] & window_mask for i in range(n)]
            
            # Quick check: if any is 0, skip
            if any(a == 0 for a in avails):
                continue
                
            # Check SDR
            if check_sdr(avails):
                # We found a valid configuration.
                # Now we need to extract the actual values.
                # We need to find the specific residues chosen.
                # We can modify backtrack to return the choices.
                
                order = sorted(range(n), key=lambda i: avails[i].bit_count())
                choices = [None] * n
                
                def backtrack_find(idx, used_mask):
                    if idx == n:
                        return True
                    i = order[idx]
                    options = avails[i] & ~used_mask
                    while options:
                        bit = options & -options
                        if backtrack_find(idx + 1, used_mask | bit):
                            # Find which residue this bit corresponds to
                            # bit is (1 << r), so r = bit.bit_length() - 1
                            r = bit.bit_length() - 1
                            choices[i] = r
                            return True
                        options ^= bit
                    return False
                
                if backtrack_find(0, 0):
                    # Construct result
                    result = [0] * n
                    for i in range(n):
                        r = choices[i]
                        # Map residue r to value in [A, A+W]
                        if r >= A:
                            val = r
                        else:
                            val = r + L
                        # Ensure val > 0
                        if val == 0:
                            # This can happen if A=0 and r=0.
                            # But we need positive integers.
                            # If val is 0, we must add L?
                            # If we add L, it might be > A+W.
                            # However, if 0 is the only option, it's invalid for "positive".
                            # Let's check if there are other options.
                            # The SDR finder picked r=0.
                            # If r=0 is picked, and A=0, val=0.
                            # We should reject this specific assignment and try another?
                            # Or just shift the whole window?
                            # Actually, if A=0, the window is [0, W].
                            # 0 is in the window.
                            # If we must use 0 for some divisor, we can't.
                            # So we should ensure that if A=0, residue 0 is not available?
                            # No, residue 0 is a multiple of all.
                            # But we need positive integers.
                            # So we must exclude 0 from candidates.
                            # This means if A=0, we should effectively treat the window as [1, W] for the purpose of 0?
                            # Or just check if val == 0 and if so, this solution is invalid.
                            # If invalid, we need to find another SDR that doesn't use 0?
                            # Or move A?
                            # Simplest: if val == 0, break and try next A or W?
                            # But there might be another SDR for the same A, W that avoids 0.
                            # Let's just return None and continue search?
                            # But backtrack_find already committed.
                            # Let's modify the logic:
                            # If val == 0, we can add L to it?
                            # If we add L, val = L.
                            # Is L in [A, A+W]?
                            # If A=0, W < L, L is not in [0, W].
                            # So we can't just add L.
                            # So if 0 is chosen, it's a problem.
                            # We can avoid choosing 0 by masking it out if A=0?
                            # If A=0, we can clear bit 0 in avails for all i?
                            # But 0 is a valid multiple. We just can't use it as a value.
                            # So yes, if A=0, we should mask out 0.
                            
                            pass # Handled below
                        result[i] = val
                    
                    # Check for 0
                    if any(v == 0 for v in result):
                        continue # Try next A or W
                    return result
                    
    # Fallback
    return divisors[:]
