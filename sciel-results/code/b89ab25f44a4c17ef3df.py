import time
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 85.0  # 85 seconds to be safe against 90s limit
    
    n = 30
    
    # Split into two halves
    half = n // 2
    left_indices = list(range(1, half + 1))
    right_indices = list(range(half + 1, n + 1))
    
    # Precompute terms as Fractions
    left_terms = [Fraction(1, i) for i in left_indices]
    right_terms = [Fraction(1, i) for i in right_indices]
    
    def generate_sums(terms):
        """Generate all possible signed sums for a list of terms."""
        sums = [Fraction(0)]
        for t in terms:
            new_sums = []
            for s in sums:
                new_sums.append(s)       # coefficient 0
                new_sums.append(s + t)   # coefficient 1
                new_sums.append(s - t)   # coefficient -1
            sums = new_sums
        return sums
    
    # Generate sums for the left half
    if time.time() - start_time > deadline:
        return [0] * 28 + [1, -1]
        
    left_sums = generate_sums(left_terms)
    
    if time.time() - start_time > deadline:
        return [0] * 28 + [1, -1]

    # Generate sums for the right half
    right_sums = generate_sums(right_terms)
    
    if time.time() - start_time > deadline:
        return [0] * 28 + [1, -1]

    # Sort right sums to allow binary search
    right_sums.sort()
    
    best_sum = None
    best_abs = Fraction(float('inf'))
    
    import bisect
    
    # For each left sum, find the right sum that makes the total closest to 0
    # We want l + r to be close to 0, so r should be close to -l
    for l in left_sums:
        if time.time() - start_time > deadline:
            break
            
        target = -l
        # Find insertion point for target in right_sums
        pos = bisect.bisect_left(right_sums, target)
        
        # Check the element at pos and pos-1
        candidates = []
        if pos < len(right_sums):
            candidates.append(right_sums[pos])
        if pos > 0:
            candidates.append(right_sums[pos-1])
            
        for r in candidates:
            s = l + r
            if s == 0:
                # We found a zero sum, which is invalid.
                # The task says "signed sum ... must be nonzero".
                # So we must find the *next* closest non-zero.
                # However, since we are minimizing |s|, and s=0 is the absolute minimum,
                # we need to be careful. If s=0 is found, we cannot use it.
                # We should continue searching for the smallest non-zero.
                # But wait, if there are multiple pairs that sum to 0, we need to find
                # the pair with the smallest non-zero sum.
                # Actually, if s=0 is possible, the "minimum non-zero" might be very small.
                # We just skip s=0 in this candidate check.
                pass
            else:
                abs_s = abs(s)
                if abs_s < best_abs:
                    best_abs = abs_s
                    best_sum = s
                    # We need to recover the coefficients.
                    # This is tricky because generate_sums just returns the sum value.
                    # We need to track the path.
                    
    # The above approach fails to recover coefficients.
    # Let's modify generate_sums to return (sum, coeffs) or use a different strategy.
    # Given the 90s limit and 3^15 ~ 14M items, storing tuples might be memory heavy but feasible (14M * ~100 bytes = 1.4GB? Might be too much).
    # Alternative: Since we just need *any* valid coefficients that achieve the best sum,
    # we can re-run the search for the specific best_sum found, or store coefficients.
    # Storing coefficients: 14M * 15 bytes (if packed) or 14M * list of 15 ints.
    # A list of 15 ints is ~120 bytes. 14M * 120 = 1.6GB. This might exceed memory or be slow.
    
    # Let's try a different MITM variant that tracks coefficients efficiently.
    # Or, we can just use the local search result if MITM is too complex/slow to recover coeffs.
    # But the prompt asks for a structural change.
    
    # Let's refine the MITM to track coefficients using a dictionary mapping sum -> coeffs.
    # Since different coefficient vectors can yield the same sum, we only need to store ONE
    # coefficient vector for each unique sum value.
    # Number of unique sums is much smaller than 3^15?
    # The sums are rationals with denominator dividing LCM(1..15). LCM(1..15) is 360360.
    # So sums are multiples of 1/360360.
    # Range of sum: sum(1/i) for i=1..15 is approx 3.3.
    # So sums are in [-3.3, 3.3].
    # Number of unique values: approx 3.3 * 360360 * 2 = 2.3M.
    # This is very manageable!
    
    # Restart with Dictionary MITM
    def generate_sums_dict(terms):
        """Generate mapping from sum to one set of coefficients achieving it."""
        # coeffs is a tuple of 0s, 1s, -1s
        sums_dict = {Fraction(0): (0,) * len(terms)}
        for i, t in enumerate(terms):
            new_sums_dict = {}
            for s, coeffs in sums_dict.items():
                # Option 0: keep coeff as 0
                c0 = coeffs
                # Option 1: coeff becomes 1
                c1 = coeffs[:i] + (1,) + coeffs[i+1:]
                # Option -1: coeff becomes -1
                c_1 = coeffs[:i] + (-1,) + coeffs[i+1:]
                
                s0 = s
                s1 = s + t
                s_1 = s - t
                
                # Store if new or if we prefer shorter? No, any is fine.
                # We just need to ensure we don't overwrite with a worse coeff? 
                # Actually, any coeff that achieves the sum is valid.
                # But we must be careful: if s1 already exists in new_sums_dict, 
                # we can keep the existing one.
                
                if s0 not in new_sums_dict:
                    new_sums_dict[s0] = c0
                if s1 not in new_sums_dict:
                    new_sums_dict[s1] = c1
                if s_1 not in new_sums_dict:
                    new_sums_dict[s_1] = c_1
                    
            sums_dict = new_sums_dict
        return sums_dict

    if time.time() - start_time > deadline:
        return [0] * 28 + [1, -1]

    left_dict = generate_sums_dict(left_terms)
    
    if time.time() - start_time > deadline:
        return [0] * 28 + [1, -1]

    right_dict = generate_sums_dict(right_terms)
    
    if time.time() - start_time > deadline:
        return [0] * 28 + [1, -1]

    # Now we have left_dict and right_dict.
    # We want to find l in left_dict, r in right_dict such that |l+r| is minimized and l+r != 0.
    # Sort right_dict keys
    right_keys = sorted(right_dict.keys())
    
    best_abs = Fraction(float('inf'))
    best_l = None
    best_r = None
    
    for l, lc in left_dict.items():
        if time.time() - start_time > deadline:
            break
        target = -l
        pos = bisect.bisect_left(right_keys, target)
        
        candidates = []
        if pos < len(right_keys):
            candidates.append(right_keys[pos])
        if pos > 0:
            candidates.append(right_keys[pos-1])
            
        for r in candidates:
            s = l + r
            if s == 0:
                continue
            abs_s = abs(s)
            if abs_s < best_abs:
                best_abs = abs_s
                best_l = l
                best_r = r
                
    if best_l is None:
        # Fallback
        return [0] * 28 + [1, -1]
        
    # Recover coefficients
    lc = left_dict[best_l]
    rc = right_dict[best_r]
    
    # lc is tuple of length 15 for indices 1..15
    # rc is tuple of length 15 for indices 16..30
    # The task requires coefficients for 1/i where i=1..30.
    # coefficient[i-1] corresponds to 1/i.
    # So coeff[0] is for 1/1, ..., coeff[14] for 1/15.
    # coeff[15] for 1/16, ..., coeff[29] for 1/30.
    
    final_coeff = list(lc) + list(rc)
    
    return final_coeff
# EVOLVE-BLOCK-END
