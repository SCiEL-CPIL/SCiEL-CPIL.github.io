# EVOLVE-BLOCK-START
import bisect
from fractions import Fraction

def construct():
    # Split indices into two halves
    n = 30
    half = n // 2  # 15
    
    def gen_sums(indices):
        """
        Generate all signed sums for the given indices.
        Returns a list of tuples (sum_value, code).
        code is an integer encoding the choices in base 3.
        digit 0 -> sign 0
        digit 1 -> sign +1
        digit 2 -> sign -1
        """
        sums = [(0.0, 0)]
        power = 1
        for i in indices:
            inv = 1.0 / (i + 1)
            new_sums = []
            for s, c in sums:
                # Sign 0
                new_sums.append((s, c))
                # Sign +1
                new_sums.append((s + inv, c + power))
                # Sign -1
                new_sums.append((s - inv, c + 2 * power))
            sums = new_sums
            power *= 3
        return sums

    # Generate sums for first half (indices 0-14)
    sums_A = gen_sums(range(half))
    # Generate sums for second half (indices 15-29)
    sums_B = gen_sums(range(half, n))
    
    # Sort sums_B by value to allow binary search
    # We need to keep track of codes
    sorted_B = sorted(sums_B, key=lambda x: x[0])
    vals_B = [x[0] for x in sorted_B]
    codes_B = [x[1] for x in sorted_B]
    
    best_abs = float('inf')
    best_a_code = None
    best_b_code = None
    
    # To handle the case where the minimum is 0, we might need the next best.
    # We will store the top 10 candidates to be safe.
    # But usually, the global minimum non-zero is very small.
    # Let's just find the global minimum absolute value first.
    # If it is 0, we need to find the next one.
    # Given the randomness, exact 0 is rare but possible.
    # Let's collect a few best candidates.
    
    candidates = []
    
    for a_val, a_code in sums_A:
        target = -a_val
        idx = bisect.bisect_left(vals_B, target)
        
        # Check idx and idx-1
        for j in [idx, idx - 1]:
            if 0 <= j < len(vals_B):
                b_val = vals_B[j]
                total = a_val + b_val
                abs_total = abs(total)
                
                # We only care if it's potentially small
                # And we want to avoid 0 if possible, but we check exact later.
                # If abs_total is very small, it's a candidate.
                
                # To avoid storing millions, we only keep if it's better than current best
                # OR if current best is 0 (or close to 0) and this is different.
                
                # Simple strategy: Keep the top 5 smallest absolute values.
                if abs_total < best_abs:
                    best_abs = abs_total
                    best_a_code = a_code
                    best_b_code = codes_B[j]
                    
                    # If we find a very small value, it's likely the one.
                    # But if it's 0, we need backup.
                    # Let's just maintain a small list of the best found so far.
                    candidates.append((abs_total, a_code, codes_B[j]))
                    candidates.sort(key=lambda x: x[0])
                    if len(candidates) > 10:
                        candidates.pop()
                    # Update best to the smallest in candidates?
                    # No, best_abs tracks the global min found so far.
                    # If we pop, we might lose track of the "next best" if the min is 0.
                    # Actually, if min is 0, best_abs is 0.
                    # We need the smallest NON-ZERO.
                    # So we should filter out 0s?
                    # Float 0 is hard to filter reliably if true sum is 1e-18.
                    # Let's just keep the top 10 smallest absolute values.
                    # At the end, we verify them.

    # If no candidates? (Should not happen)
    if not candidates:
        # Fallback to seed logic or simple heuristic
        coeff = [0] * 30
        coeff[0] = 1
        return coeff

    # Verify candidates with exact arithmetic
    # Sort candidates by absolute value (already sorted, but ensure)
    candidates.sort(key=lambda x: x[0])
    
    for abs_val, a_code, b_code in candidates:
        # Decode a_code
        coeff = [0] * n
        temp_code = a_code
        for i in range(half):
            digit = temp_code % 3
            temp_code //= 3
            if digit == 1:
                coeff[i] = 1
            elif digit == 2:
                coeff[i] = -1
            else:
                coeff[i] = 0
                
        # Decode b_code
        temp_code = b_code
        for i in range(half):
            digit = temp_code % 3
            temp_code //= 3
            if digit == 1:
                coeff[half + i] = 1
            elif digit == 2:
                coeff[half + i] = -1
            else:
                coeff[half + i] = 0
                
        # Calculate exact sum
        exact_sum = sum(Fraction(c, i + 1) for i, c in enumerate(coeff))
        
        if exact_sum != 0:
            return coeff

    # If all top candidates were zero (unlikely), fallback
    # Try next candidates if we stored more?
    # We only stored 10. If all 10 are zero, we might need to search deeper.
    # But probability of 10 distinct zero sums is low.
    # Fallback to a valid non-zero sum
    coeff = [0] * 30
    coeff[0] = 1
    return coeff

# EVOLVE-BLOCK-END
