"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
import bisect
from math import gcd

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    """
    Return exactly 30 integer coefficients from {-1, 0, 1}.
    The signed sum sum_{i=1}^{30} c[i-1] / i must be nonzero.
    Minimize its absolute value.
    """
    start_time = time.time()
    DEADLINE = 85.0  # Safety margin before 90s kill

    N = 30
    half = N // 2
    first_half = list(range(1, half + 1))
    second_half = list(range(half + 1, N + 1))
    
    # Calculate LCM of 1..30 to know the minimum possible nonzero step
    lcm_val = 1
    for i in range(1, N + 1):
        lcm_val = _lcm(lcm_val, i)
    
    # The smallest possible nonzero absolute value is 1/lcm_val
    min_possible = Fraction(1, lcm_val)
    
    def generate_sums(indices):
        """Generate all possible signed sums and their coefficient vectors for the given indices."""
        sums = [0]
        coeffs_list = [[0] * len(indices)]
        for idx in indices:
            val = Fraction(1, idx)
            new_sums = []
            new_coeffs_list = []
            for s, coeffs in zip(sums, coeffs_list):
                # Option 1: +1
                new_sums.append(s + val)
                new_coeffs = coeffs[:]
                new_coeffs[-1] = 1
                new_coeffs_list.append(new_coeffs)
                
                # Option 2: -1
                new_sums.append(s - val)
                new_coeffs = coeffs[:]
                new_coeffs[-1] = -1
                new_coeffs_list.append(new_coeffs)
                
                # Option 3: 0
                new_sums.append(s)
                new_coeffs = coeffs[:]
                new_coeffs[-1] = 0
                new_coeffs_list.append(new_coeffs)
            sums = new_sums
            coeffs_list = new_coeffs_list
        return sums, coeffs_list

    # Generate sums for first half
    sums1, coeffs1 = generate_sums(first_half)
    
    # Check time
    if time.time() - start_time > DEADLINE:
        return [0] * 28 + [1, -1]

    # Generate sums for second half
    sums2, coeffs2 = generate_sums(second_half)
    
    # Check time
    if time.time() - start_time > DEADLINE:
        return [0] * 28 + [1, -1]

    # Create sorted list of (sum, coeffs) for second half
    pairs2 = list(zip(sums2, coeffs2))
    pairs2.sort(key=lambda x: x[0])
    sorted_sums2_vals = [p[0] for p in pairs2]
    
    best_val = None
    best_coeffs_full = None
    
    # For each sum in first half, find the best matching sum in second half
    for s1, c1 in zip(sums1, coeffs1):
        target = -s1
        pos = bisect.bisect_left(sorted_sums2_vals, target)
        
        # Check a small window around pos to find the minimum absolute sum
        # We check pos, pos-1, pos+1, etc.
        # Since we want the closest to 0, checking immediate neighbors is enough.
        for p in range(max(0, pos - 2), min(len(pairs2), pos + 3)):
            s2, c2 = pairs2[p]
            total = s1 + s2
            if total == 0:
                continue
            abs_total = abs(total)
            if best_val is None or abs_total < best_val:
                best_val = abs_total
                full_coeffs = c1 + c2
                best_coeffs_full = full_coeffs
                
                # Early termination if we found the theoretical minimum
                if best_val == min_possible:
                    return best_coeffs_full
    
    if best_coeffs_full is None:
        # Fallback
        return [0] * 28 + [1, -1]
    
    return best_coeffs_full

# EVOLVE-BLOCK-END
