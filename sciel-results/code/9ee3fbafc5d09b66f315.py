"""Meet-in-the-middle exact search for minimal nonzero signed harmonic subsum."""
import math
import time
from itertools import product
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    """Return exactly 30 integer coefficients from {-1,0,1} such that
    sum(coeff[i-1]/i for i in 1..30) is nonzero and has minimal absolute value."""
    deadline = time.time() + 80  # safety margin before 90s kill
    
    n = 30
    # Split into two halves: indices 0..14 and 15..29
    half1 = 15  # denominators 1..15
    half2 = 15  # denominators 16..30
    
    # LCM of 1..30
    def lcm(a, b):
        return a // math.gcd(a, b) * b
    
    L = 1
    for i in range(1, n + 1):
        L = lcm(L, i)
    
    # Scaled weights: weight[i] = L // (i+1) for coefficient at position i (denom i+1)
    # Position 0 has denom 1, position 1 has denom 2, ..., position 29 has denom 30
    weights = [L // (i + 1) for i in range(n)]
    
    # Meet in the middle: enumerate all 3^15 assignments for each half
    # For each assignment, compute the sum of coeff[i] * weights[i]
    
    # Generate all 3^15 combinations for first half
    def enumerate_half(start, length):
        """Generate all sums for coefficients at positions start..start+length-1."""
        sums = []
        # Use recursive generation for efficiency
        current_sum = [0]
        for pos in range(start, start + length):
            new_sums = []
            w = weights[pos]
            for s in current_sum:
                new_sums.append(s)       # coeff = 0
                new_sums.append(s + w)   # coeff = 1
                new_sums.append(s - w)   # coeff = -1
            current_sum = new_sums
        return current_sum
    
    # We need to also track the coefficient assignments to reconstruct the solution
    # Better approach: enumerate with index tracking
    
    def enumerate_half_with_coeffs(start, length):
        """Generate list of (sum, coeffs_tuple) for positions start..start+length-1."""
        results = [(0, ())]
        for pos in range(start, start + length):
            w = weights[pos]
            new_results = []
            for s, c in results:
                new_results.append((s, c + (0,)))
                new_results.append((s + w, c + (1,)))
                new_results.append((s - w, c + (-1,)))
            results = new_results
        return results
    
    # For efficiency, let's just enumerate sums and keep track of one assignment per sum
    # Actually, we need the assignment to return it. Let's use a dict: sum -> assignment
    
    def enumerate_half_dict(start, length):
        """Generate dict mapping sum -> coeffs tuple for positions start..start+length-1."""
        d = {0: ()}
        for pos in range(start, start + length):
            w = weights[pos]
            new_d = {}
            for s, c in d.items():
                # coeff = 0
                key = s
                if key not in new_d:
                    new_d[key] = c + (0,)
                # coeff = 1
                key = s + w
                if key not in new_d:
                    new_d[key] = c + (1,)
                # coeff = -1
                key = s - w
                if key not in new_d:
                    new_d[key] = c + (-1,)
            d = new_d
        return d
    
    # Time check
    if time.time() > deadline:
        return [0] * 28 + [1, -1]
    
    # Enumerate first half (positions 0..14, denominators 1..15)
    d1 = enumerate_half_dict(0, half1)
    
    if time.time() > deadline:
        return [0] * 28 + [1, -1]
    
    # Enumerate second half (positions 15..29, denominators 16..30)
    d2 = enumerate_half_dict(half1, half2)
    
    if time.time() > deadline:
        return [0] * 28 + [1, -1]
    
    # Now find the pair (s1, s2) from d1 and d2 such that s1 + s2 != 0 and |s1 + s2| is minimized
    # Sort the sums from d2
    keys2 = sorted(d2.keys())
    
    best_sum = None
    best_abs = None
    best_c1 = None
    best_c2 = None
    
    # For each s1 in d1, we want to find s2 in d2 such that s1+s2 is closest to 0 but not 0
    # Use binary search on sorted keys2
    
    import bisect
    
    for s1, c1 in d1.items():
        # We want s2 such that s1 + s2 is close to 0, i.e., s2 is close to -s1
        # But s1 + s2 != 0, so s2 != -s1
        
        # Find position where -s1 would be in keys2
        target = -s1
        idx = bisect.bisect_left(keys2, target)
        
        # Check candidates around idx
        candidates = []
        for i in [idx - 1, idx, idx + 1]:
            if 0 <= i < len(keys2):
                s2 = keys2[i]
                total = s1 + s2
                if total != 0:
                    candidates.append((abs(total), total, s2))
        
        for abs_val, total, s2 in candidates:
            if best_abs is None or abs_val < best_abs:
                best_abs = abs_val
                best_sum = total
                best_c1 = c1
                best_c2 = d2[s2]
    
    if best_sum is None:
        # Fallback
        return [0] * 28 + [1, -1]
    
    # Reconstruct the full coefficient list
    coeffs = list(best_c1) + list(best_c2)
    
    # Verify
    total_frac = sum(Fraction(c, i + 1) for i, c in enumerate(coeffs))
    assert total_frac != 0, "Sum must be nonzero"
    assert all(c in (-1, 0, 1) for c in coeffs), "Coefficients must be in {-1,0,1}"
    assert len(coeffs) == 30, "Must have exactly 30 coefficients"
    
    return coeffs

# EVOLVE-BLOCK-END
