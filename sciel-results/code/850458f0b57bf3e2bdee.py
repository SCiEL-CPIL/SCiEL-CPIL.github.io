"""
Small nonzero signed harmonic subsums - optimized search
"""

def construct():
    import time
    import math
    from fractions import Fraction
    from itertools import product
    
    deadline = time.time() + 60.0
    n = 30
    L = 2329089562800  # lcm(1..30)
    
    # Precompute L/i for i=1..30
    weights = [L // i for i in range(1, n + 1)]
    
    # Meet-in-the-middle
    half = n // 2  # 15
    
    # Generate all sums for first half (positions 1..15)
    sums_a = []
    # Use iterative generation for speed
    def gen_sums(weights_subset):
        """Generate all (sum, coeff_indices) pairs."""
        results = [(0, [])]
        for idx, w in enumerate(weights_subset):
            new_results = []
            for s, cs in results:
                new_results.append((s, cs))
                new_results.append((s + w, cs + [idx]))
                new_results.append((s - w, cs + [idx]))
            results = new_results
        return results
    
    # First half: indices 0..14 correspond to positions 1..15
    first_half_weights = weights[:half]
    second_half_weights = weights[half:]
    
    # Generate sums for first half
    sums_a = gen_sums(first_half_weights)
    
    # Generate sums for second half
    sums_b = gen_sums(second_half_weights)
    
    # Sort sums_b by sum value
    sums_b.sort(key=lambda x: x[0])
    
    best_abs = None
    best_coeff = None
    
    # For each sum in A, find closest sum in B to make total near 0
    b_values = [x[0] for x in sums_b]
    b_indices = list(range(len(sums_b)))
    
    import bisect
    
    for sum_a, coeff_a in sums_a:
        # We want sum_b ≈ -sum_a
        target = -sum_a
        pos = bisect.bisect_left(b_values, target)
        
        # Check candidates around position pos
        for check_pos in [pos - 2, pos - 1, pos, pos + 1, pos + 2]:
            if 0 <= check_pos < len(sums_b):
                sum_b = sums_b[check_pos][0]
                total = sum_a + sum_b
                if total == 0:
                    continue
                abs_total = abs(total)
                if best_abs is None or abs_total < best_abs:
                    best_abs = abs_total
                    coeff_b = sums_b[check_pos][1]
                    # Build full coefficient vector
                    coeff_full = [0] * n
                    for idx in coeff_a:
                        coeff_full[idx] = 1
                    for idx in coeff_b:
                        coeff_full[half + idx] = 1
                    best_coeff = coeff_full
    
    # Verify and return
    if best_coeff is None:
        # Fallback: try simple pairs
        for i in range(n):
            for j in range(i + 1, n):
                total = Fraction(weights[i], L) - Fraction(weights[j], L)
                if total != 0:
                    abs_total = abs(total)
                    if best_abs is None:
                        best_abs = 1  # placeholder
                        best_coeff = [0] * n
                        best_coeff[i] = 1
                        best_coeff[j] = -1
    
    # Actually, let's do a cleaner verification
    if best_coeff is not None:
        total = Fraction(0)
        for i in range(n):
            if best_coeff[i] != 0:
                total += Fraction(best_coeff[i], i + 1)
        if total != 0:
            return best_coeff
    
    # Final fallback
    return [0] * 28 + [1, -1]
