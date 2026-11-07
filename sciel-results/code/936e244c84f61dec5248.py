import time
import math
import numpy as np
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin for 90s limit

    def compute_sigma_numpy(L):
        """Compute sigma(n) for all n from 1 to L using NumPy vectorized sieve."""
        sigma = np.zeros(L + 1, dtype=np.int64)
        for i in range(1, L + 1):
            sigma[i::i] += i
        return sigma

    L = 10_000_000
    if time.time() - start_time > deadline - 30:
        # Fallback for very small limits
        best_a, best_b = 11, 6
        return [best_a, best_b]

    sigma = compute_sigma_numpy(L)

    best_error = Fraction(1, 3)  # |11/6 - 3/2| = |11-9|/6 = 2/6 = 1/3
    best_a, best_b = 11, 6

    # For each b, check a near 3b/2
    # a must be in [1, L], b in [1, L]
    # a = round(3b/2) + d for d in small range
    
    # We want |a/b - 3/2| = |2a - 3b| / (2b)
    # So we want 2a ≈ 3b, i.e., a ≈ 3b/2
    
    # For b from 1 to L:
    #   target_a = 3*b // 2
    #   Check a in {target_a - 2, target_a - 1, target_a, target_a + 1, target_a + 2}
    #   (and also (3b+1)//2 and (3b-1)//2 to cover odd cases)
    
    # Actually, 3b/2 might not be integer. Let's just check a range around 3b/2.
    
    # To be efficient, vectorize where possible, but the sigma lookup is random access.
    # Let's do it in chunks with NumPy for the arithmetic, then check.
    
    # Process in chunks to keep memory reasonable
    chunk_size = 1_000_000
    b_start = 1
    while b_start <= L:
        b_end = min(b_start + chunk_size - 1, L)
        if time.time() - start_time > deadline - 10:
            break
        
        bs = np.arange(b_start, b_end + 1, dtype=np.int64)
        # 3b values
        three_b = 3 * bs
        # a candidates: floor(3b/2) - 2 to floor(3b/2) + 2
        base_a = three_b // 2
        # Check 5 candidates: base_a-2, base_a-1, base_a, base_a+1, base_a+2
        # But also need to handle the case where 3b is odd, so 3b/2 is x.5
        # In that case, floor(3b/2) and floor(3b/2)+1 are the two closest integers.
        # Our range of 5 should cover it.
        
        for offset in [-2, -1, 0, 1, 2]:
            if time.time() - start_time > deadline - 5:
                break
            as_ = base_a + offset
            # Filter valid a: 1 <= a <= L, a != b
            valid = (as_ >= 1) & (as_ <= L) & (as_ != bs)
            if not np.any(valid):
                continue
            as_valid = as_[valid]
            bs_valid = bs[valid]
            # Get sigma values
            sig_a = sigma[as_valid]
            sig_b = sigma[bs_valid]
            # Find where they match
            match_mask = sig_a == sig_b
            if not np.any(match_mask):
                continue
            as_match = as_valid[match_mask]
            bs_match = bs_valid[match_mask]
            sig_match = sig_a[match_mask]
            
            # For each match, compute error |2a - 3b| / (2b)
            # num = |2a - 3b|, den = 2b
            # Compare with best_error: num/den < best_num/best_den
            # => num * best_den < best_num * den
            
            best_num = best_error.numerator
            best_den = best_error.denominator
            
            for i in range(len(as_match)):
                a_val = int(as_match[i])
                b_val = int(bs_match[i])
                num = abs(2 * a_val - 3 * b_val)
                den = 2 * b_val
                if num == 0:
                    continue  # Exact match, invalid
                # Check if this is better
                if num * best_den < best_num * den:
                    best_error = Fraction(num, den)
                    best_a, best_b = a_val, b_val
                    best_num = best_error.numerator
                    best_den = best_error.denominator
        
        b_start = b_end + 1

    return [int(best_a), int(best_b)]
