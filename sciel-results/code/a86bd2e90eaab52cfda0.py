import numpy as np

def construct():
    N = 10_000_000
    
    # Compute sigma(n) for all n from 1 to N using a vectorized sieve
    sigma = np.zeros(N + 1, dtype=np.int64)
    for d in range(1, N + 1):
        sigma[d::d] += d
    
    # We want to minimize |a/b - 3/2| = |2a - 3b| / (2b)
    # Let k = |2a - 3b|. For a fixed k, the error is k/(2b), so we want large b.
    # For each k, 2a = 3b + k or 2a = 3b - k.
    # This requires 3b ± k to be even, i.e., b and k have the same parity.
    
    best_error = float('inf')
    best_a = 11
    best_b = 6
    
    # Check k values up to a reasonable limit.
    # For b ~ 10^7, k=1 gives error ~ 5e-8.
    # To beat the current score of ~4e6 (which corresponds to error ~ 2.5e-7),
    # we need k/(2b) < 2.5e-7.
    # If b is small, say 1000, we need k < 500.
    # But for small b, finding sigma(a)=sigma(b) is harder or the error is larger.
    # Let's check k up to 100. This covers a wide range of potential improvements.
    
    for k in range(1, 101):
        # Determine valid b values based on parity
        # 3b + k is even => b and k have same parity
        # 3b - k is even => b and k have same parity
        
        if k % 2 == 0:
            b_vals = np.arange(2, N + 1, 2, dtype=np.int64)  # even b
        else:
            b_vals = np.arange(1, N + 1, 2, dtype=np.int64)  # odd b
        
        if len(b_vals) == 0:
            continue
        
        # Case 1: a = (3b + k) / 2
        a_plus = (3 * b_vals + k) // 2
        mask_plus = (a_plus >= 1) & (a_plus <= N)
        
        if np.any(mask_plus):
            bp = b_vals[mask_plus]
            ap = a_plus[mask_plus]
            valid = sigma[ap] == sigma[bp]
            if np.any(valid):
                # Error is k / (2 * b). We want to maximize b among valid pairs.
                max_b_idx = np.argmax(bp[valid])
                b_candidate = int(bp[valid][max_b_idx])
                a_candidate = int(ap[valid][max_b_idx])
                err = k / (2.0 * b_candidate)
                if err < best_error:
                    best_error = err
                    best_a = a_candidate
                    best_b = b_candidate
        
        # Case 2: a = (3b - k) / 2
        a_minus = (3 * b_vals - k) // 2
        mask_minus = (a_minus >= 1) & (a_minus <= N)
        
        if np.any(mask_minus):
            bm = b_vals[mask_minus]
            am = a_minus[mask_minus]
            valid = sigma[am] == sigma[bm]
            if np.any(valid):
                max_b_idx = np.argmax(bm[valid])
                b_candidate = int(bm[valid][max_b_idx])
                a_candidate = int(am[valid][max_b_idx])
                err = k / (2.0 * b_candidate)
                if err < best_error:
                    best_error = err
                    best_a = a_candidate
                    best_b = b_candidate
    
    # Ensure types are correct
    best_a = int(best_a)
    best_b = int(best_b)
    
    return [best_a, best_b]
