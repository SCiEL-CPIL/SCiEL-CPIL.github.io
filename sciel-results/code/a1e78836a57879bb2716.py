import numpy as np

def construct():
    N = 10_000_000
    
    # Compute sigma(n) for all n from 1 to N using sieve
    sigma = np.ones(N + 1, dtype=np.int64)
    for d in range(1, N + 1):
        sigma[d::d] += d
        
    # We want to minimize |a/b - 1.5|.
    # Candidates for a given b are floor(1.5*b) and ceil(1.5*b).
    
    b_vals = np.arange(1, N + 1, dtype=np.int64)
    
    a_floor = (3 * b_vals) // 2
    a_ceil = a_floor + 1
    
    # Valid ranges
    mask_floor = (a_floor >= 1) & (a_floor <= N)
    mask_ceil = (a_ceil >= 1) & (a_ceil <= N)
    
    best_a = 11
    best_b = 6
    best_err = float('inf')
    
    # Check floor candidates
    if np.any(mask_floor):
        bf = b_vals[mask_floor]
        af = a_floor[mask_floor]
        valid_f = sigma[af] == sigma[bf]
        
        if np.any(valid_f):
            # Get indices of valid floor candidates
            valid_indices_f = np.where(valid_f)[0]
            # Calculate errors for valid candidates
            # err = |af/bf - 1.5|
            # To avoid division, compare |2*af - 3*bf| / (2*bf)
            # We want to minimize this.
            # Numerator is integer. Denominator is integer.
            # We can compute the rational error exactly?
            # The task says "score is the binary64 serialization of that rational".
            # So we just need the float value.
            
            af_valid = af[valid_f]
            bf_valid = bf[valid_f]
            
            # Calculate errors
            # Using float for comparison is fine if we just need the min.
            # But to be safe, we can compute |2*af - 3*bf| and then divide.
            # Since we want the global min, we can just compute the float error.
            
            err_f = np.abs(af_valid.astype(np.float64) / bf_valid.astype(np.float64) - 1.5)
            
            # Filter out zero errors
            nonzero_mask = err_f > 0
            if np.any(nonzero_mask):
                min_idx_local = np.argmin(err_f[nonzero_mask])
                # Get the actual index in the valid_f array
                # This is tricky. Better to set zeros to inf.
                err_f_safe = np.where(nonzero_mask, err_f, np.inf)
                min_idx_f = np.argmin(err_f_safe)
                
                if err_f_safe[min_idx_f] < best_err:
                    best_err = err_f_safe[min_idx_f]
                    best_a = int(af_valid[min_idx_f])
                    best_b = int(bf_valid[min_idx_f])
            else:
                # All errors are 0? Then all are invalid.
                pass

    # Check ceil candidates
    if np.any(mask_ceil):
        bc = b_vals[mask_ceil]
        ac = a_ceil[mask_ceil]
        valid_c = sigma[ac] == sigma[bc]
        
        if np.any(valid_c):
            ac_valid = ac[valid_c]
            bc_valid = bc[valid_c]
            
            err_c = np.abs(ac_valid.astype(np.float64) / bc_valid.astype(np.float64) - 1.5)
            
            nonzero_mask = err_c > 0
            if np.any(nonzero_mask):
                err_c_safe = np.where(nonzero_mask, err_c, np.inf)
                min_idx_c = np.argmin(err_c_safe)
                
                if err_c_safe[min_idx_c] < best_err:
                    best_err = err_c_safe[min_idx_c]
                    best_a = int(ac_valid[min_idx_c])
                    best_b = int(bc_valid[min_idx_c])
            else:
                pass

    if best_err == float('inf'):
        return [11, 6]
        
    return [best_a, best_b]
