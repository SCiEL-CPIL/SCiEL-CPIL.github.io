import numpy as np
def _score(U):
    if not U or 0 not in U:
        return 0.0
    U_arr = np.array(sorted(set(U)), dtype=np.int64)
    if len(U_arr) < 2:
        return 1.0 # Degenerate case, but score formula needs log(ratio)
    # Actually if len==1 (just 0), maxU=0, log(1)=0, division by zero? 
    # Formula: 1 + log(|U-U|/|U+U|)/log(2 max U + 1)
    # If U={0}, |U-U|=1, |U+U|=1, ratio=1, log(1)=0. log(1)=0. 0/0.
    # The prompt says "containing 0". If maxU=0, return 1.0 as per original code logic? 
    # Original code: if maxU == 0: return 1.0.
    
    maxU = U_arr[-1]
    if maxU == 0:
        return 1.0
        
    # Compute sumset and diffset
    # U_arr is 1D. 
    sumset_arr = np.add.outer(U_arr, U_arr)
    diffset_arr = np.subtract.outer(U_arr, U_arr)
    
    n_sum = len(np.unique(sumset_arr))
    n_diff = len(np.unique(diffset_arr))
    
    if n_sum == 0:
        return 0.0
    import math
    ratio = n_diff / n_sum
    if ratio <= 0:
        return 0.0
    denom = math.log(2 * maxU + 1)
    if denom == 0:
        return 1.0
    return 1.0 + math.log(ratio) / denom
