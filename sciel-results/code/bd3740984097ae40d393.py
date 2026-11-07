import time
import numpy as np
import bisect

def construct():
    t0 = time.time()
    HARD_STOP = t0 + 85.0
    
    N = 10000000
    
    # 1. Compute sigma(n) for n=1..N using vectorized sieve
    sigma = np.zeros(N + 1, dtype=np.int64)
    
    # Additive sieve
    # Optimization: We can skip even numbers? No, sigma is defined for all.
    # The loop is the bottleneck.
    # Let's try to run it.
    for i in range(1, N + 1):
        if time.time() > HARD_STOP - 15.0:
            return [11, 6]
        sigma[i::i] += i
        
    # 2. Group by sigma
    indices = np.arange(1, N + 1, dtype=np.int64)
    
    if time.time() > HARD_STOP - 10.0:
        return [11, 6]
        
    order = np.argsort(sigma[1:], kind='stable')
    sorted_indices = order + 1
    
    if time.time() > HARD_STOP - 5.0:
        return [11, 6]
        
    sorted_sigma = sigma[sorted_indices]
    
    changes = np.nonzero(sorted_sigma[1:] != sorted_sigma[:-1])[0]
    starts = np.concatenate(([0], changes + 1))
    ends = np.concatenate((changes + 1, [len(sorted_sigma)]))
    
    best_val = float('inf')
    best_pair = None
    
    # Iterate through groups
    for gi in range(len(starts)):
        if time.time() > HARD_STOP - 2.0:
            break
            
        s0 = starts[gi]
        e0 = ends[gi]
        length = e0 - s0
        
        if length < 2:
            continue
            
        gn = sorted_indices[s0:e0].copy()
        gn.sort()
        
        L = len(gn)
        
        # Vectorized search for best pair in this group
        # We want to minimize |2a - 3b| / (2b) for a, b in gn, a != b
        # Let b be the denominator.
        # For each b, target a is 1.5 * b.
        # Find indices in gn closest to 1.5 * b.
        
        targets = 1.5 * gn.astype(np.float64)
        
        # searchsorted gives the index where target would be inserted
        # to maintain sorted order.
        # The closest elements are at idx and idx-1.
        idxs = np.searchsorted(gn, targets)
        
        # Candidate 1: gn[idxs]
        # Need to handle bounds: idxs can be L
        # If idxs == L, candidate is gn[L-1]
        # If idxs == 0, candidate is gn[0]
        
        # We need to check two candidates for each b:
        # c1 = gn[min(idxs, L-1)]
        # c2 = gn[max(idxs-1, 0)]
        
        # But we must ensure c1 != b and c2 != b.
        # And we want to minimize |2a - 3b| / (2b).
        
        # Let's construct arrays of candidates.
        # To avoid out of bounds, we can clip indices.
        # However, if idxs == L, gn[idxs] is invalid.
        
        # Let's create a safe index array.
        # For each i in 0..L-1 (representing b_i):
        #   i1 = min(idxs[i], L-1)
        #   i2 = max(idxs[i]-1, 0)
        
        i1 = np.minimum(idxs, L - 1)
        i2 = np.maximum(idxs - 1, 0)
        
        a1 = gn[i1]
        a2 = gn[i2]
        b_vals = gn
        
        # Calculate error for a1
        # err1 = |2*a1 - 3*b| / (2*b)
        # Only valid if a1 != b
        mask1 = a1 != b_vals
        err1 = np.full(L, np.inf)
        if np.any(mask1):
            num1 = np.abs(2 * a1[mask1] - 3 * b_vals[mask1])
            den1 = 2 * b_vals[mask1]
            # Avoid division by zero? b >= 1, so no.
            # But if num1 == 0, error is 0.
            # The task says "raw value of exactly 0 cannot be scored and counts as invalid".
            # So we should avoid pairs where 2a == 3b.
            # If num1 == 0, we should set error to inf or handle it.
            # Actually, if 2a == 3b, then a/b = 1.5.
            # Since a, b are integers, this is only possible if a=3k, b=2k.
            # Then sigma(3k) == sigma(2k)?
            # If such a pair exists, the raw value is 0.
            # The prompt says "A raw value of exactly 0 cannot be scored and counts as invalid."
            # So we MUST NOT return a pair with 2a == 3b.
            # So we treat num == 0 as invalid (inf error).
            
            valid_num1 = num1 > 0
            if np.any(valid_num1):
                err1[mask1][valid_num1] = num1[valid_num1] / den1[valid_num1]
            # If num1 == 0, err1 remains inf.
            
        # Calculate error for a2
        mask2 = a2 != b_vals
        err2 = np.full(L, np.inf)
        if np.any(mask2):
            num2 = np.abs(2 * a2[mask2] - 3 * b_vals[mask2])
            den2 = 2 * b_vals[mask2]
            valid_num2 = num2 > 0
            if np.any(valid_num2):
                err2[mask2][valid_num2] = num2[valid_num2] / den2[valid_num2]
        
        # Combine errors
        # We want the minimum of err1 and err2 for each b.
        # But we also need to recover the pair (a, b).
        # If err1 < err2, best a for this b is a1.
        # If err2 < err1, best a for this b is a2.
        
        # Find global minimum
        # We can create an array of all candidate errors and find the min.
        # Or just take min of err1 and err2 element-wise.
        
        # Let's just find the index of the minimum in the combined list?
        # No, simpler:
        # Find min of err1 and corresponding b, a.
        # Find min of err2 and corresponding b, a.
        # Compare the two bests.
        
        min_idx1 = np.argmin(err1)
        min_val1 = err1[min_idx1]
        
        min_idx2 = np.argmin(err2)
        min_val2 = err2[min_idx2]
        
        if min_val1 < min_val2:
            curr_val = min_val1
            curr_b = int(b_vals[min_idx1])
            curr_a = int(a1[min_idx1])
        else:
            curr_val = min_val2
            curr_b = int(b_vals[min_idx2])
            curr_a = int(a2[min_idx2])
            
        if curr_val < best_val:
            best_val = curr_val
            best_pair = (curr_a, curr_b)
            
    if best_pair is None:
        return [11, 6]
        
    a, b = best_pair
    # Verify
    if sigma[a] != sigma[b]:
        return [11, 6]
    if 2 * a == 3 * b:
        return [11, 6]
        
    return [int(a), int(b)]
