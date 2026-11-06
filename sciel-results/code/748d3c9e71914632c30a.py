"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin before 60s limit
    
    # 1. Generate primes <= 10000
    limit = 10000
    is_prime = [True] * (limit + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    
    primes = [i for i, flag in enumerate(is_prime) if flag]
    N = len(primes)
    
    if N < 2:
        return primes
    
    # dp[i][j] = max length of valid sequence ending with primes[i], primes[j] (i < j)
    # prev[i][j] = index k such that dp[k][i] was the max used to compute dp[i][j]
    
    # Use NumPy arrays for dp and prev
    dp = np.zeros((N, N), dtype=np.int32)
    prev = np.full((N, N), -1, dtype=np.int32)
    
    max_len = 1
    best_i = 0
    best_j = 0
    
    p = np.array(primes, dtype=np.int32)
    
    # Iterate i (first index of the pair)
    for i in range(N - 1):
        if time.time() > deadline:
            break
            
        pi = p[i]
        
        if i == 0:
            # Base case: dp[0][j] = 2 for all j > 0
            dp[0, 1:] = 2
            # prev[0][j] remains -1, which is correct as there is no k < 0
            # Update max_len
            if 2 > max_len:
                max_len = 2
                best_i = 0
                best_j = 1 # Any j > 0 works, but we track the first one found or the one with max length
                # Actually, we should track the specific (i,j) that gives the global max
                # Since all are 2, we can just pick one. But we need to be consistent for reconstruction.
                # If max_len is 2, the sequence is just [p[0], p[j]].
                # We'll update best_i, best_j properly in the loop below or after.
        else:
            # Get the column dp[0:i, i]
            col = dp[:i, i]
            
            # Compute suffix maximums and argmaxs for col
            # suf_val[k] = max(col[k], col[k+1], ..., col[i-1])
            # suf_arg[k] = index m in [k, i-1] such that col[m] == suf_val[k]
            
            # We want to find the index of the max in the suffix.
            # If there are ties, we can pick any. Let's pick the largest index (closest to i) 
            # or smallest? It doesn't matter for length, but for reconstruction, 
            # any valid k works.
            
            # Reverse col to use accumulate
            rev_col = col[::-1]
            # np.maximum.accumulate finds the max from left to right.
            # So on rev_col, it finds max of rev_col[0..k], which corresponds to col[i-1..i-1-k].
            # Let's just build it with a loop for clarity and correctness of argmax, 
            # as vectorized argmax with "last" or "first" logic is tricky in numpy without extra steps.
            # However, i is at most 1229. A Python loop of size 1229 is fast.
            
            suf_val = np.empty(i, dtype=np.int32)
            suf_arg = np.empty(i, dtype=np.int32)
            
            # Initialize last element (k = i-1)
            suf_val[i-1] = col[i-1]
            suf_arg[i-1] = i-1
            
            for k in range(i-2, -1, -1):
                if col[k] > suf_val[k+1]:
                    suf_val[k] = col[k]
                    suf_arg[k] = k
                else:
                    # If equal, we can keep the previous arg (which is larger index) 
                    # or update to k. 
                    # Keeping larger index (closer to i) is fine.
                    suf_val[k] = suf_val[k+1]
                    suf_arg[k] = suf_arg[k+1]
            
            # Now compute dp[i, j] for all j > i
            # j ranges from i+1 to N-1
            # X_j = 2 * p_i - p_j
            # We need p_k >= X_j. 
            # L_j = searchsorted(p, X_j)
            # If L_j < i, max_val = suf_val[L_j], arg = suf_arg[L_j]
            # Else max_val = 0, arg = -1
            
            # Vectorized calculation for j in [i+1, N)
            # p[i+1:]
            p_j = p[i+1:]
            X = 2 * pi - p_j
            
            # Find indices L_j in p such that p[L_j] >= X_j
            # searchsorted returns the index where the value would be inserted to maintain order.
            # Since we want >=, and p is sorted, this gives the first index >= X.
            L = np.searchsorted(p, X)
            
            # We are only interested in k < i.
            # So if L >= i, there is no valid k in [0, i-1].
            # If L < i, the valid range is [L, i-1].
            
            # Create a mask for valid L
            valid_mask = L < i
            
            # Initialize results
            new_dp = np.full(N - i - 1, 2, dtype=np.int32)
            new_prev = np.full(N - i - 1, -1, dtype=np.int32)
            
            # Process only valid entries
            if np.any(valid_mask):
                # Get the valid L indices
                L_valid = L[valid_mask]
                
                # Get the max values and args
                vals = suf_val[L_valid]
                args = suf_arg[L_valid]
                
                # vals should be > 0 because dp[k][i] is at least 2 for valid sequences?
                # Wait, dp[k][i] is the length ending at k,i.
                # Minimum length is 2 (just p_k, p_i).
                # So if L < i, there is at least one k in [L, i-1].
                # dp[k][i] is initialized to 0, but for k < i, dp[k][i] is computed.
                # Is it possible dp[k][i] is 0?
                # dp is initialized to 0.
                # We fill dp[i][j] for i < j.
                # So dp[k][i] is filled when the loop was at index k.
                # Since k < i, and we process i in increasing order, dp[k][i] is already computed.
                # The minimum value assigned to dp[k][i] is 2.
                # So vals should be >= 2.
                
                # Calculate new length
                new_vals = vals + 1
                
                # Assign back to the valid positions
                # We need to map valid indices back to the original array indices
                # The indices in new_dp correspond to j = i+1, i+2, ...
                # valid_mask corresponds to the same range.
                
                # We can use boolean indexing on new_dp and new_prev
                new_dp[valid_mask] = new_vals
                new_prev[valid_mask] = args
                
                # Update global max
                # Find the max in new_vals
                local_max = np.max(new_vals)
                if local_max > max_len:
                    max_len = local_max
                    # Find the index in new_vals that corresponds to this max
                    # We need the global j index.
                    # valid_indices = np.where(valid_mask)[0]
                    # max_local_idx = np.argmax(new_vals)
                    # j_offset = valid_indices[max_local_idx]
                    # best_j = i + 1 + j_offset
                    # best_i = i
                    # But we need to be careful if there are multiple maxes.
                    # We just need one.
                    
                    # Get all indices where new_vals == local_max
                    max_indices_in_valid = np.where(new_vals == local_max)[0]
                    if len(max_indices_in_valid) > 0:
                        # Pick the first one
                        first_idx = max_indices_in_valid[0]
                        # Map back to the position in the full j array
                        # valid_mask is a boolean array of size N-i-1
                        # valid_indices = np.nonzero(valid_mask)[0]
                        # j_offset = valid_indices[first_idx]
                        # This is inefficient to compute np.nonzero every time.
                        
                        # Alternative: 
                        # We can just check the global max at the end? 
                        # No, we need to track best_i, best_j.
                        
                        # Let's just find the argmax of the entire new_dp array?
                        # new_dp has 2s for invalid ones.
                        # If local_max > 2, then the max of new_dp is local_max.
                        # argmax of new_dp gives the offset.
                        offset = np.argmax(new_dp)
                        best_j = i + 1 + offset
                        best_i = i

            # Assign to dp and prev
            dp[i, i+1:] = new_dp
            prev[i, i+1:] = new_prev

    # Reconstruct sequence
    if max_len == 1:
        return [primes[0]]
        
    seq = [primes[best_j]]
    curr_i = best_i
    curr_j = best_j
    while curr_i != -1:
        seq.append(primes[curr_i])
        prev_i = prev[curr_i, curr_j]
        curr_j = curr_i
        curr_i = prev_i
    
    seq.reverse()
    
    return seq

# EVOLVE-BLOCK-END
