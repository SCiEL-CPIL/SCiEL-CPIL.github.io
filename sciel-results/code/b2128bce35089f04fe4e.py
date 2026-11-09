import numpy as np
import time

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # Safety margin

    # 1. Generate primes <= 10000
    limit = 10000
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False
    for p in range(2, int(limit**0.5) + 1):
        if is_prime[p]:
            is_prime[p*p : limit+1 : p] = False

    primes = np.where(is_prime)[0]
    N = len(primes)

    if N == 0:
        return []

    # Create index map: index_map[p] = index of p in primes, or -1
    index_map = np.full(limit + 1, -1, dtype=np.int32)
    for idx, p in enumerate(primes):
        index_map[p] = idx

    # DP arrays
    # dp[i, g]: max length of seq ending at primes[i] with last gap exactly g
    # best_prefix[i, g]: max length of seq ending at primes[i] with last gap <= g
    # arg_gap[i, g]: the gap h <= g that achieves best_prefix[i, g]
    # par_idx[i, g]: index j of previous prime
    # par_gap[i, g]: gap h' at j used to achieve the state
    
    dp = np.zeros((N, limit + 1), dtype=np.uint16)
    best_prefix = np.zeros((N, limit + 1), dtype=np.uint16)
    arg_gap = np.zeros((N, limit + 1), dtype=np.uint16)
    par_idx = np.full((N, limit + 1), -1, dtype=np.int32)
    par_gap = np.zeros((N, limit + 1), dtype=np.uint16)

    # Init
    dp[:, 0] = 1
    best_prefix[:, 0] = 1
    arg_gap[:, 0] = 0

    global_max = 1
    best_i = 0
    best_g = 0

    for i in range(1, N):
        if time.time() - start_time > DEADLINE - 5:
            break
        
        pi = primes[i]
        
        # Find valid previous primes j < i
        # g = pi - primes[j]
        # We only care about g <= limit
        # primes[j] >= pi - limit
        # Since primes are increasing, we can find the range of j
        
        # Actually, just take all j < i
        prev_primes = primes[:i]
        gaps = pi - prev_primes
        # Filter gaps <= limit (always true since pi <= 10000, min prime 2, max gap 9998)
        # But we only store up to limit=10000.
        
        # We need to update dp[i, gaps]
        # dp[i, g] = 1 + best_prefix[j, g]
        # j = index_map[prev_prime]
        
        # Vectorized fetch
        # prev_primes are primes, so index_map[prev_primes] gives indices
        prev_indices = index_map[prev_primes]
        
        # Fetch best_prefix[prev_indices, gaps]
        # Note: gaps are unique for each j?
        # Yes, primes are distinct, so pi - p_j are distinct.
        
        fetched_vals = best_prefix[prev_indices, gaps]
        fetched_args = arg_gap[prev_indices, gaps]
        
        # Initialize dp_i row
        dp_i = np.zeros(limit + 1, dtype=np.uint16)
        dp_i[0] = 1
        
        # Update dp_i at positions 'gaps'
        # If multiple j give same gap? No, gaps are distinct.
        dp_i[gaps] = 1 + fetched_vals
        
        # Update par arrays
        par_idx_i = np.full(limit + 1, -1, dtype=np.int32)
        par_gap_i = np.zeros(limit + 1, dtype=np.uint16)
        par_idx_i[gaps] = prev_indices
        par_gap_i[gaps] = fetched_args
        
        # Compute best_prefix and arg_gap for row i
        cum_max = np.maximum.accumulate(dp_i)
        best_prefix[i, :] = cum_max
        
        # Compute arg_gap
        # We want the largest h <= g such that dp_i[h] == cum_max[g]
        # is_max = dp_i == cum_max
        # indices = np.arange(limit+1)
        # masked_indices = np.where(is_max, indices, -1)
        # arg_gap_i = np.maximum.accumulate(masked_indices)
        
        is_max = dp_i == cum_max
        indices = np.arange(limit + 1, dtype=np.int32)
        masked_indices = np.where(is_max, indices, -1)
        arg_gap_i = np.maximum.accumulate(masked_indices)
        # Ensure arg_gap_i is uint16 compatible (values <= 10000)
        arg_gap[i, :] = arg_gap_i.astype(np.uint16)
        
        par_idx[i, :] = par_idx_i
        par_gap[i, :] = par_gap_i
        
        # Update global max
        row_max = cum_max[-1]
        if row_max > global_max:
            global_max = row_max
            best_i = i
            # Find the gap that achieved this max
            # It's arg_gap[i, limit]
            best_g = int(arg_gap[i, limit])

    # Reconstruct
    if global_max <= 1:
        return [int(primes[0])]

    seq = []
    cur_i = best_i
    cur_g = best_g
    while cur_i >= 0:
        seq.append(int(primes[cur_i]))
        if par_idx[cur_i, cur_g] == -1:
            break
        next_g = int(par_gap[cur_i, cur_g])
        cur_i = int(par_idx[cur_i, cur_g])
        cur_g = next_g

    seq.reverse()
    return seq
