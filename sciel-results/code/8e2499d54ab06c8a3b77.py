import numpy as np
import time

def construct():
    start_time = time.time()
    DEADLINE = 85.0

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

    max_gap = limit
    # dp[i, d]: length of longest valid seq ending at primes[i] with last gap exactly d
    # We only need to store dp for the current i? No, we need best[j] for all j < i.
    # So we need to store best for all i.
    # best[i, d]: length of longest valid seq ending at primes[i] with last gap <= d
    best = np.zeros((N, max_gap + 1), dtype=np.int32)
    # To reconstruct, we need:
    # parent_idx[i, d]: index j of previous prime
    # parent_gap[i, d]: the gap d' used at j
    parent_idx = np.full((N, max_gap + 1), -1, dtype=np.int32)
    parent_gap = np.zeros((N, max_gap + 1), dtype=np.int32)
    
    # arg_gap[i, d]: the gap d' <= d such that dp[i, d'] = best[i, d]
    # We don't need to store arg_gap for all i if we can derive it?
    # No, we need it for reconstruction.
    arg_gap = np.zeros((N, max_gap + 1), dtype=np.int32)
    
    # dp_temp for current i
    # We can just update best[i] directly?
    # No, best[i, d] = max(best[i, d-1], dp[i, d])
    # So we need dp[i, d] first.
    
    global_max_len = 1
    best_i = 0
    best_d = 0
    
    for i in range(N):
        if time.time() - start_time > DEADLINE - 5:
            break
            
        # Initialize dp for row i
        # dp[i, d] is initially 1 (sequence of just primes[i]) if d=0?
        # No, a sequence of length 1 has no gaps.
        # Let's say the "gap" for a single element is 0.
        # Then best[i, 0] = 1.
        # And for d > 0, dp[i, d] is computed from transitions.
        
        # Actually, let's treat the first element specially.
        # If i is the first element, len=1.
        # If we transition from j to i with gap d, len = best[j, d] + 1.
        
        # Let's use a temporary array for dp[i]
        dp_i = np.zeros(max_gap + 1, dtype=np.int32)
        
        # Base case: sequence of length 1 ending at i.
        # This corresponds to "last gap" 0.
        dp_i[0] = 1
        
        for j in range(i):
            d = int(primes[i] - primes[j])
            # The previous gap must be <= d
            # The best length ending at j with last gap <= d is best[j, d]
            val = int(best[j, d])
            if val > 0:
                new_len = val + 1
                if new_len > dp_i[d]:
                    dp_i[d] = new_len
                    parent_idx[i, d] = j
                    # We need to store which gap was used at j
                    # This is arg_gap[j, d]
                    parent_gap[i, d] = int(arg_gap[j, d])
        
        # Compute prefix max for row i to get best[i]
        # best[i, g] = max(dp_i[0], ..., dp_i[g])
        # Also compute arg_gap[i, g]
        
        # We can do this iteratively
        current_max = 0
        current_arg = 0
        for g in range(max_gap + 1):
            if dp_i[g] > current_max:
                current_max = dp_i[g]
                current_arg = g
            best[i, g] = current_max
            arg_gap[i, g] = current_arg
            
            if current_max > global_max_len:
                global_max_len = current_max
                best_i = i
                best_d = current_arg
                
    # Reconstruct
    if global_max_len <= 1:
        return [int(primes[0])]
        
    seq = []
    cur_i = best_i
    cur_g = best_d
    
    while True:
        seq.append(int(primes[cur_i]))
        if parent_idx[cur_i, cur_g] == -1:
            break
        next_g = int(parent_gap[cur_i, cur_g])
        cur_i = int(parent_idx[cur_i, cur_g])
        cur_g = next_g
        
    seq.reverse()
    return seq
