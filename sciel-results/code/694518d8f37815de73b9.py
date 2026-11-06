"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

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
    # Use list of lists for dp and prev
    dp = [[0] * N for _ in range(N)]
    prev = [[-1] * N for _ in range(N)]
    
    # suf_max_val[i] = array of suffix max of dp[k][i] for k in [0, i-1]
    # suf_max_arg[i] = array of argmax indices
    suf_max_val = [None] * N
    suf_max_arg = [None] * N
    
    max_len = 1
    best_i = 0
    best_j = 0
    
    # Local references for speed
    p = primes
    
    # Iterate i (first index of the pair)
    for i in range(N):
        if time.time() > deadline:
            break
            
        # 1. Compute suffix max for column i
        if i > 0:
            sm_val = [0] * i
            sm_arg = [-1] * i
            # sm_val[k] = max(dp[k][i], dp[k+1][i], ..., dp[i-1][i])
            sm_val[i-1] = dp[i-1][i]
            sm_arg[i-1] = i-1
            for k in range(i-2, -1, -1):
                if dp[k][i] > sm_val[k+1]:
                    sm_val[k] = dp[k][i]
                    sm_arg[k] = k
                else:
                    sm_val[k] = sm_val[k+1]
                    sm_arg[k] = sm_arg[k+1]
            suf_max_val[i] = sm_val
            suf_max_arg[i] = sm_arg
        else:
            suf_max_val[i] = []
            suf_max_arg[i] = []
            
        # 2. Compute dp[i][j] for all j > i
        if i < N - 1:
            L = i  # Pointer for lower bound in p[0..i-1]
            pi = p[i]
            for j in range(i+1, N):
                X = 2 * pi - p[j]
                # Decrease L while p[L-1] >= X
                # p is increasing, so p[L-1] is the largest element < p[L]
                # We want the smallest index L such that p[L] >= X
                # If p[L-1] >= X, then L-1 is also valid, so we can decrement L
                while L > 0 and p[L-1] >= X:
                    L -= 1
                
                if L < i:
                    val = suf_max_val[i][L]
                    if val > 0:
                        dp[i][j] = val + 1
                        prev[i][j] = suf_max_arg[i][L]
                    else:
                        dp[i][j] = 2
                else:
                    dp[i][j] = 2
                    
                if dp[i][j] > max_len:
                    max_len = dp[i][j]
                    best_i = i
                    best_j = j

    # Reconstruct sequence
    seq = [primes[best_j]]
    curr_i = best_i
    curr_j = best_j
    while curr_i != -1:
        seq.append(primes[curr_i])
        prev_i = prev[curr_i][curr_j]
        curr_j = curr_i
        curr_i = prev_i
    
    seq.reverse()
    
    return seq

# EVOLVE-BLOCK-END
