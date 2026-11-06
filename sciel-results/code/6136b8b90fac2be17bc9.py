import time
import bisect

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
    # prev[i][j] = index k such that dp[k][i] was the max predecessor
    dp = [[0] * N for _ in range(N)]
    prev = [[-1] * N for _ in range(N)]
    
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
        # We need max dp[k][i] for k in [L, i-1]
        # suf_val[k] = max(dp[k][i], dp[k+1][i], ..., dp[i-1][i])
        # suf_arg[k] = argmax index
        if i > 0:
            # Array size i, indices 0 to i-1
            # We only care about indices < i
            sm_val = [0] * i
            sm_arg = [-1] * i
            
            # Initialize from the end
            # dp[i-1][i] is the last one
            sm_val[i-1] = dp[i-1][i]
            sm_arg[i-1] = i-1
            
            for k in range(i-2, -1, -1):
                if dp[k][i] > sm_val[k+1]:
                    sm_val[k] = dp[k][i]
                    sm_arg[k] = k
                else:
                    sm_val[k] = sm_val[k+1]
                    sm_arg[k] = sm_arg[k+1]
        else:
            sm_val = []
            sm_arg = []
            
        # 2. Compute dp[i][j] for all j > i
        if i < N - 1:
            pi = p[i]
            for j in range(i+1, N):
                # Condition: p_k >= 2*pi - p[j]
                X = 2 * pi - p[j]
                
                # Find smallest index L such that p[L] >= X
                # bisect_left returns first index where p[idx] >= X
                L = bisect.bisect_left(p, X)
                
                # Valid k must be < i and >= L
                # So k is in [L, i-1]
                if L < i:
                    # Query suffix max at index L
                    # sm_val[L] is max of dp[L][i] ... dp[i-1][i]
                    val = sm_val[L]
                    if val > 0:
                        dp[i][j] = val + 1
                        prev[i][j] = sm_arg[L]
                    else:
                        dp[i][j] = 2
                else:
                    dp[i][j] = 2
                    
                if dp[i][j] > max_len:
                    max_len = dp[i][j]
                    best_i = i
                    best_j = j

    # Reconstruct sequence
    if max_len == 1:
        return [primes[0]]
        
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
