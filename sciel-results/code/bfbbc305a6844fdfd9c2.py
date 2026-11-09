import time
import bisect

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s margin
    
    # 1. Generate primes <= 10000
    limit = 10000
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i, prime in enumerate(is_prime) if prime]
    N = len(primes)
    
    if N == 0:
        return []
    if N == 1:
        return primes
    
    # 2. DP Initialization
    # dp[i][j] : length of longest valid subsequence ending at primes[i] with previous primes[j]
    # parent[i][j] : index k such that primes[k] is before primes[j] in the optimal subsequence
    dp = [[0] * N for _ in range(N)]
    parent = [[-1] * N for _ in range(N)]
    
    # pref[i][m] : max(dp[i][k] for k in 0..m)
    # best_k[i][m] : argmax index k
    pref = [[0] * N for _ in range(N)]
    best_k = [[-1] * N for _ in range(N)]
    
    max_len = 0
    best_end_i = 0
    best_end_j = 0
    
    for i in range(N):
        for j in range(i):
            # Check deadline
            if time.time() > deadline:
                # Fallback to greedy if DP is too slow (unlikely)
                values = []
                for p in primes:
                    if len(values) < 2 or p - values[-1] >= values[-1] - values[-2]:
                        values.append(p)
                return values
            
            limit_val = 2 * primes[j] - primes[i]
            
            # Find largest m < j such that primes[m] <= limit_val
            # We search in primes[0:j]
            m = bisect.bisect_right(primes, limit_val, 0, j) - 1
            
            if m < 0:
                dp[i][j] = 2
                parent[i][j] = -1
            else:
                # max dp[j][k] for k in 0..m
                dp[i][j] = 1 + pref[j][m]
                parent[i][j] = best_k[j][m]
                
                if dp[i][j] > max_len:
                    max_len = dp[i][j]
                    best_end_i = i
                    best_end_j = j
        
        # Compute pref[i] and best_k[i] for use in future iterations (i' > i)
        if time.time() > deadline:
            values = []
            for p in primes:
                if len(values) < 2 or p - values[-1] >= values[-1] - values[-2]:
                    values.append(p)
            return values
            
        if i > 0:
            curr_max = 0
            curr_best = -1
            for k in range(i):
                if dp[i][k] > curr_max:
                    curr_max = dp[i][k]
                    curr_best = k
                pref[i][k] = curr_max
                best_k[i][k] = curr_best
        
    # Backtrack
    if max_len == 0:
        return primes[:1]
        
    seq = []
    i = best_end_i
    j = best_end_j
    while j != -1:
        seq.append(primes[i])
        k = parent[i][j]
        i, j = j, k
    seq.append(primes[i])
    seq.reverse()
    
    return seq
