import math
import bisect

def construct():
    limit = 10000
    # Sieve of Eratosthenes
    is_prime = [True] * (limit + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i * i, limit + 1, i):
                is_prime[j] = False
    
    primes = [i for i, prime in enumerate(is_prime) if prime]
    N = len(primes)
    
    if N == 0:
        return []
    if N == 1:
        return primes
    
    # dp[j][i] = max length of sequence ending with primes[j], primes[i] (j < i)
    dp = [[0] * N for _ in range(N)]
    parent = [[-1] * N for _ in range(N)]
    
    # suf_max[j] will be a list of length j, where suf_max[j][k] = max(dp[m][j] for m in k..j-1)
    suf_max = [None] * N
    
    max_len = 2
    best_i = 1
    best_j = 0
    
    for i in range(N):
        for j in range(i):
            gap = primes[i] - primes[j]
            target = 2 * primes[j] - primes[i]
            # Find smallest index L such that primes[L] >= target
            L = bisect.bisect_left(primes, target)
            
            best_val = 0
            best_m = -1
            if L < j:
                # Query max in dp[m][j] for m in [L, j-1]
                # suf_max[j] is built from previous iterations
                if suf_max[j] is not None:
                    best_val = suf_max[j][L]
                    # We need to find which m gave this max to store in parent
                    # This requires a linear scan in the worst case if we don't store argmax
                    # To keep it simple and fast, we can just store the max value.
                    # For backtracking, we need the index m.
                    # Let's modify suf_max to store (value, index) or just re-scan.
                    # Re-scanning is O(j) which makes it O(N^3).
                    # Better: store argmax in suf_max.
                    pass
            
            # We need to update the strategy to store argmax in suf_max
            # Let's redefine suf_max[j][k] as (max_val, arg_index)
            
            # Let's restart the inner logic with argmax storage
            # This requires changing the structure.
            
            # For now, let's just compute max_val. 
            # To get parent, we can scan back from j-1 to L to find the m with dp[m][j] == best_val.
            # Since we only need one valid parent, and the range might be small on average, 
            # or we can optimize by storing the argmax.
            
            # Let's use a list of tuples for suf_max.
            # suf_max[j][k] = (max_val, index)
            
            if L < j and suf_max[j] is not None:
                best_val, best_m = suf_max[j][L]
            else:
                best_val = 0
                best_m = -1
            
            if best_val > 0:
                dp[j][i] = best_val + 1
                parent[j][i] = best_m
            else:
                dp[j][i] = 2
                parent[j][i] = -1
                
            if dp[j][i] > max_len:
                max_len = dp[j][i]
                best_i = i
                best_j = j
        
        # Build suf_max[i] for use in future iterations
        # V_i = [dp[m][i] for m in 0..i-1]
        if i > 0:
            # Create suffix max array for column i
            # suf_max[i] has length i
            # suf_max[i][k] = max of dp[m][i] for m in k..i-1
            sm = [None] * i
            max_v = -1
            max_idx = -1
            for m in range(i - 1, -1, -1):
                if dp[m][i] > max_v:
                    max_v = dp[m][i]
                    max_idx = m
                sm[m] = (max_v, max_idx)
            suf_max[i] = sm
    
    # Backtrack
    result = []
    cur_i = best_i
    cur_j = best_j
    result.append(primes[cur_i])
    result.append(primes[cur_j])
    
    while parent[cur_j][cur_i] != -1:
        prev_m = parent[cur_j][cur_i]
        result.append(primes[prev_m])
        cur_i = cur_j
        cur_j = prev_m
        
    result.reverse()
    return result
