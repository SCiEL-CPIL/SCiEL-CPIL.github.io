import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    # Sieve for primes up to 10000
    limit = 10000
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    
    primes = [i for i, v in enumerate(is_prime) if v]
    N = len(primes)
    
    if N == 0:
        return []
        
    INF = 10**9
    
    # dp[i][l] = min last gap for sequence of length l ending at i
    # parent[i][l] = j
    # We use a list of lists.
    # Max possible length is N, but we can cap it.
    # Let's use N+1 to be safe.
    dp = [[INF] * (N + 1) for _ in range(N)]
    parent = [[-1] * (N + 1) for _ in range(N)]
    
    for i in range(N):
        dp[i][1] = 0
        parent[i][1] = -1
        
    best_len = 1
    best_index = 0
    
    # Optimization: Precompute gaps? No, $N^2$ gaps is fine.
    
    for i in range(1, N):
        if time.time() > deadline:
            break
            
        pi = primes[i]
        max_len_at_i = 1
        
        # We only need to check l up to best_len + 1
        # because we can't extend a sequence longer than best_len + 1
        # to get a new best?
        # Actually, we can.
        # But we can limit the inner l loop.
        
        for j in range(i):
            gap = pi - primes[j]
            
            # dp[j][l] is non-decreasing in l
            # We stop when dp[j][l] > gap
            # Also stop if l > best_len (since we can't form l+1 > best_len+1? No)
            # If best_len is 10, and we are at i, we can form 11.
            # We should check l up to best_len.
            
            # Iterate l
            # We know dp[j][1] = 0.
            # If gap >= 0, we can update dp[i][2].
            
            # We can iterate l from 1 to best_len
            # If dp[j][l] <= gap, update dp[i][l+1]
            
            # To speed up, we can find the max l such that dp[j][l] <= gap
            # Since dp[j][l] is sorted, we can binary search?
            # But L is small, linear scan is faster.
            
            for l in range(1, best_len + 1):
                if dp[j][l] > gap:
                    break
                new_len = l + 1
                if gap < dp[i][new_len]:
                    dp[i][new_len] = gap
                    parent[i][new_len] = j
                    if new_len > max_len_at_i:
                        max_len_at_i = new_len
                        
        if max_len_at_i > best_len:
            best_len = max_len_at_i
            best_index = i
            
    # Reconstruct
    if best_len == 1:
        return [primes[best_index]]
        
    path = []
    curr_idx = best_index
    curr_len = best_len
    
    while curr_len > 0:
        path.append(primes[curr_idx])
        if curr_len == 1:
            break
        prev_idx = parent[curr_idx][curr_len]
        curr_idx = prev_idx
        curr_len -= 1
        
    path.reverse()
    return path
