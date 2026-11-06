import time
import math

def construct():
    start_time = time.time()
    limit = 10000
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
        
    primes = [i for i, flag in enumerate(is_prime) if flag]
    N = len(primes)
    
    # dp[j] is a list of (gap, length) pairs, sorted by gap, with strictly increasing lengths
    # This is the Pareto frontier of states ending at prime j.
    dp = [[] for _ in range(N)]
    parent = [[] for _ in range(N)]
    
    for j in range(N):
        dp[j].append((0, 1))
        parent[j].append((-1, -1))
    
    # Precompute gaps
    gap_table = [[0] * N for _ in range(N)]
    for i in range(N):
        pi = primes[i]
        for j in range(i + 1, N):
            gap_table[i][j] = primes[j] - pi
    
    for j in range(N):
        if time.time() - start_time > 55:
            break
        for i in range(j):
            G = gap_table[i][j]
            # Find rightmost index in dp[i] with gap <= G
            lo, hi = 0, len(dp[i]) - 1
            best_idx = -1
            while lo <= hi:
                mid = (lo + hi) // 2
                if dp[i][mid][0] <= G:
                    best_idx = mid
                    lo = mid + 1
                else:
                    hi = mid - 1
            
            if best_idx == -1:
                continue
            
            G_last, L_last = dp[i][best_idx]
            new_L = L_last + 1
            new_G = G
            
            # Check if (new_G, new_L) is dominated in dp[j]
            lo, hi = 0, len(dp[j]) - 1
            pos = -1
            while lo <= hi:
                mid = (lo + hi) // 2
                if dp[j][mid][0] <= new_G:
                    pos = mid
                    lo = mid + 1
                else:
                    hi = mid - 1
            
            if pos != -1 and dp[j][pos][1] >= new_L:
                continue
            
            # Find range to remove from dp[j]: gap >= new_G and length <= new_L
            lo, hi = 0, len(dp[j]) - 1
            start_remove = len(dp[j])
            while lo <= hi:
                mid = (lo + hi) // 2
                if dp[j][mid][0] >= new_G:
                    start_remove = mid
                    hi = mid - 1
                else:
                    lo = mid + 1
            
            lo, hi = start_remove, len(dp[j]) - 1
            end_remove = len(dp[j])
            while lo <= hi:
                mid = (lo + hi) // 2
                if dp[j][mid][1] > new_L:
                    end_remove = mid
                    hi = mid - 1
                else:
                    lo = mid + 1
            
            dp[j] = dp[j][:start_remove] + [(new_G, new_L)] + dp[j][end_remove:]
            parent[j] = parent[j][:start_remove] + [(i, best_idx)] + parent[j][end_remove:]
    
    # Find best
    max_len = 0
    best_j = 0
    best_k = 0
    for j in range(N):
        for k, (g, l) in enumerate(dp[j]):
            if l > max_len:
                max_len = l
                best_j = j
                best_k = k
    
    path = []
    cur_j = best_j
    cur_k = best_k
    while cur_j != -1:
        path.append(cur_j)
        prev_j, prev_k = parent[cur_j][cur_k]
        cur_j = prev_j
        cur_k = prev_k
    
    path.reverse()
    return [primes[idx] for idx in path]
