import heapq
import bisect
from math import isqrt

def construct():
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, isqrt(n) + 1):
            if is_prime[i]:
                for j in range(i * i, n + 1, i):
                    is_prime[j] = False
        return [i for i in range(2, n + 1) if is_prime[i]]
    
    primes = sieve(10000)
    n = len(primes)
    
    # best[i] = list of (gap, length) non-dominated, sorted by gap
    best = [[] for _ in range(n)]
    # parent[i][idx_in_best] = (prev_prime_idx, prev_gap) or None
    parent = [dict() for _ in range(n)]  # key: (gap, length) -> (prev_idx, prev_gap)
    
    heap = []
    for i, p in enumerate(primes):
        heapq.heappush(heap, (0, -1, i))
        best[i].append((0, 1))
        parent[i][(0, 1)] = None
    
    max_len = 1
    max_prime_idx = 0
    max_gap = 0
    
    while heap:
        g, neg_L, i = heapq.heappop(heap)
        L = -neg_L
        p = primes[i]
        
        # Verify state is still valid
        if (g, L) not in parent[i]:
            continue
        
        target = p + g
        j = bisect.bisect_left(primes, target, i + 1)
        
        for k in range(j, n):
            q = primes[k]
            new_gap = q - p
            new_len = L + 1
            new_key = (new_gap, new_len)
            
            # Check if dominated
            dominated = False
            for bg, bL in best[k]:
                if bg <= new_gap and bL >= new_len:
                    dominated = True
                    break
            if dominated:
                continue
            
            # Remove dominated states
            new_best = []
            for bg, bL in best[k]:
                if not (new_gap <= bg and new_len >= bL):
                    new_best.append((bg, bL))
            new_best.append(new_key)
            new_best.sort(key=lambda x: x[0])
            best[k] = new_best
            
            parent[k][new_key] = (i, g)
            
            if new_len > max_len:
                max_len = new_len
                max_prime_idx = k
                max_gap = new_gap
            
            heapq.heappush(heap, (new_gap, -new_len, k))
    
    # Backtrack
    if max_len == 1:
        return [primes[max_prime_idx]]
    
    seq = []
    idx = max_prime_idx
    gap = max_gap
    while parent[idx].get((gap, len(seq) if seq else None)) is not None or (gap, max_len - len(seq)) in parent[idx]:
        # This is getting messy. Let me use a cleaner backtrack.
        break
    
    # Cleaner backtrack:
    seq = []
    cur_idx = max_prime_idx
    cur_gap = max_gap
    cur_len = max_len
    while cur_len > 0:
        seq.append(primes[cur_idx])
        if cur_len == 1:
            break
        prev_key = parent[cur_idx].get((cur_gap, cur_len))
        if prev_key is None:
            break
        prev_idx, prev_gap = prev_key
        cur_idx = prev_idx
        cur_gap = prev_gap
        cur_len -= 1
    
    seq.reverse()
    return seq

