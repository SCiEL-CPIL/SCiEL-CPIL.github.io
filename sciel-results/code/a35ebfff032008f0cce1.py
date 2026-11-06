import time
from bisect import bisect_right

def construct():
    t0 = time.time()
    DEADLINE = 80.0

    N = 10000
    is_prime = bytearray(b'\x01') * (N + 1)
    is_prime[0] = is_prime[1] = 0
    for x in range(2, int(N ** 0.5) + 1):
        if is_prime[x]:
            is_prime[x * x::x] = bytearray(len(is_prime[x * x::x]))
    primes = [i for i in range(2, N + 1) if is_prime[i]]
    n = len(primes)

    # dp[i] is the Pareto frontier of (last_gap, length, prev_idx, prev_entry_idx)
    # sorted by last_gap ascending. Lengths are strictly increasing.
    dp = [[] for _ in range(n)]
    
    # Helper to reduce candidates to Pareto frontier
    def reduce_to_frontier(cands):
        if not cands:
            return []
        # Group by last_gap, keep max length for each gap
        # cands is list of (gap, len, prev_idx, prev_entry_idx)
        best_by_gap = {}
        for g, l, pi, pei in cands:
            if g not in best_by_gap or l > best_by_gap[g][0]:
                best_by_gap[g] = (l, pi, pei)
        
        # Sort by gap
        sorted_items = sorted(best_by_gap.items())
        
        frontier = []
        max_len_so_far = 0
        for g, (l, pi, pei) in sorted_items:
            if l > max_len_so_far:
                frontier.append((g, l, pi, pei))
                max_len_so_far = l
            # else dominated by a previous entry with smaller gap and >= length
        return frontier

    for i in range(n):
        if time.time() - t0 >= DEADLINE:
            break
        pi = primes[i]
        cands = []
        for j in range(i):
            if time.time() - t0 >= DEADLINE:
                break
            gap = pi - primes[j]
            # Option 1: Start new sequence
            cands.append((gap, 2, j, -1))
            
            # Option 2: Extend existing sequence from j
            if dp[j]:
                # dp[j] is sorted by last_gap
                # Find largest last_gap <= gap
                # We can use bisect on a list of gaps
                # To avoid creating a list of gaps every time, we can just binary search manually or use bisect on a precomputed list if we stored it.
                # But dp[j] is a list of tuples.
                # Let's extract gaps for binary search? That's O(K).
                # Alternatively, since K is small, linear scan might be faster than creating list + bisect?
                # Or we can store gaps separately?
                # Let's just do a binary search on the list of tuples by accessing [0].
                # Python's bisect doesn't support key function in older versions, but we can use a helper or just loop if K is small.
                # Given N=1229, average j is 600. Total ops ~ 750k.
                # If we do O(K) scan, and K is ~10-20, that's 10M-15M ops. Fine.
                # If we do O(log K), it's faster.
                
                # Let's implement binary search for max index k such that dp[j][k][0] <= gap
                lo, hi = 0, len(dp[j]) - 1
                best_k = -1
                while lo <= hi:
                    mid = (lo + hi) // 2
                    if dp[j][mid][0] <= gap:
                        best_k = mid
                        lo = mid + 1
                    else:
                        hi = mid - 1
                
                if best_k != -1:
                    lg, ln, pj, pei = dp[j][best_k]
                    cands.append((gap, ln + 1, j, best_k))
        
        dp[i] = reduce_to_frontier(cands)

    # Find best
    best_len = 1
    best_i = 0
    best_k = -1
    for i in range(n):
        for k, (g, l, pi, pei) in enumerate(dp[i]):
            if l > best_len:
                best_len = l
                best_i = i
                best_k = k

    # Reconstruct
    if best_len <= 1:
        return [primes[0]]

    result = []
    cur_i = best_i
    cur_k = best_k
    while True:
        result.append(primes[cur_i])
        if cur_k == -1:
            break
        g, l, pi, pei = dp[cur_i][cur_k]
        cur_i = pi
        cur_k = pei
    
    result.reverse()
    return result
