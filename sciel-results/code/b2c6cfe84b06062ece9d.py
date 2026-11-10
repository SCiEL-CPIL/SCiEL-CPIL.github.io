import math
import time
from functools import lru_cache

def construct():
    """
    Finds m in [0, 1000000] maximizing f(12, m).
    f(12, m) is the least L such that (m, m+L] contains distinct b_1..b_12 with i|b_i.
    """
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    n = 12
    # LCM of 1..12
    K = 1
    for i in range(1, n + 1):
        K = K * i // math.gcd(K, i)
    
    best_m = 0
    best_f = 0
    
    # Precompute for each residue class r in [0, K), the divisibility pattern
    # For each i in 1..12, which positions j in 1..144 have (m+j) % i == 0
    # This is periodic with period K
    
    # For each m in [0, K), compute the "signature" of divisibility
    # and cache the result of _can_match_fast
    
    # We'll use a more efficient approach:
    # For each m, compute the adjacency masks once, then try L from best_f+1 upward
    # (since if L works, all larger L also work)
    
    # Actually, let's be smarter: we know L is at most 144.
    # For each m, we can compute the maximum L directly using a sliding window approach
    # or by binary search.
    
    # Let's use binary search on L for each m, since if L works, L+1 also works.
    # This reduces the number of matching checks from ~144*K to ~7*K.
    
    for m in range(K):
        if time.time() > deadline:
            break
        
        # Binary search for the minimum L such that matching exists
        # If no matching exists for L=144, skip
        lo = 1
        hi = 144
        found = False
        while lo <= hi:
            if time.time() > deadline:
                break
            mid = (lo + hi) // 2
            if _can_match_fast(m, mid, n):
                found = True
                if mid > best_f:
                    best_f = mid
                    best_m = m
                hi = mid - 1
            else:
                lo = mid + 1
        
    return best_m

def _can_match_fast(m, L, n):
    """
    Checks if there is a perfect matching for indices 1..n
    in the interval (m, m+L] using bitmasks with memoization.
    """
    adj_masks = [0] * n
    
    # Precompute masks for each index
    # bit j-1 corresponds to number m+j
    for j in range(1, L + 1):
        num = m + j
        bit = 1 << (j - 1)
        for i in range(1, n + 1):
            if num % i == 0:
                adj_masks[i - 1] |= bit
    
    # Order indices by number of choices (popcount)
    # This is a heuristic to prune the search tree earlier
    indices = list(range(n))
    indices.sort(key=lambda x: bin(adj_masks[x]).count('1'))
    
    # Use a set for visited states to avoid re-exploring
    visited = set()
    
    def dfs(idx, used_mask):
        if idx == n:
            return True
        state = (idx, used_mask)
        if state in visited:
            return False
        visited.add(state)
        
        i = indices[idx]
        avail = adj_masks[i] & ~used_mask
        while avail:
            bit = avail & -avail
            avail ^= bit
            if dfs(idx + 1, used_mask | bit):
                return True
        return False
        
    return dfs(0, 0)
