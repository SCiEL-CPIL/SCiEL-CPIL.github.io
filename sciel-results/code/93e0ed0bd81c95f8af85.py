import time
import sys
from math import gcd

def construct():
    """
    Find m in [0, 1000000] that maximizes f(12, m).
    f(12, m) is the least L > 0 such that (m, m+L] contains 12 distinct integers
    b_1, ..., b_12 with i | b_i for each i in 1..12.
    
    The function has period 27720 (LCM of 1..12).
    We precompute f(12, m) for all m in [0, 27719] using a fast bitset-based
    bipartite matching algorithm, then return the m that maximizes f.
    """
    start_time = time.time()
    deadline = start_time + 85.0
    
    n = 12
    period = 27720
    
    best_m = 0
    best_L = 0
    
    def can_match(m, L):
        """Fast bipartite matching check with degree-ordered DFS using bitmasks."""
        # Build adjacency as bitmasks
        adj = [0] * n
        for i in range(1, n + 1):
            mask = 0
            rem = (-m - 1) % i
            p = rem
            while p < L:
                mask |= (1 << p)
                p += i
            adj[i-1] = mask
            if mask == 0:
                return False
        
        # Order by degree (fewest options first) for better pruning
        order = sorted(range(n), key=lambda idx: bin(adj[idx]).count('1'))
        
        match_r = [-1] * L
        
        def dfs(i, visited_mask):
            mask = adj[i]
            m2 = mask
            while m2:
                lb = m2 & (-m2)
                p = lb.bit_length() - 1
                m2 ^= lb
                bit = 1 << p
                if visited_mask & bit:
                    continue
                visited_mask |= bit
                if match_r[p] == -1 or dfs(match_r[p], visited_mask):
                    match_r[p] = i
                    return True
            return False
        
        for idx in order:
            visited_mask = 0
            if not dfs(idx, visited_mask):
                return False
        return True
    
    for m in range(period):
        if time.time() > deadline:
            break
        
        # Compute lower bound for L
        lb = 12
        for i in range(1, n + 1):
            first_pos = (-m - 1) % i
            if first_pos + 1 > lb:
                lb = first_pos + 1
        
        # Optimization: Only compute exact f(12,m) if it can beat best_L
        if lb > best_L:
            # f(12,m) >= lb > best_L, so it must improve (or tie).
            for L in range(lb, 145):
                if can_match(m, L):
                    if L > best_L:
                        best_L = L
                        best_m = m
                    break
        else:
            # f(12,m) >= lb. But lb <= best_L.
            # Check if f(12,m) > best_L.
            # If can_match(m, best_L) is True, then f(12,m) <= best_L. No improvement.
            # If can_match(m, best_L) is False, then f(12,m) > best_L.
            if not can_match(m, best_L):
                for L in range(best_L + 1, 145):
                    if can_match(m, L):
                        if L > best_L:
                            best_L = L
                            best_m = m
                        break
    
    return int(best_m)

if __name__ == "__main__":
    construct()
