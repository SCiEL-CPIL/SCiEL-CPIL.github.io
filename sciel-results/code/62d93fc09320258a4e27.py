import time
import sys
import math
from functools import reduce

def _lcm(a, b):
    return a * b // math.gcd(a, b)

def _can_match(m, L, n=12):
    """Check if (m, m+L] contains distinct b_1..b_n with i | b_i using bipartite matching."""
    adj = [[] for _ in range(n)]
    end = m + L
    
    for i in range(1, n + 1):
        # First multiple of i strictly greater than m
        r = ((m // i) + 1) * i
        if r > end:
            return False
        v = r
        while v <= end:
            adj[i - 1].append(v)
            v += i
            
    # Sort left nodes by number of neighbors (heuristic for DFS pruning)
    # We need to preserve the original index for matching logic
    order = sorted(range(n), key=lambda i: len(adj[i]))
    
    # match_r maps a value (right node) to the left node index that is currently matched to it
    match_r = {}
    
    sys.setrecursionlimit(10000)
    
    def dfs(u, visited):
        for v in adj[u]:
            if v in visited:
                continue
            visited.add(v)
            if v not in match_r or dfs(match_r[v], visited):
                match_r[v] = u
                return True
        return False
        
    for u in order:
        # Use a set for visited values in each DFS call
        if not dfs(u, set()):
            return False
            
    return True

def _f(m, n=12):
    """Find smallest L in [1, 144] such that (m, m+L] has valid matching."""
    lo, hi = 1, 144
    
    # Quick check: if L=144 doesn't work, return 144 (max possible)
    if not _can_match(m, hi, n):
        return hi
        
    while lo < hi:
        mid = (lo + hi) // 2
        if _can_match(m, mid, n):
            hi = mid
        else:
            lo = mid + 1
            
    return lo

def construct():
    deadline = time.time() + 85  # 85s budget, 5s safety margin
    
    # LCM of 1..12 = 27720; f(m) is periodic with this period
    period = reduce(_lcm, range(1, 13))
    
    best_residue = 0
    best_f = 0
    
    # Evaluate f at each m in the period
    # We can skip some if time is running out, but we want the global max
    for r in range(period):
        if time.time() > deadline - 5:
            break
        f_val = _f(r)
        if f_val > best_f:
            best_f = f_val
            best_residue = r
            
    # Find the largest m <= 1000000 with m % period == best_residue
    if best_residue <= 1000000:
        k_max = (1000000 - best_residue) // period
        best_m = best_residue + k_max * period
    else:
        best_m = 0
        
    # Local refinement: check neighbors of best_m
    if time.time() < deadline - 5:
        candidates = set()
        candidates.add(best_m)
        
        # Check immediate neighbors
        for delta in range(-10, 11):
            mm = best_m + delta
            if 0 <= mm <= 1000000:
                candidates.add(mm)
                
        # Also check other high-scoring residues if time permits
        # We don't store them, so we re-check a few around best_residue
        for delta in range(-10, 11):
            rr = (best_residue + delta) % period
            if rr <= 1000000:
                k_max = (1000000 - rr) // period
                mm = rr + k_max * period
                candidates.add(mm)

        for mm in sorted(candidates):
            if time.time() > deadline - 3:
                break
            f_val = _f(mm)
            if f_val > best_f:
                best_f = f_val
                best_m = mm

    return int(best_m)
