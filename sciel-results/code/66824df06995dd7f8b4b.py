"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

def _can_match(m, L):
    """Check if (m, m+L] contains 12 distinct b_i with i|b_i."""
    # For each i in 1..12, find all numbers in (m, m+L] divisible by i
    # Build bipartite graph: left = divisors 1..12, right = candidate numbers
    # Use DFS augmenting path matching
    
    # Candidate numbers in (m, m+L]
    # For efficiency, we only need to consider numbers that are divisible by at least one i in 1..12
    # A number x in (m, m+L] is a candidate if gcd-related... actually any x divisible by some i
    
    # For each i, the multiples in (m, m+L]:
    # first multiple > m: ((m // i) + 1) * i
    # last multiple <= m+L: ((m+L) // i) * i
    
    # Build adjacency: for each i, list of candidate numbers divisible by i
    adj = {}
    for i in range(1, 13):
        first = ((m // i) + 1) * i
        last = ((m + L) // i) * i
        if first > last:
            return False  # no multiple of i in range
        adj[i] = list(range(first, last + 1, i))
    
    # Now do bipartite matching: left side = {1,...,12}, right side = all candidate numbers
    # We need to match each i to a distinct number in adj[i]
    
    # Collect all candidate numbers
    all_candidates = set()
    for i in range(1, 13):
        all_candidates.update(adj[i])
    candidates = list(all_candidates)
    
    # Map number to index
    num_to_idx = {x: idx for idx, x in enumerate(candidates)}
    
    # For matching, we track: for each candidate number, which divisor is matched to it
    match_to_num = [-1] * len(candidates)  # match_to_num[j] = i if candidate j is matched to divisor i
    
    def dfs(i, visited):
        for x in adj[i]:
            j = num_to_idx[x]
            if visited[j]:
                continue
            visited[j] = True
            if match_to_num[j] == -1 or dfs(match_to_num[j], visited):
                match_to_num[j] = i
                return True
        return False
    
    # Try to match all 12 divisors
    for i in range(1, 13):
        visited = [False] * len(candidates)
        if not dfs(i, visited):
            return False
    return True

def _f_val(m):
    """Compute f(12, m): least positive L such that (m, m+L] has the matching."""
    # Binary search on L
    lo = 1
    hi = 145  # max is 144 per contract, use 145 as exclusive upper bound
    while lo < hi:
        mid = (lo + hi) // 2
        if _can_match(m, mid):
            hi = mid
        else:
            lo = mid + 1
    return lo

def construct():
    """Find m in [0, 1000000] that maximizes f(12, m)."""
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s limit
    
    best_m = 0
    best_f = 0
    
    # First, do a coarse search over m to find promising regions
    # f(12,m) depends on m mod lcm(1..12) = 27720
    # Actually, the pattern repeats with period lcm(1,2,...,12) = 27720
    # So we only need to check m in [0, 27719]
    
    # Let's verify: f(12, m) depends on which multiples of each i fall in (m, m+L]
    # This is periodic in m with period lcm(1,...,12) = 27720
    
    # So search m in [0, 27719]
    # That's 27720 values, each requiring binary search (~8 iterations) * matching
    # Should be fast enough
    
    for m in range(27720):
        if time.time() > deadline:
            break
        f = _f_val(m)
        if f > best_f:
            best_f = f
            best_m = m
    
    # If we have time, verify and potentially refine
    # But since period is 27720, we've covered all cases
    
    return int(best_m)
# EVOLVE-BLOCK-END
