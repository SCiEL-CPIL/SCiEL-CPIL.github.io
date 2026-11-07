"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from itertools import product

P = {'n': 12, 'maximum_start': 1000000}
N = 12
MAX_START = 1000000

def _can_match(L, m):
    """
    Check if interval (m, m+L] contains distinct b_1..b_12 with i | b_i.
    Uses bipartite matching via DFS (Hopcroft-Karp would be overkill for 12x12).
    """
    # For each i in 1..12, find all candidates in (m, m+L] divisible by i
    # Candidates: numbers k in (m, m+L] such that i | k
    # We need to assign distinct candidates to each i.
    
    # Build adjacency: for each i, list of candidate values
    # To keep it small, we only need at most 12 candidates per i (actually we can limit to first few)
    # But let's be safe: collect all candidates up to some reasonable limit
    
    # Actually, the number of candidates for i in (m, m+L] is about L/i.
    # For i=1, that's L candidates. L can be up to 144.
    # For matching, we can use a simple DFS-based bipartite matching.
    
    # Let's build the bipartite graph: left side = indices 1..12, right side = candidate values
    # But candidate values can be many. Instead, we can think of it as:
    # For each i, the candidates are {k : m < k <= m+L, i|k}
    
    # A simpler approach: since we only need 12 distinct values, and each i has specific candidates,
    # we can use a backtracking search with pruning, or a proper bipartite matching.
    
    # Let's use a standard bipartite matching (Kuhn's algorithm) on a graph where
    # left nodes are i=1..12 and right nodes are candidate values.
    # But the right side can be large. Instead, note that for each i, candidates are spaced by i.
    
    # Alternative: formulate as exact cover or use a small DFS.
    # Since n=12 is small, let's try a DFS with memoization on which i's are assigned.
    
    # Better: use a proper bipartite matching. The right side values are specific integers.
    # Let's collect all unique candidate values across all i.
    
    # Actually, the simplest correct approach: 
    # For each i, candidates are numbers in (m, m+L] divisible by i.
    # We need to pick one distinct candidate for each i.
    
    # Let's build adjacency list for i=1..12
    adj = [[] for _ in range(N + 1)]  # 1-indexed
    for i in range(1, N + 1):
        # First candidate > m divisible by i: ceil((m+1)/i) * i
        first = ((m + 1 + i - 1) // i) * i
        # Actually: first = ((m + 1) + i - 1) // i * i = ((m + i) // i) * i
        first = ((m + 1) + i - 1) // i * i
        if first <= m:
            first += i
        # Ensure first > m
        while first <= m:
            first += i
        k = first
        while k <= m + L:
            adj[i].append(k)
            k += i
    
    # Now do bipartite matching: left = 1..12, right = candidate values
    # Use Kuhn's algorithm (DFS-based)
    # match[i] = candidate value assigned to i, or 0 if unassigned
    # match_val[v] = which i is matched to value v
    
    # To keep the right side manageable, note that values are in (m, m+L]
    # We can use a dict for match_val
    
    match_val = {}  # value -> i
    match_i = [0] * (N + 1)  # i -> value
    
    def dfs(i, visited):
        for v in adj[i]:
            if visited.get(v, False):
                continue
            visited[v] = True
            if v not in match_val or dfs(match_val[v], visited):
                match_val[v] = i
                match_i[i] = v
                return True
        return False
    
    # Try to match all i from 1 to 12
    for i in range(1, N + 1):
        visited = {}
        if not dfs(i, visited):
            return False
    return True

def _f(m):
    """
    Compute f(12, m): the LEAST positive L such that (m, m+L] contains
    distinct b_1..b_12 with i | b_i.
    """
    # L must be at least 12 (need 12 distinct numbers in the interval)
    # Binary search on L: if _can_match(L, m) is True, then f <= L
    # f is the minimum L such that _can_match(L, m) is True.
    # _can_match is monotone: if True for L, then True for any L' >= L.
    
    # Lower bound: 12 (need 12 distinct integers)
    # Upper bound: 144 (given in problem statement)
    lo = 12
    hi = 144
    
    # Check if hi is sufficient
    if not _can_match(hi, m):
        # Should not happen per problem statement, but return hi as fallback
        return hi
    
    while lo < hi:
        mid = (lo + hi) // 2
        if _can_match(mid, m):
            hi = mid
        else:
            lo = mid + 1
    return lo

def construct():
    """
    Find m in [0, 1000000] that maximizes f(12, m).
    """
    start_time = time.time()
    deadline = start_time + 75  # Leave 15s safety margin
    
    best_m = 12
    best_f = _f(best_m)
    
    # Strategy: 
    # 1. Sample some m values to get a sense of the landscape
    # 2. Try to find m that maximizes f
    
    # First, let's understand the structure.
    # f(m) = min L such that (m, m+L] has distinct multiples for 1..12.
    # To maximize f(m), we want m such that for small L, the interval (m, m+L] 
    # fails to contain a valid matching.
    
    # A key insight: if m is such that many of the "easy" multiples are just outside 
    # the interval, f(m) will be larger.
    
    # Let's do a randomized search + local improvement.
    
    import random
    random.seed(42)
    
    # Try some structured candidates
    candidates = set()
    
    # Small m values
    for m in range(0, 200):
        candidates.add(m)
    
    # Random samples
    for _ in range(2000):
        candidates.add(random.randint(0, MAX_START))
    
    # Some structured: near multiples of LCM(1..12) = 27720
    lcm_12 = 27720
    for k in range(0, MAX_START // lcm_12 + 2):
        for offset in range(-50, 51):
            m = k * lcm_12 + offset
            if 0 <= m <= MAX_START:
                candidates.add(m)
    
    # Evaluate all candidates
    for m in candidates:
        if time.time() > deadline - 10:
            break
        f_val = _f(m)
        if f_val > best_f:
            best_f = f_val
            best_m = m
    
    # Local search around best_m
    if time.time() < deadline - 5:
        # Try neighbors of best_m
        for delta in range(-100, 101):
            if time.time() > deadline - 5:
                break
            m = best_m + delta
            if 0 <= m <= MAX_START:
                f_val = _f(m)
                if f_val > best_f:
                    best_f = f_val
                    best_m = m
    
    # More extensive local search
    if time.time() < deadline - 3:
        # Try a wider neighborhood
        for delta in range(-500, 501, 2):
            if time.time() > deadline - 3:
                break
            m = best_m + delta
            if 0 <= m <= MAX_START:
                f_val = _f(m)
                if f_val > best_f:
                    best_f = f_val
                    best_m = m
                    # Restart local search from new best
                    for delta2 in range(-100, 101):
                        if time.time() > deadline - 3:
                            break
                        m2 = m + delta2
                        if 0 <= m2 <= MAX_START:
                            f_val2 = _f(m2)
                            if f_val2 > best_f:
                                best_f = f_val2
                                best_m = m2
    
    # Final check
    return int(best_m)

# EVOLVE-BLOCK-END
