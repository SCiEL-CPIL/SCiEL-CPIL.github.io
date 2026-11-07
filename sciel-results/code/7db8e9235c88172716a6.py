"""Rewritten program: search for m maximizing f(12,m) via bipartite matching + structured search."""
import time
import math
from collections import deque

# EVOLVE-BLOCK-START
N = 12
MAX_M = 1000000
DEADLINE = 85  # seconds, safety margin under 90s

def _can_match(m, L):
    """Check if (m, m+L] contains distinct b_1..b_12 with i | b_i.
    Uses bipartite matching (Hopcroft-Karp style or DFS augmenting path).
    Left side: indices 1..12. Right side: values m+1 .. m+L.
    Edge from i to v if v % i == 0.
    """
    # Build adjacency: for each i in 1..12, list of values in (m, m+L] divisible by i
    # Values are m+1 .. m+L
    # We need a matching that covers all 12 left nodes.
    
    # For efficiency, map each value to an index 0..L-1
    # adj[i] = list of value-indices (0-based) where value = m+1+j and (m+1+j) % i == 0
    
    adj = [[] for _ in range(N + 1)]  # 1-indexed left
    for i in range(1, N + 1):
        # values v in (m, m+L] i.e., v = m+1, m+2, ..., m+L
        # v % i == 0
        # first v >= m+1 with v % i == 0:
        # v = m+1, m+2, ...
        # We want v ≡ 0 (mod i)
        # The smallest v >= m+1 divisible by i:
        r = (m + 1) % i
        if r == 0:
            first = m + 1
        else:
            first = m + 1 + (i - r)
        if first > m + L:
            continue
        for v in range(first, m + L + 1, i):
            j = v - (m + 1)  # 0-based index
            adj[i].append(j)
    
    # Now run bipartite matching: left size N=12, right size L
    # match_right[j] = which left node is matched to right node j, or -1
    match_right = [-1] * L
    
    def dfs(u, visited):
        for j in adj[u]:
            if visited[j]:
                continue
            visited[j] = True
            if match_right[j] == -1 or dfs(match_right[j], visited):
                match_right[j] = u
                return True
        return False
    
    count = 0
    for u in range(1, N + 1):
        visited = [False] * L
        if dfs(u, visited):
            count += 1
        else:
            return False  # early exit: can't match u
    return count == N

def _f(m):
    """Compute f(12, m): least positive L such that (m, m+L] has distinct b_i with i|b_i."""
    # Binary search on L. Lower bound: at least N (need 12 distinct values).
    # Upper bound: 144 per problem statement.
    lo = N
    hi = 144
    while lo < hi:
        mid = (lo + hi) // 2
        if _can_match(m, mid):
            hi = mid
        else:
            lo = mid + 1
    return lo

def construct():
    start_time = time.time()
    
    best_m = 12
    best_f = _f(12)
    
    def check_deadline():
        return time.time() - start_time < DEADLINE
    
    # Phase 1: Try structured candidates near multiples of lcm(1..12) = 27720
    lcm_12 = 27720
    # Also try small m values
    candidates = set()
    
    # Small m
    for m in range(0, 200):
        candidates.add(m)
    
    # Near multiples of 27720
    for k in range(0, MAX_M // lcm_12 + 1):
        base = k * lcm_12
        for offset in range(-50, 51):
            m = base + offset
            if 0 <= m <= MAX_M:
                candidates.add(m)
    
    # Also try values near lcm*2, lcm*3, etc. with larger offsets
    for k in range(1, min(40, MAX_M // lcm_12 + 1)):
        base = k * lcm_12
        for offset in range(-200, 201):
            m = base + offset
            if 0 <= m <= MAX_M:
                candidates.add(m)
    
    # Sort candidates for deterministic evaluation
    candidates = sorted(candidates)
    
    # Evaluate candidates
    for m in candidates:
        if not check_deadline():
            break
        try:
            f_val = _f(m)
            if f_val > best_f:
                best_f = f_val
                best_m = m
        except Exception:
            pass
    
    # Phase 2: If time remains, do a broader random-ish search
    if check_deadline():
        import random
        rng = random.Random(42)
        # Generate more candidates: random multiples + offsets, and random values
        additional = set()
        for _ in range(5000):
            if not check_deadline():
                break
            # Random m in [0, MAX_M]
            m = rng.randint(0, MAX_M)
            additional.add(m)
        # Also try near lcm multiples with random offsets
        for k in range(0, min(100, MAX_M // lcm_12 + 1)):
            base = k * lcm_12
            for _ in range(20):
                if not check_deadline():
                    break
                offset = rng.randint(-1000, 1000)
                m = base + offset
                if 0 <= m <= MAX_M:
                    additional.add(m)
        
        additional = sorted(additional)
        for m in additional:
            if not check_deadline():
                break
            if m in candidates:
                continue
            try:
                f_val = _f(m)
                if f_val > best_f:
                    best_f = f_val
                    best_m = m
            except Exception:
                pass
    
    # Phase 3: Local refinement around best_m
    if check_deadline():
        for delta in range(-200, 201):
            if not check_deadline():
                break
            m = best_m + delta
            if 0 <= m <= MAX_M:
                try:
                    f_val = _f(m)
                    if f_val > best_f:
                        best_f = f_val
                        best_m = m
                except Exception:
                    pass
    
    # Ensure we return a Python int
    return int(best_m)
# EVOLVE-BLOCK-END
