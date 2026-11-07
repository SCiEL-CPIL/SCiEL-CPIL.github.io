import time
from itertools import product

DEADLINE = 85.0  # safety margin before 90s kill
_start_time = time.time()

def _remaining():
    return DEADLINE - (time.time() - _start_time)

def _has_matching(m, L):
    """Check if (m, m+L] contains distinct b_1..b_12 with i | b_i."""
    # For each i in 1..12, collect multiples of i in (m, m+L]
    # Use bipartite matching: left = i in 1..12, right = candidate numbers
    # We'll use DFS-based bipartite matching (Kuhn's algorithm)
    
    # For each i, the multiples of i in (m, m+L]:
    # first multiple of i > m is m//i * i + i (if m%i==0 then m+i else (m//i+1)*i)
    # We need to find a matching.
    
    # Build adjacency: for each i, list of candidate b values (multiples of i in (m, m+L])
    # But the number of candidates can be large. Better: for each i, the candidates are
    # specific values. We need to assign distinct b_i to each i.
    
    # Since L <= 144, the interval (m, m+L] has at most 144 integers.
    # For each integer x in (m, m+L], which i's can use it? i must divide x.
    # So build: for each i in 1..12, the set of x in (m+1, ..., m+L) with x % i == 0.
    
    # Kuhn's algorithm: match[i] = which x is assigned to i
    # adj[i] = list of x that i can take
    
    adj = [[] for _ in range(13)]  # 1-indexed, i=1..12
    for i in range(1, 13):
        # multiples of i in (m, m+L]
        first = (m // i + 1) * i  # smallest multiple of i > m
        if first <= m:
            first += i
        for x in range(first, m + L + 1, i):
            adj[i].append(x)
    
    # Now bipartite matching: left nodes 1..12, right nodes are the x values
    # Use DFS augmenting path
    match_x = {}  # x -> i that is matched to x
    
    def dfs(i, visited):
        for x in adj[i]:
            if x in visited:
                continue
            visited.add(x)
            if x not in match_x or dfs(match_x[x], visited):
                match_x[x] = i
                return True
        return False
    
    count = 0
    for i in range(1, 13):
        visited = set()
        if dfs(i, visited):
            count += 1
    
    return count == 12

def _f_12(m):
    """Compute f(12, m): least L such that (m, m+L] has a valid matching."""
    # Binary search on L. L ranges from 1 to 144.
    lo, hi = 1, 144
    # Check if L=144 works
    if not _has_matching(m, 144):
        return 144  # should not happen, but safety
    while lo < hi:
        mid = (lo + hi) // 2
        if _has_matching(m, mid):
            hi = mid
        else:
            lo = mid + 1
    return lo

def construct():
    best_m = 12
    best_L = _f_12(12)
    
    # Search strategy: 
    # 1. Try m values near multiples of 27720 (lcm(1..12))
    # 2. Random search
    # 3. Local refinement
    
    import random
    random.seed(42)
    
    candidates = set()
    
    # Around multiples of 27720
    for k in range(0, 37):  # 27720 * 37 ~ 1025640
        base = 27720 * k
        for offset in range(-20, 21):
            m = base + offset
            if 0 <= m <= 1000000:
                candidates.add(m)
    
    # Also try m = 0, and small values
    for m in range(0, 100):
        candidates.add(m)
    
    # Random candidates
    for _ in range(5000):
        candidates.add(random.randint(0, 1000000))
    
    # Evaluate candidates, keeping best
    # Sort by some heuristic? Just evaluate all if time permits
    # Prioritize: evaluate candidates near multiples of 27720 first (likely worst cases)
    
    # Actually, let's think: the worst interval is one where the matching fails for L=143 but works for L=144.
    # This happens when the "bottleneck" i (say i=12) has its first available multiple just at m+144.
    # So m+144 should be a multiple of 12, and (m, m+143] should not contain enough distinct multiples.
    # m+144 = 12k => m = 12k - 144. Also need that in (m, m+143], the multiples of 12 are at 12(k-12), 12(k-11), ..., 12(k-1) = m+132.
    # Wait, m = 12k - 144, so m+1 = 12k - 143. First multiple of 12 > m is 12k - 144 + 12 = 12k - 132? No.
    # m = 12k - 144. Multiples of 12 > m: 12k - 132, 12k - 120, ..., 12k. In (m, m+143] = (12k-144, 12k-1].
    # So multiples of 12 in that range: 12k-132, 12k-120, ..., 12k-12. That's 11 multiples.
    # For i=11: multiples in (12k-144, 12k-1]. Depends on k.
    
    # Let me just search m = 12k - 144 for various k, and also nearby.
    for k in range(12, 100000):
        m = 12 * k - 144
        if 0 <= m <= 1000000:
            candidates.add(m)
        if _remaining() < 5.0:
            break
    
    # Also m = 11k - 144, etc. for each i
    for i in range(1, 13):
        for k in range(0, 1000000 // i + 1):
            m = i * k - 144
            if 0 <= m <= 1000000:
                candidates.add(m)
            if _remaining() < 5.0:
                break
    
    # Convert to list and evaluate
    cand_list = list(candidates)
    # Shuffle to get variety, but prioritize the structured ones
    # Actually, let's just evaluate in order, checking time
    
    for m in cand_list:
        if _remaining() < 2.0:
            break
        L = _f_12(m)
        if L > best_L:
            best_L = L
            best_m = m
            if best_L == 144:
                break
    
    # Local refinement around best_m
    if best_L < 144 and _remaining() > 3.0:
        for dm in range(-50, 51):
            m2 = best_m + dm
            if 0 <= m2 <= 1000000 and m2 != best_m:
                L2 = _f_12(m2)
                if L2 > best_L:
                    best_L = L2
                    best_m = m2
                    if best_L == 144:
                        break
    
    return int(best_m)
