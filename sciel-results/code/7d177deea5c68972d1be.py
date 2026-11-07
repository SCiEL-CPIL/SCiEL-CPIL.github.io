import time
import math
from functools import reduce

def _lcm(a, b):
    return a * b // math.gcd(a, b)

# Precomputation for speed
# For each i in 1..12, and each residue r in 0..i-1,
# store the list of x in [1, 144] such that x % i == r
# Note: x must be > 0.
# If r == 0, x % i == 0 means x is multiple of i.
# The condition is x == -m (mod i).
# So r = (-m) % i.
# We need x in [1, L] such that x % i == r.

MAX_L = 144
N = 12

# precomp[i][r] = list of x in [1, MAX_L] such that x % i == r
precomp = [[[] for _ in range(i)] for i in range(N + 1)]
for i in range(1, N + 1):
    for r in range(i):
        # x = r, r+i, r+2i...
        # but x must be >= 1
        # if r == 0, x starts at i
        # if r > 0, x starts at r
        start = r if r > 0 else i
        for x in range(start, MAX_L + 1, i):
            precomp[i][r].append(x)

def _bipartite_match(adj, n_left, n_right):
    """Kuhn's algorithm for bipartite matching. adj[i] = list of right-node indices."""
    match_r = [-1] * n_right  # match_r[j] = left node matched to right j, or -1
    
    def try_kuhn(u, visited):
        for v in adj[u]:
            if visited[v]:
                continue
            visited[v] = True
            if match_r[v] == -1 or try_kuhn(match_r[v], visited):
                match_r[v] = u
                return True
        return False
    
    for u in range(n_left):
        visited = [False] * n_right
        try_kuhn(u, visited)
    
    return sum(1 for v in match_r if v != -1)

def _can_match(m, L, n=12):
    if L < n:
        return False
        
    # Build adjacency lists using precomputed lists
    # We need to map x values to right-node indices.
    # The right nodes are the specific x values used in the matching.
    # However, different i's might use the same x value.
    # So we need a global mapping of x -> index.
    
    val_to_idx = {}
    right_vals = []
    adj_idx = [[] for _ in range(n)]
    
    for i in range(n):
        idx = i + 1
        r = (-m) % idx
        
        # Get valid x's from precomp, filter by x <= L
        # precomp[idx][r] is sorted
        # We can just iterate and break when x > L
        valid_xs = []
        for x in precomp[idx][r]:
            if x > L:
                break
            valid_xs.append(x)
            
        if not valid_xs:
            return False
            
        for x in valid_xs:
            if x not in val_to_idx:
                val_to_idx[x] = len(right_vals)
                right_vals.append(x)
            adj_idx[i].append(val_to_idx[x])
            
    if len(right_vals) < n:
        return False
    
    # Quick check: if any left node has no edges, fail
    for i in range(n):
        if not adj_idx[i]:
            return False
            
    return _bipartite_match(adj_idx, n, len(right_vals)) == n

def _f(m, n=12):
    # Binary search smallest L in [n, MAX_L]
    lo, hi = n, MAX_L
    while lo < hi:
        mid = (lo + hi) // 2
        if _can_match(m, mid, n):
            hi = mid
        else:
            lo = mid + 1
    return lo

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    # Calculate LCM of 1..12
    p = reduce(_lcm, range(1, 13))
    
    best_m = 0
    best_f = _f(0)
    
    # Generate candidates: m where m+1 is divisible by at least one of 2..12
    # Sort by descending difficulty
    scores = []
    for m in range(p):
        mp1 = m + 1
        cnt = 0
        for i in range(2, 13):
            if mp1 % i == 0:
                cnt += 1
        scores.append((cnt, m))
    
    scores.sort(key=lambda x: -x[0])
    
    for cnt, m in scores:
        if time.time() > deadline:
            break
        f_val = _f(m)
        if f_val > best_f:
            best_f = f_val
            best_m = m
        if best_f >= MAX_L:
            break
    
    return int(best_m)
