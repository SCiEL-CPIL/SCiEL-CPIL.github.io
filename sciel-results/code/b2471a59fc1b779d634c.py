"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def _can_match(m, L):
    """Check if (m, m+L] contains distinct b_1..b_12 with i|b_i."""
    # For each i in 1..12, find multiples of i in (m, m+L]
    # Build bipartite matching: left = i in 1..12, right = candidate numbers
    # Use simple DFS-based bipartite matching
    n = 12
    # candidate[i] = list of numbers in (m, m+L] divisible by i
    candidates = []
    for i in range(1, n + 1):
        # first multiple of i > m
        first = (m // i + 1) * i
        cands = []
        x = first
        while x <= m + L:
            cands.append(x)
            x += i
        candidates.append(cands)
    
    # Bipartite matching: match each i to a distinct number
    # match_to_num[num] = i that is matched to num
    # We'll use standard Kuhn's algorithm
    # Since numbers can be up to m+144, use a dict
    match_to_i = {}  # number -> i
    
    def dfs(i, visited):
        for num in candidates[i]:
            if num in visited:
                continue
            visited.add(num)
            if num not in match_to_i or dfs(match_to_i[num], visited):
                match_to_i[num] = i
                return True
        return False
    
    for i in range(n):
        visited = set()
        if not dfs(i, visited):
            return False
    return True

def _f_12(m):
    """Compute f(12, m): least positive L such that (m, m+L] has distinct b_i with i|b_i."""
    # Binary search on L in [1, 144]
    lo, hi = 1, 144
    while lo < hi:
        mid = (lo + hi) // 2
        if _can_match(m, mid):
            hi = mid
        else:
            lo = mid + 1
    return lo

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s kill
    
    best_m = 0
    best_f = 0
    
    # The function f(12, m) is periodic with period lcm(1..12) = 27720
    # because divisibility by i depends on m mod i, and lcm(1..12)=27720
    # So we only need to check m in [0, 27719]
    # But m can be up to 1000000, and f(12, m) depends only on m mod 27720
    # Actually, let's verify: the condition is that (m, m+L] contains multiples of i.
    # The set of multiples of i in (m, m+L] depends on m mod i.
    # Since lcm(1..12) = 27720, the pattern repeats every 27720.
    
    period = 27720
    # Search all residues
    for m in range(period):
        if time.time() > deadline:
            break
        f_val = _f_12(m)
        if f_val > best_f:
            best_f = f_val
            best_m = m
    
    # Return best_m (any m in [0, 1000000] with the same residue works)
    # Make sure best_m is in range
    result = int(best_m % 1000000)
    if result > 1000000:
        result = 0
    return result
# EVOLVE-BLOCK-END
