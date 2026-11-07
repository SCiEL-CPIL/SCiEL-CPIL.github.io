import time
from math import gcd
from functools import lru_cache

# EVOLVE-BLOCK-START
DIVISORS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
N = len(DIVISORS)
MAX_VAL = 1000000
DEADLINE = 85.0  # safety margin from 90s

def _lcm(a, b):
    return a * b // gcd(a, b)

def _all_lcm():
    result = 1
    for d in DIVISORS:
        result = _lcm(result, d)
    return result

LCM_ALL = _all_lcm()  # 5005000

def _matching_exists(L, W):
    """Check if there's a perfect matching: assign each divisor a distinct multiple in [L, L+W]."""
    # Build adjacency: for each divisor i, list of multiples of DIVISORS[i] in [L, L+W]
    # We'll use a simple DFS-based bipartite matching (Kuhn's algorithm)
    # Left side: N divisors (0..N-1)
    # Right side: integers in [L, L+W] that are multiples of at least one divisor
    
    # For efficiency, we only need to consider multiples that appear
    # Build list of candidate values for each divisor
    adj = []
    for i in range(N):
        d = DIVISORS[i]
        # First multiple >= L
        first = ((L + d - 1) // d) * d
        candidates = []
        v = first
        while v <= L + W:
            candidates.append(v)
            v += d
        adj.append(candidates)
    
    # Check if any divisor has no candidates
    for i in range(N):
        if not adj[i]:
            return False, None
    
    # Kuhn's algorithm
    # match_right: maps value -> divisor index assigned to it
    match_right = {}
    
    def try_kuhn(i, visited):
        for v in adj[i]:
            if v in visited:
                continue
            visited.add(v)
            if v not in match_right or try_kuhn(match_right[v], visited):
                match_right[v] = i
                return True
        return False
    
    for i in range(N):
        visited = set()
        if not try_kuhn(i, visited):
            return False, None
    
    # Extract assignment
    assignment = [None] * N
    for v, i in match_right.items():
        assignment[i] = v
    
    return True, assignment

def _check_width(W):
    """Check if there exists a window of width W with a valid assignment."""
    # We need to try different starting positions L
    # Key insight: L only matters modulo the LCM of some subset
    # But LCM_ALL is too large. Instead, note that the pattern of available
    # multiples repeats with period LCM_ALL, but we can't iterate that.
    # 
    # Alternative: for a given W, the set of "useful" L values is limited.
    # Two L values that differ by LCM_ALL give the same pattern.
    # But we can be smarter: for each divisor d_i, the multiples in [L, L+W]
    # depend on L mod d_i. So the overall pattern has period LCM_ALL.
    # 
    # Practical approach: try L in a range that covers one full period of the
    # "critical" structure. Since divisors are small, try L from 0 to some bound.
    # 
    # Better: for a fixed W, if a solution exists at L, then solutions exist
    # at L + k*LCM_ALL. We just need to find ONE L.
    # 
    # Heuristic: try L values that are multiples of small numbers, or just
    # scan L in [0, max_divisor * (W + max_divisor)] which should cover
    # enough variety. Actually, let's try L from 0 to 50000 (as measured
    # in the successful approach).
    
    # The measured approach used up to 50k values of L per W.
    # Let's be a bit more targeted: the period of the "constraint pattern"
    # is LCM_ALL = 5005000, but we don't need to check all of it.
    # 
    # For small W, the number of multiples of d_i in [L, L+W] is at most
    # W/d_i + 1. The assignment feasibility depends on which specific
    # multiples are present, which cycles with period LCM_ALL.
    # 
    # Practical: scan L from 1 to min(50000, LCM_ALL) and check.
    # If W is small enough, 50000 should be sufficient.
    
    max_L = min(50000, LCM_ALL)
    
    for L in range(1, max_L + 1):
        if L + W > MAX_VAL:
            break
        ok, assignment = _matching_exists(L, W)
        if ok:
            return True, assignment, L
    return False, None, None

def construct():
    start_time = time.time()
    
    # Binary search or linear scan on W
    # Start from a reasonable lower bound
    # Each divisor d_i needs at least one multiple in the window.
    # The minimum W must be at least max(0, ...) but practically small.
    
    # Let's try W from 1 upwards, but jump intelligently.
    # The measured best was around 485, so let's start there and search nearby.
    
    best_result = None
    best_W = None
    
    # First, try a range around the known good value
    # Search W from 1 to 1000, checking each
    # But with time guard
    
    for W in range(1, 1001):
        if time.time() - start_time > DEADLINE:
            break
        ok, assignment, L = _check_width(W)
        if ok:
            best_W = W
            best_result = assignment
            # Verify distinctness
            vals = list(assignment)
            if len(set(vals)) == N and all(1 <= v <= MAX_VAL for v in vals):
                break
    
    if best_result is None:
        # Fallback: just return the divisors themselves (valid but not optimal)
        best_result = DIVISORS[:]
    
    # Ensure types are correct
    result = [int(v) for v in best_result]
    return result
# EVOLVE-BLOCK-END
