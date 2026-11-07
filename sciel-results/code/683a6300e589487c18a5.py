"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from collections import deque

P = {'n': 12, 'maximum_start': 1000000}

def _has_matching(m, L, n=12):
    """Check if (m, m+L] contains distinct b_1,...,b_n with i | b_i."""
    # For each i in 1..n, the available candidates are numbers in (m, m+L] divisible by i
    # We need to check if there's a perfect matching in the bipartite graph
    # where left side is indices 1..n and right side is candidate numbers
    
    # For efficiency, we'll use a greedy/Hall's theorem approach or direct matching
    # Since n=12 is small, we can use backtracking with pruning
    
    # First, build the candidate lists
    candidates = []
    for i in range(1, n + 1):
        # Numbers in (m, m+L] divisible by i
        start = m + 1
        end = m + L
        # First multiple of i >= start
        first = ((start + i - 1) // i) * i
        # Last multiple of i <= end
        last = (end // i) * i
        if first <= last:
            candidates.append(list(range(first, last + 1, i)))
        else:
            candidates.append([])
    
    # Quick check: if any index has no candidates, fail
    for i in range(n):
        if not candidates[i]:
            return False
    
    # Use backtracking with memoization or simple DFS
    # Since n=12, we can do DFS with bitmasks on used numbers
    # But the numbers can be large, so let's use a different approach
    
    # Actually, let's use the Hungarian algorithm or a simple matching
    # For n=12, we can do DFS with pruning
    
    # Let's use a greedy approach first: sort by number of candidates (most constrained first)
    # Then try to assign
    
    # Better: use the fact that we just need existence, not the actual matching
    # Use Hall's condition or just do DFS
    
    # DFS approach: assign numbers to indices 1..n
    # State: which numbers are used
    # Since numbers can be up to m+L ~ 1000000+144, we can't use bitmask on numbers directly
    # Instead, we'll track used numbers in a set
    
    # But DFS might be slow. Let's use a smarter approach.
    
    # Actually, for each i, the candidates are multiples of i in (m, m+L].
    # Two indices i and j can share a candidate only if that number is divisible by both i and j,
    # i.e., divisible by lcm(i,j).
    
    # Let's use a simple recursive backtracking with ordering by scarcity.
    
    # Order indices by number of candidates (ascending)
    order = sorted(range(n), key=lambda i: len(candidates[i]))
    
    # If any has 0 candidates, already handled
    if len(candidates[order[0]]) == 0:
        return False
    
    # Backtracking
    used = set()
    
    def dfs(idx):
        if idx == n:
            return True
        i = order[idx]
        for num in candidates[i]:
            if num not in used:
                used.add(num)
                if dfs(idx + 1):
                    return True
                used.remove(num)
        return False
    
    return dfs(0)

def _compute_f(m, n=12, max_L=144):
    """Compute f(n, m): least positive L such that (m, m+L] contains distinct b_i with i|b_i."""
    for L in range(1, max_L + 1):
        if _has_matching(m, L, n):
            return L
    return max_L  # Should not happen per problem statement

def construct():
    """Find m in [0, 1000000] that maximizes f(12, m)."""
    deadline = time.time() + 85  # 85s deadline with 5s safety margin
    best_m = 12
    best_f = _compute_f(best_m)
    
    # Strategy: search for m values that maximize f(12, m)
    # f(12, m) is the least L such that (m, m+L] has a valid matching.
    # To maximize f, we want m such that for small L, there's no valid matching.
    
    # Key insight: for a given L, (m, m+L] has no valid matching if for some i,
    # there are too few multiples of i in (m, m+L], or if the Hall's condition fails.
    
    # Let's try a heuristic: look for m where the interval (m, m+1] through (m, m+L-1]
    # all fail to have a matching.
    
    # We'll sample m values and compute f(12, m) for them.
    
    # First, let's try some structured search.
    # The value f(12, m) depends on the distribution of multiples of 1..12 in (m, m+L].
    
    # Let's try to find m where for L=1, (m, m+1] = {m+1} must be divisible by all of 1..12?
    # No, we need distinct b_i, so b_i = m+1 for all i is not allowed.
    # For L=1, (m, m+1] = {m+1}, only one number, but we need 12 distinct numbers. So f >= 12 always?
    # Wait, we need 12 distinct numbers b_1,...,b_12 in (m, m+L]. So L must be at least 12.
    
    # So f(12, m) >= 12 always.
    
    # For L=12, (m, m+12] has 12 numbers. We need to assign them to indices 1..12 such that i|b_i.
    # This is possible only if the 12 numbers can be matched to 1..12 with divisibility.
    
    # Let's do a more systematic search.
    # We'll try m values and compute f(12, m).
    
    # To be efficient, let's first compute f for a set of candidate m values.
    # Candidate m values: those where the interval (m, m+12] is "hard" to match.
    
    # Let's try m = 0, 1, ..., some range, and also some structured values.
    
    # Actually, let's just do a random/structured search within the time budget.
    
    # Try m values in a structured way:
    # 1. Small m values
    # 2. m values near multiples of lcm(1..12) = 27720
    
    lcm_12 = 27720  # lcm(1,2,...,12)
    
    # Try m values
    candidates_m = set()
    
    # Small range
    for m in range(0, 200):
        candidates_m.add(m)
    
    # Near multiples of lcm
    for k in range(0, 40):
        base = k * lcm_12
        for delta in range(-50, 51):
            m = base + delta
            if 0 <= m <= 1000000:
                candidates_m.add(m)
    
    # Also try some random values
    import random
    random.seed(42)
    for _ in range(200):
        candidates_m.add(random.randint(0, 1000000))
    
    # Sort candidates and evaluate
    candidates_m = sorted(candidates_m)
    
    for m in candidates_m:
        if time.time() > deadline - 5:
            break
        f_val = _compute_f(m)
        if f_val > best_f:
            best_f = f_val
            best_m = m
            if best_f >= 144:
                break  # Maximum possible
    
    # If we have time, do a more thorough search around the best m
    if time.time() < deadline - 10:
        # Search in a window around best_m
        for m in range(max(0, best_m - 200), min(1000001, best_m + 200)):
            if time.time() > deadline - 5:
                break
            f_val = _compute_f(m)
            if f_val > best_f:
                best_f = f_val
                best_m = m
                if best_f >= 144:
                    break
    
    return best_m
# EVOLVE-BLOCK-END
