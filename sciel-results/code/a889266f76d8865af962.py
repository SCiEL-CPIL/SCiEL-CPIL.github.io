"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

N = 12
MAX_M = 1000000

# LCM of 1..12 = 27720
_LCM = 27720


def _lcm(a, b):
    g = _gcd(a, b)
    return a // g * b


def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a


# Precompute LCM of 1..12
_LCM = 1
for _i in range(2, 13):
    _LCM = _lcm(_LCM, _i)
assert _LCM == 27720


def _has_perfect_matching(m, L):
    """Check if (m, m+L] contains distinct b_1..b_12 with i | b_i.
    
    Uses bitmask-based bipartite matching (Hopcroft-Karp style or DFS augmenting paths).
    Left side: i in 1..12. Right side: numbers in (m, m+L].
    Edge from i to number x if i divides x.
    """
    # Build adjacency: for each i, which numbers in (m, m+L] are divisible by i
    # Numbers in (m, m+L] are m+1, m+2, ..., m+L
    
    # For efficiency, we only need to consider numbers that are divisible by at least one i
    # Since i ranges 1..12, every number is divisible by 1, so all L numbers are candidates.
    # But we can prune: for each i, the numbers divisible by i in (m, m+L] form an arithmetic progression.
    
    # Use DFS-based bipartite matching
    # right_nodes: 0..L-1 representing m+1, m+2, ..., m+L
    
    # adj[i] = list of right node indices (0-indexed) that i can match to
    adj = [[] for _ in range(N)]
    for i in range(1, N + 1):
        # Find all numbers in (m, m+L] divisible by i
        # First number >= m+1 divisible by i:
        first = ((m + 1 + i - 1) // i) * i
        if first > m + L:
            continue
        x = first
        while x <= m + L:
            adj[i - 1].append(x - m - 1)  # 0-indexed right node
            x += i
    
    # If any i has no candidates, no matching
    for i in range(N):
        if not adj[i]:
            return False
    
    # Bipartite matching: left = 0..11 (representing i=1..12), right = 0..L-1
    match_r = [-1] * L  # match_r[j] = left node matched to right node j, or -1
    
    def dfs(u, visited):
        for v in adj[u]:
            if visited[v]:
                continue
            visited[v] = True
            if match_r[v] == -1 or dfs(match_r[v], visited):
                match_r[v] = u
                return True
        return False
    
    result = 0
    for u in range(N):
        visited = [False] * L
        if dfs(u, visited):
            result += 1
        else:
            return False
    
    return result == N


def _compute_f(m):
    """Compute f(12, m): least positive L such that (m, m+L] has distinct b_i with i|b_i."""
    # Binary search on L. L is at most 144 per task description.
    # Actually, we can just linearly scan from L=1 upward, but binary search is faster.
    # Lower bound: need at least 12 numbers, so L >= 12.
    # Upper bound: 144 per task.
    
    lo, hi = 1, 144
    # First check if hi works
    if not _has_perfect_matching(m, hi):
        # This shouldn't happen per task description, but handle gracefully
        # Try larger
        while hi <= 1000 and not _has_perfect_matching(m, hi):
            hi *= 2
        if hi > 1000:
            return hi  # fallback
    
    while lo < hi:
        mid = (lo + hi) // 2
        if _has_perfect_matching(m, mid):
            hi = mid
        else:
            lo = mid + 1
    return lo


def construct():
    deadline = time.time() + 85  # 85s deadline, 5s safety margin
    
    best_m = 12
    best_L = _compute_f(best_m)
    
    # The problem has periodicity with period LCM(1..12) = 27720.
    # f(12, m) depends only on m mod 27720.
    # So we only need to search m in [0, 27719] (or any complete period).
    # But we also need m <= 1000000, which is fine.
    
    # Strategy: search all residues in one period, find the one that maximizes f(12, m).
    # Then return the smallest m in [0, 1000000] with that residue.
    
    # Precompute f for all residues 0..27719
    # This is 27720 calls to _compute_f, each doing binary search with up to ~8 matching checks.
    # Each matching check is O(N * L) in the worst case for DFS, with L up to 144.
    # So roughly 27720 * 8 * 12 * 144 ≈ 38 million operations. Should be fast enough.
    
    # Let's try a smarter approach: for each residue r in 0..27719, compute f(12, r).
    # But 27720 * (binary search ~8 steps) * (matching ~12*144) might be tight.
    # Let's optimize: many residues will have the same f value.
    
    # Actually, let's just do it directly with a time check.
    
    residues_to_check = list(range(_LCM))
    
    # Sort by a heuristic: residues where the "bottleneck" is likely highest.
    # A residue r is a bottleneck if there exists a subset S of {1..12} such that
    # the numbers in (r, r+L] divisible by elements of S are fewer than |S| for small L.
    # This is related to Hall's condition.
    
    # For now, just iterate through all residues.
    best_residue = 0
    best_L_res = 0
    
    for r in residues_to_check:
        if time.time() > deadline - 10:
            break
        
        L = _compute_f(r)
        if L > best_L_res:
            best_L_res = L
            best_residue = r
    
    # Find the smallest m in [0, 1000000] with m % 27720 == best_residue
    if best_residue <= MAX_M:
        m = best_residue
    else:
        # best_residue < 27720 <= 1000000, so this always works
        m = best_residue
    
    # Verify
    if time.time() < deadline - 5:
        L_final = _compute_f(m)
        # If our precomputation was correct, L_final should equal best_L_res
        # But double-check
        if L_final > best_L:
            best_m = m
            best_L = L_final
    
    return int(best_m)

# EVOLVE-BLOCK-END
