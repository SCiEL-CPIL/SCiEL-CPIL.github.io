"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys
sys.setrecursionlimit(10000)

N = 12
LCM_12 = 27720  # LCM of 1..12

def can_assign(m, L):
    """Check if we can assign distinct b_i in (m, m+L] with i | b_i for i=1..12."""
    candidates = {}
    for i in range(1, N + 1):
        first = ((m // i) + 1) * i
        cands = []
        val = first
        while val <= m + L:
            cands.append(val)
            val += i
        if not cands:
            return False
        candidates[i] = cands
    
    # Order by most constrained (fewest candidates) first
    order = sorted(range(1, N + 1), key=lambda i: len(candidates[i]))
    used = set()
    
    def backtrack(pos):
        if pos == N:
            return True
        i = order[pos]
        for v in candidates[i]:
            if v not in used:
                used.add(v)
                if backtrack(pos + 1):
                    return True
                used.remove(v)
        return False
    
    return backtrack(0)

def compute_f(m):
    """Compute f(12, m): least positive L such that (m, m+L] has distinct valid b_i."""
    lo, hi = 1, 144
    while lo < hi:
        mid = (lo + hi) // 2
        if can_assign(m, mid):
            hi = mid
        else:
            lo = mid + 1
    return lo

def construct():
    best_m = 0
    best_L = -1
    for m in range(LCM_12):
        L = compute_f(m)
        if L > best_L:
            best_L = L
            best_m = m
    return best_m
# EVOLVE-BLOCK-END
