import time
import sys

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _lcm(a, b):
    return a * b // _gcd(a, b)

PERIOD = 27720
N = 12
MAX_L = 144

# Precompute change lists for each residue m in 0..PERIOD-1
# changes[m][i] = list of L values in [1, MAX_L] where a new multiple of i enters (m, m+L]
# Since i divides PERIOD, changes depend only on m mod i, determined by m mod PERIOD.

def _compute_changes(m):
    """For a given m, compute list of change points for each i in 1..12."""
    result = []
    for i in range(1, N + 1):
        pts = []
        k = (m // i) + 1
        val = k * i
        while val - m <= MAX_L:
            pts.append(val - m)
            val += i
        result.append(pts)
    return result

# Precompute for all residues
_all_changes = [None] * PERIOD
for _m in range(PERIOD):
    _all_changes[_m] = _compute_changes(_m)

def _has_perfect_matching_bitmask(adj_bitmasks, left_size, right_size):
    """Kuhn's algorithm using bitmask adjacency. adj_bitmasks[u] is a bitmask of right nodes."""
    match_to_right = [-1] * right_size

    def dfs(u, seen_mask):
        bm = adj_bitmasks[u]
        # Iterate over set bits
        while bm:
            v = (bm & (-bm)).bit_length() - 1
            bm &= bm - 1
            if (seen_mask >> v) & 1:
                continue
            seen_mask |= (1 << v)
            if match_to_right[v] == -1 or dfs(match_to_right[v], seen_mask):
                match_to_right[v] = u
                return True
        return False

    for u in range(left_size):
        seen_mask = 0
        if not dfs(u, seen_mask):
            return False
    return True

def _can_match(m, L):
    """Return True if (m, m+L] contains 12 distinct b_i with i | b_i."""
    adj_bitmasks = [0] * N
    for i in range(1, N + 1):
        bm = 0
        for Lc in _all_changes[m][i - 1]:
            if Lc <= L:
                bm |= (1 << (Lc - 1))
            else:
                break
        adj_bitmasks[i - 1] = bm

    # Quick Hall's check: for each left node, must have at least 1 neighbor
    for i in range(N):
        if adj_bitmasks[i] == 0:
            return False

    # Check union of all neighbors has enough capacity
    all_union = 0
    for i in range(N):
        all_union |= adj_bitmasks[i]
    if all_union.bit_count() < N:
        return False

    return _has_perfect_matching_bitmask(adj_bitmasks, N, L)

def _f_of_m(m):
    """Least positive L such that can_match(m, L)."""
    # Lower bound: max over i of distance to next multiple of i after m
    lo = 1
    for i in range(1, N + 1):
        d = ((m // i) + 1) * i - m
        if d > lo:
            lo = d
    hi = MAX_L + 1
    while lo < hi:
        mid = (lo + hi) // 2
        if _can_match(m, mid):
            hi = mid
        else:
            lo = mid + 1
    return lo

def construct():
    start = time.time()
    deadline = 85.0  # safety margin before 90s kill

    best_m = 0
    best_L = -1

    for m in range(PERIOD):
        if time.time() - start > deadline:
            break
        L = _f_of_m(m)
        if L > best_L:
            best_L = L
            best_m = m

    return int(best_m)

# EVOLVE-BLOCK-START
# Entry point: construct() -> int in [0, 1000000]
# EVOLVE-BLOCK-END
