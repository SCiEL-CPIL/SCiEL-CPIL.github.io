import time

# LCM(1..12)
def _lcm(a, b):
    return a * b // _gcd(a, b)

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

PERIOD = 27720
N = 12
MAX_L = 144

# Precompute, for each i in 1..12, the sorted list of L values (1..MAX_L)
# at which the count of multiples of i in (m, m+L] increases.
# count_i(L) = floor((m+L)/i) - floor(m/i). This increases exactly when (m+L) % i == 0.
# For a fixed residue m, the change points are deterministic, so precompute per residue.
# Instead we compute change points on the fly cheaply (there are <= L/i of them).

def _changes_for_i(m, i):
    """Yield L in [1, MAX_L] where a new multiple of i enters (m, m+L]."""
    # First multiple of i greater than m:
    k = (m // i) + 1
    val = k * i
    # subsequent multiples
    while val - m <= MAX_L:
        yield val - m
        val += i

# Build the change lists once for all residues (they depend on m mod i only,
# and i divides PERIOD, so change points depend only on m mod i, which is
# determined by m mod PERIOD). We memoize by (m % i).
_change_cache = {}
def get_changes(m, i):
    key = (m % i, i)
    if key not in _change_cache:
        _change_cache[key] = list(_changes_for_i(m % i, i))
    return _change_cache[key]

def has_perfect_matching(adj, left_size, right_size):
    """Kuhn's algorithm with bitmask adjacency. adj[u] = bitmask of right nodes. Returns bool."""
    match_to_right = [-1] * right_size

    def dfs(u, seen):
        # Iterate set bits in adj[u]
        mask = adj[u]
        while mask:
            # Get lowest set bit position
            lsb = mask & (-mask)
            v = lsb.bit_length() - 1
            mask ^= lsb
            if seen & lsb:
                continue
            seen |= lsb
            if match_to_right[v] == -1 or dfs(match_to_right[v], seen):
                match_to_right[v] = u
                return True
        return False

    for u in range(left_size):
        if not dfs(u, 0):
            return False
    return True

def can_match(m, L):
    """Return True if (m, m+L] contains 12 distinct b_i with i | b_i."""
    right_size = L
    adj = [0] * N
    for i in range(1, N + 1):
        mask = 0
        for Lc in get_changes(m, i):
            if Lc <= L:
                # The multiple that entered at Lc is the value m + Lc.
                # Its 0-indexed slot in (m, m+L] is Lc - 1.
                mask |= (1 << (Lc - 1))
            else:
                break
        adj[i - 1] = mask
    return has_perfect_matching(adj, N, right_size)

def f_of_m(m):
    """Least positive L such that can_match(m, L)."""
    # lower bound: max over i of distance to next multiple of i after m
    lo = 0
    for i in range(1, N + 1):
        d = ((m // i) + 1) * i - m
        if d > lo:
            lo = d
    lo = max(lo, 1)
    hi = MAX_L + 1  # we know answer <= 144
    # binary search on L
    while lo < hi:
        mid = (lo + hi) // 2
        if can_match(m, mid):
            hi = mid
        else:
            lo = mid + 1
    return lo

def construct():
    start = time.time()
    deadline = 55.0  # safety margin before 60s kill

    best_m = 0
    best_L = -1

    # Precompute change cache for all residues is unnecessary; on-the-fly memo suffices.
    # Iterate residues 0..PERIOD-1; stop early on deadline, keep best so far.
    for m in range(PERIOD):
        if time.time() - start > deadline:
            break
        L = f_of_m(m)
        if L > best_L:
            best_L = L
            best_m = m

    # best_m is in [0, PERIOD-1] which is within [0, 1000000].
    return int(best_m)

# EVOLVE-BLOCK-START
# Entry point: construct() -> int in [0, 1000000]
# EVOLVE-BLOCK-END
