import time

def _can_match(m, L, n=12):
    """Check if (m, m+L] contains distinct b_1..b_n with i | b_i.
    Uses bitmask DP over the 12 left nodes.
    """
    # For each i in 1..n, the set of offsets k in [1,L] where (m+k) is divisible by i.
    # We represent candidate offsets per i as a bitset over k=1..L.
    # Two different i's can share an offset k only if we assign them to distinct k.
    # We do DP over i: dp[mask] = True means the first popcount(mask) indices
    # (in a fixed order) can be assigned to distinct offsets encoded by mask bits.
    # But offsets up to 144 -> bitset of 144 bits is fine as int.

    # For each i, precompute the set of valid offsets as a bitmask (bit k-1 for offset k).
    valid = [0] * n
    for i in range(1, n + 1):
        # offset k valid iff (m + k) % i == 0  =>  k % i == (-m) % i
        r = (-m) % i
        mask = 0
        # k ranges 1..L
        if r == 0:
            start = i
        else:
            start = r
        for k in range(start, L + 1, i):
            mask |= (1 << (k - 1))
        if mask == 0:
            return False
        valid[i - 1] = mask

    # DP: iterate indices in order of fewest candidates first to prune early.
    order = sorted(range(n), key=lambda i: bin(valid[i]).count('1'))

    # dp is a dict/set of reachable masks after processing some prefix.
    # Start with mask 0.
    reachable = {0}
    for idx in order:
        mask_i = valid[idx]
        new_reachable = set()
        for mask in reachable:
            # We can add any offset k in mask_i that is not already in mask.
            avail = mask_i & ~mask
            # For each set bit in avail, add mask | (1<<bit)
            a = avail
            while a:
                low = a & (-a)
                new_reachable.add(mask | low)
                a -= low
        reachable = new_reachable
        if not reachable:
            return False
    return True

def _f(m, n=12):
    """Compute f(n, m): least positive L such that (m, m+L] has distinct b_i with i|b_i."""
    lo, hi = 1, 144  # f is at most 144
    while lo < hi:
        mid = (lo + hi) // 2
        if _can_match(m, mid, n):
            hi = mid
        else:
            lo = mid + 1
    return lo

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before the 90s kill

    n = 12
    period = 27720  # lcm(1, 2, ..., 12)

    best_m = 0
    best_f = 0

    # f(n, m) is periodic in m with period lcm(1..n)=27720, so scanning [0, period)
    # covers the entire search space. Use the fast bitmask matcher to make this feasible.
    for m in range(period):
        if time.time() > deadline:
            break
        f_val = _f(m, n)
        if f_val > best_f:
            best_f = f_val
            best_m = m

    return int(best_m)

# EVOLVE-BLOCK-END
