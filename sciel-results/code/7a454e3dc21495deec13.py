"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

N = 12
MAX_M = 1000000
MAX_L = 144  # proven upper bound: any 144 consecutive ints contain a full matching
LCM_12 = 27720  # lcm(1,2,...,12)


def _has_matching(m, L):
    """Return True iff (m, m+L] contains distinct b_i with i | b_i for i=1..N.
    Uses bipartite matching (Hopcroft-Karp style DFS).
    """
    # For each i in 1..N, find all positions j in 0..L-1 where (m+1+j) is divisible by i
    # Build adjacency: adj[i] = list of j values where i divides (m+1+j)
    adj = [[] for _ in range(N)]
    for i in range(1, N + 1):
        # first multiple of i in (m, m+L]
        start = m + 1
        q = (start + i - 1) // i
        v = q * i
        while v <= m + L:
            j = v - m - 1
            adj[i - 1].append(j)
            v += i

    # DFS-based bipartite matching
    # match[j] = i that is matched to position j, or -1
    match = [-1] * L

    def dfs(i, visited):
        for j in adj[i]:
            if visited[j]:
                continue
            visited[j] = True
            if match[j] == -1 or dfs(match[j], visited):
                match[j] = i
                return True
        return False

    # Try to find matching for each left node
    count = 0
    for i in range(N):
        visited = [False] * L
        if dfs(i, visited):
            count += 1
        else:
            return False
    return True


def _compute_f(m):
    """Compute f(N, m): least positive L with a matching in (m, m+L]."""
    lo, hi = 1, MAX_L
    while lo < hi:
        mid = (lo + hi) // 2
        if _has_matching(m, mid):
            hi = mid
        else:
            lo = mid + 1
    return lo


def _find_best(deadline):
    """Search for m maximizing f(N,m). Returns (best_m, best_f)."""
    t0 = time.time()

    best_m = 0
    best_f = 0

    # Exploit periodicity: f(12, m) depends only on m mod 27720
    # because the divisibility pattern repeats every LCM(1..12) = 27720.
    # So we only need to check m in [0, 27719] and then map back.

    # Phase 1: Coarse scan with step 10 to find promising regions
    step = 10
    for m in range(0, LCM_12, step):
        if (time.time() - t0) > deadline - 10:
            break
        f = _compute_f(m)
        if f > best_f:
            best_f = f
            best_m = m

    # Phase 2: Fine scan around best region
    if (time.time() - t0) < deadline - 5:
        # Scan neighbors of best_m with step 1
        start = max(0, best_m - step)
        end = min(LCM_12, best_m + step + 1)
        for m in range(start, end):
            if (time.time() - t0) > deadline - 5:
                break
            f = _compute_f(m)
            if f > best_f:
                best_f = f
                best_m = m

    # Phase 3: If time remains, do full scan
    if (time.time() - t0) < deadline - 5:
        # Check if we can afford full scan
        elapsed = time.time() - t0
        remaining = deadline - 5 - elapsed
        # Estimate: we've done ~2772 coarse + ~20 fine = ~2792 checks so far
        # Full scan needs 27720 checks
        checks_done = LCM_12 // step + 2 * step
        checks_needed = LCM_12
        if remaining > 0 and (checks_needed - checks_done) * (elapsed / max(1, checks_done)) < remaining:
            for m in range(0, LCM_12):
                if (time.time() - t0) > deadline - 3:
                    break
                f = _compute_f(m)
                if f > best_f:
                    best_f = f
                    best_m = m

    # best_m is in [0, 27719], which is valid since MAX_M = 1000000
    return best_m, best_f


def construct():
    deadline = 85.0  # safety margin before 90s kill
    t0 = time.time()
    best_m, best_f = _find_best(deadline)
    # Ensure type is Python int
    return int(best_m)
# EVOLVE-BLOCK-END
