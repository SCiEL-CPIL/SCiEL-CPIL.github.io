import time
from math import gcd
from functools import reduce

# EVOLVE-BLOCK-START
DIVISORS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
N = len(DIVISORS)
MAX_VAL = 1000000
PERIOD = reduce(lambda a, b: a * b // gcd(a, b), DIVISORS)  # 30030
START_TIME = time.time()
TIME_LIMIT = 85.0  # Safety margin for 90s limit

def construct():
    start = time.time()
    deadline = start + TIME_LIMIT

    n = N
    divisors = DIVISORS
    max_val = MAX_VAL
    period = PERIOD

    def fast_feasibility(L, W):
        """Fast feasibility check using backtracking with MRV and forward checking."""
        if W < n:
            return False
        R = L + W
        if R > max_val:
            return False

        # Build candidate lists for each divisor
        cands = []
        for i in range(n):
            d = divisors[i]
            rem = (-L) % d
            c = []
            j = rem
            if j >= W:
                return False
            while j < W:
                c.append(j)
                j += d
            cands.append(c)

        # Sort by number of candidates (MRV)
        order = sorted(range(n), key=lambda i: len(cands[i]))

        if len(cands[order[0]]) == 0:
            return False

        used = [False] * W

        def backtrack(idx):
            if idx == n:
                return True
            i = order[idx]
            for j in cands[i]:
                if not used[j]:
                    used[j] = True
                    if backtrack(idx + 1):
                        return True
                    used[j] = False
            return False

        return backtrack(0)

    def find_assignment(L, W):
        """Find actual assignment in window [L, L+W]."""
        if W < n:
            return None
        R = L + W
        if R > max_val:
            return None

        cands = []
        for i in range(n):
            d = divisors[i]
            rem = (-L) % d
            c = []
            j = rem
            if j >= W:
                return None
            while j < W:
                c.append(j)
                j += d
            cands.append(c)

        order = sorted(range(n), key=lambda i: len(cands[i]))
        if len(cands[order[0]]) == 0:
            return None

        used = [False] * W
        assignment = [0] * n

        def backtrack(idx):
            if idx == n:
                return True
            i = order[idx]
            for j in cands[i]:
                if not used[j]:
                    used[j] = True
                    assignment[i] = L + j
                    if backtrack(idx + 1):
                        return True
                    used[j] = False
            return False

        if backtrack(0):
            result = [assignment[i] for i in range(n)]
            # Validate
            if len(set(result)) == n:
                for i in range(n):
                    if not (isinstance(result[i], int) and result[i] % divisors[i] == 0 and 1 <= result[i] <= max_val):
                        return None
                return result
        return None

    # Fallback: the divisors themselves
    fallback = divisors[:]

    best_diff = float('inf')
    best_L = -1

    # Phase 1: Coarse search to get initial bound
    # Try various L values with small W
    step = max(1, period // 2000)
    for L in range(1, period + 1, step):
        if time.time() > deadline - 15:
            break
        for W in range(n - 1, min(n - 1 + 150, max_val - L)):
            if time.time() > deadline - 15:
                break
            if fast_feasibility(L, W):
                if W < best_diff:
                    best_diff = W
                    best_L = L
                break

    # Phase 2: Fine search with binary search on W for each L
    if best_diff == float('inf'):
        # If no feasible solution found in coarse search, try a broader search
        for L in range(1, min(period + 1, 1000)):
            if time.time() > deadline - 10:
                break
            for W in range(n - 1, min(n - 1 + 500, max_val - L)):
                if time.time() > deadline - 10:
                    break
                if fast_feasibility(L, W):
                    if W < best_diff:
                        best_diff = W
                        best_L = L
                    break

    if best_diff == float('inf'):
        # Last resort: try with larger W
        result = find_assignment(1, 1000)
        if result:
            return result
        return fallback

    # Now do a thorough search for better solutions
    for L in range(1, period + 1):
        if time.time() > deadline - 5:
            break

        # Upper bound on W
        hi = min(best_diff - 1, max_val - L)
        if hi < 0:
            continue
        lo = n - 1

        if lo > hi:
            continue

        found_width = None
        while lo <= hi:
            if time.time() > deadline - 5:
                break
            mid = (lo + hi) // 2
            if fast_feasibility(L, mid):
                found_width = mid
                hi = mid - 1
            else:
                lo = mid + 1

        if found_width is not None:
            if found_width < best_diff:
                best_diff = found_width
                best_L = L

    # Find the actual assignment
    if best_L != -1:
        result = find_assignment(best_L, best_diff)
        if result:
            return result

    return fallback
# EVOLVE-BLOCK-END
