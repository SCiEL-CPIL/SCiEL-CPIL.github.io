import time
from math import gcd
from functools import reduce
from collections import deque

DIVISORS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
N = len(DIVISORS)
MAX_VAL = 1000000
PERIOD = reduce(lambda a, b: a * b // gcd(a, b), DIVISORS)  # 60060

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    n = N
    divisors = DIVISORS
    max_val = MAX_VAL
    period = PERIOD

    # Precompute for each divisor d, the residues that matter
    # For a window starting at L with width W, divisor d has candidates at positions
    # j where (L + j) % d == 0, i.e., j ≡ (-L) % d (mod d)
    # Number of candidates in [0, W-1] is either W//d or W//d + 1

    def fast_feasibility(L, W):
        """Fast feasibility check using backtracking with MRV."""
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

        # If any divisor has 0 candidates, infeasible
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

    # Strategy: First, do a coarse search to find a good upper bound on W.
    # Then, search all L positions with binary search on W using fast feasibility.

    best_diff = float('inf')
    best_L = -1

    # Phase 1: Coarse search - try various L values with small W to get initial bound
    # Try L values that are multiples of small numbers to find tight windows
    coarse_Ls = list(range(1, period + 1, max(1, period // 2000)))
    # Also add some strategic L values
    for L in coarse_Ls:
        if time.time() > deadline - 10:
            break
        # Try W from n-1 upward, but cap at a reasonable value
        for W in range(n - 1, min(n - 1 + 200, max_val - L)):
            if time.time() > deadline - 10:
                break
            if fast_feasibility(L, W):
                if W < best_diff:
                    best_diff = W
                    best_L = L
                break  # Found feasible W for this L, move to next L

    # Phase 2: Fine search - for each L, binary search on W
    # Use the current best_diff as upper bound
    if best_diff == float('inf'):
        # Fallback: use a simple heuristic
        result = find_assignment(1, 1000)
        if result:
            return result
        return [divisors[i] for i in range(n)]

    # Now do a thorough search
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

    # Fallback
    return [divisors[i] for i in range(n)]
