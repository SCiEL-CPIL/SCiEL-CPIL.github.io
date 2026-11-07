import time
from math import gcd
from functools import reduce

DIVISORS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
N = len(DIVISORS)
MAX_VAL = 1000000
PERIOD = reduce(lambda a, b: a * b // gcd(a, b), DIVISORS)  # 60060

def construct():
    start_time = time.time()
    deadline = start_time + 80.0

    n = N
    divisors = DIVISORS
    max_val = MAX_VAL
    period = PERIOD

    def find_assignment_fc(L, W):
        """Find assignment in window [L, L+W) using backtracking with forward checking."""
        if W < n:
            return None
        if L + W > max_val:
            return None

        # Build candidate lists (offsets within window)
        cands = []
        for i in range(n):
            d = divisors[i]
            rem = (-L) % d
            if rem >= W:
                return None
            c = list(range(rem, W, d))
            cands.append(c)

        # MRV ordering: sort by number of candidates
        order = sorted(range(n), key=lambda i: len(cands[i]))
        if len(cands[order[0]]) == 0:
            return None

        assignment = [None] * n
        used = [False] * W

        def backtrack(idx):
            if idx == n:
                return True
            
            # Dynamic MRV: pick unassigned variable with fewest available candidates
            best_var = -1
            best_count = W + 1
            for vi in range(idx, n):
                var = order[vi]
                if assignment[var] is not None:
                    continue
                avail = sum(1 for j in cands[var] if not used[j])
                if avail == 0:
                    return False
                if avail < best_count:
                    best_count = avail
                    best_var = vi
            
            if best_var == -1:
                return True
            
            var = order[best_var]
            for j in cands[var]:
                if not used[j]:
                    used[j] = True
                    assignment[var] = L + j
                    if backtrack(idx + 1):
                        return True
                    used[j] = False
                    assignment[var] = None
            return False

        if backtrack(0):
            result = [assignment[i] for i in range(n)]
            for i in range(n):
                if not (isinstance(result[i], int) and result[i] % divisors[i] == 0 and 1 <= result[i] <= max_val):
                    return None
            return result
        return None

    def feasible(L, W):
        """Quick feasibility check."""
        if W < n:
            return False
        if L + W > max_val:
            return False
        for i in range(n):
            d = divisors[i]
            rem = (-L) % d
            if rem >= W:
                return False
        return True

    best_diff = float('inf')
    best_result = None

    # Phase 1: Try small W values, scanning ALL L across one period
    # For small W, the feasible L positions are sparse but we must check all
    for W in range(n, n + 20):
        if time.time() > deadline - 30:
            break
        # Scan all L positions in one period
        for L in range(1, period + 1):
            if time.time() > deadline - 30:
                break
            if feasible(L, W):
                result = find_assignment_fc(L, W)
                if result:
                    if W < best_diff:
                        best_diff = W
                        best_result = result
                    break  # Found for this W, try next W

    # Phase 2: If found, try to improve by scanning all L for smaller W
    if best_diff < float('inf'):
        for W in range(n, best_diff):
            if time.time() > deadline - 15:
                break
            for L in range(1, period + 1):
                if time.time() > deadline - 15:
                    break
                if feasible(L, W):
                    result = find_assignment_fc(L, W)
                    if result:
                        if W < best_diff:
                            best_diff = W
                            best_result = result
                        break  # Found for this W, try smaller W

    # Phase 3: Thorough search with binary search on W for each L
    if time.time() < deadline - 5:
        for L in range(1, period + 1):
            if time.time() > deadline - 3:
                break
            
            hi = min(best_diff - 1, max_val - L) if best_diff < float('inf') else min(50, max_val - L)
            lo = n - 1
            
            if lo > hi:
                continue
            
            found_width = None
            while lo <= hi:
                if time.time() > deadline - 3:
                    break
                mid = (lo + hi) // 2
                if feasible(L, mid):
                    result = find_assignment_fc(L, mid)
                    if result:
                        found_width = mid
                        hi = mid - 1
                    else:
                        lo = mid + 1
                else:
                    lo = mid + 1
            
            if found_width is not None and found_width < best_diff:
                best_diff = found_width
                best_result = find_assignment_fc(L, found_width)

    if best_result:
        return best_result
    
    return [divisors[i] for i in range(n)]
