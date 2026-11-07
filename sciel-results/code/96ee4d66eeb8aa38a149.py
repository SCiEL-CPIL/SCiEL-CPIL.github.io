import time
from math import gcd
from functools import reduce

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

    def find_assignment_fast(L, W):
        """Find actual assignment in window [L, L+W) using optimized backtracking."""
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

        used = [0] * W  # Use int array for speed
        assignment = [0] * n

        def backtrack(idx):
            if idx == n:
                return True
            i = order[idx]
            for j in cands[i]:
                if used[j] == 0:
                    used[j] = 1
                    assignment[i] = L + j
                    if backtrack(idx + 1):
                        return True
                    used[j] = 0
            return False

        if backtrack(0):
            result = [assignment[i] for i in range(n)]
            if len(set(result)) == n:
                for i in range(n):
                    if not (isinstance(result[i], int) and result[i] % divisors[i] == 0 and 1 <= result[i] <= max_val):
                        return None
                return result
        return None

    def feasible(L, W):
        """Quick feasibility check - just verify each divisor has at least one candidate."""
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
    best_L = -1
    best_result = None

    # Phase 1: Quick scan to find any feasible solution
    # Try small W values first with strategic L positions
    for W in range(n - 1, n - 1 + 50):
        if time.time() > deadline - 20:
            break
        for L in range(1, period + 1, max(1, period // 500)):
            if time.time() > deadline - 20:
                break
            if feasible(L, W):
                result = find_assignment_fast(L, W)
                if result:
                    if W < best_diff:
                        best_diff = W
                        best_L = L
                        best_result = result
                    break  # Found for this W, try smaller W
    
    # Phase 2: If we found something, try to improve
    if best_diff < float('inf'):
        # Try all L positions with the current best W
        for L in range(1, period + 1):
            if time.time() > deadline - 10:
                break
            # Try to find a solution with width < best_diff
            for W in range(n - 1, best_diff):
                if feasible(L, W):
                    result = find_assignment_fast(L, W)
                    if result:
                        if W < best_diff:
                            best_diff = W
                            best_L = L
                            best_result = result
                        break
    
    # Phase 3: Thorough search if time remains
    if time.time() < deadline - 5:
        # Binary search on W for each L
        for L in range(1, period + 1):
            if time.time() > deadline - 3:
                break
            
            hi = min(best_diff - 1, max_val - L) if best_diff < float('inf') else min(100, max_val - L)
            lo = n - 1
            
            if lo > hi:
                continue
            
            found_width = None
            while lo <= hi:
                if time.time() > deadline - 3:
                    break
                mid = (lo + hi) // 2
                if feasible(L, mid):
                    result = find_assignment_fast(L, mid)
                    if result:
                        found_width = mid
                        hi = mid - 1
                    else:
                        lo = mid + 1
                else:
                    lo = mid + 1
            
            if found_width is not None and found_width < best_diff:
                best_diff = found_width
                best_L = L
                best_result = find_assignment_fast(L, found_width)

    # Return best result
    if best_result:
        return best_result
    
    # Fallback
    return [divisors[i] for i in range(n)]
