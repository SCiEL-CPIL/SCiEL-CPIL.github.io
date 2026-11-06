"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill

    target = Fraction(5, 31)
    MAX_D_GLOBAL = 10000000
    MAX_TERMS = 256

    # ---------- Baseline via greedy from various starts ----------
    def greedy_complete(remaining, max_d=MAX_D_GLOBAL):
        vals = []
        r = remaining
        while r > 0:
            if r.numerator == 0:
                break
            d = (r.denominator + r.numerator - 1) // r.numerator
            if d > max_d:
                return None
            vals.append(d)
            r -= Fraction(1, d)
        return vals

    baseline = None
    best_max = float('inf')
    for d0 in range(2, 20):
        if time.time() > deadline:
            break
        r = target - Fraction(1, d0)
        if r <= 0:
            continue
        vals = greedy_complete(r)
        if vals is not None:
            full = [d0] + vals
            if len(full) != len(set(full)):
                continue
            if max(full) <= MAX_D_GLOBAL and len(full) <= MAX_TERMS:
                mx = max(full)
                if mx < best_max:
                    best_max = mx
                    baseline = full

    if baseline is None:
        # pure greedy
        vals = greedy_complete(target)
        if vals:
            baseline = vals
            best_max = max(vals)
        else:
            return [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    best_solution = list(baseline)
    best_max_den = best_max

    # ---------- Branch and bound DFS ----------
    # We search for a set of distinct denominators (sorted ascending) summing to target,
    # minimizing the maximum denominator.
    #
    # State: (remaining_value, min_d, depth, current_list, current_max_allowed)
    # Pruning:
    #   1. If remaining_value == 0: success if current_list non-empty
    #   2. If depth >= MAX_TERMS: fail
    #   3. If remaining_value <= 0: fail
    #   4. Upper bound on sum: sum of 1/d for d in [min_d, min_d+k-1] (k = remaining terms budget)
    #      If this upper bound < remaining_value, prune.
    #   5. Lower bound: 1/min_d (the largest single reciprocal we can add). If 1/min_d > remaining_value,
    #      we must skip min_d and try larger.
    #   6. If we're trying to beat best_max_den, cap d at best_max_den - 1.

    # Precompute: for a given min_d and number of terms k, the maximum possible sum
    # is sum_{i=0}^{k-1} 1/(min_d + i). We can compute this on the fly.

    def max_sum_from(min_d, k):
        """Upper bound: sum of k largest reciprocals starting from min_d."""
        if k <= 0:
            return Fraction(0, 1)
        # For small k, compute directly; for large k, approximate
        if k <= 50:
            s = Fraction(0, 1)
            for i in range(k):
                s += Fraction(1, min_d + i)
            return s
        else:
            # Approximate with integral: sum_{i=0}^{k-1} 1/(min_d+i) < 1/min_d + ln((min_d+k-1)/min_d)
            # But for pruning we need a valid upper bound, so use a looser one
            # Actually, the sum is < 1/min_d * k is too loose. Let's just cap at a reasonable value.
            # For large k, the sum is bounded by 1/min_d + 1/(min_d+1) + ... which is < 1/min_d + ln(1 + (k-1)/min_d)
            import math
            return Fraction(1, min_d) + Fraction(math.log1p((k - 1) / min_d), 1)  # approximate upper bound

    def min_sum_from(min_d, k):
        """Lower bound: the smallest sum of k distinct denominators >= min_d.
        This is achieved by taking the k largest possible denominators, but since we have no upper cap
        in the search (other than best_max_den), the minimum sum approaches 0.
        So the useful lower bound is just 1/(best_max_den) * k if we cap, but that's not tight.
        Actually, for feasibility we mainly need the upper bound check."""
        return Fraction(0, 1)

    # DFS with iterative deepening on max denominator
    # We'll try to find a solution with max_den < best_max_den
    # by searching with d_cap = best_max_den - 1

    def dfs(remaining, min_d, depth, current, d_cap):
        """
        Try to find distinct denominators >= min_d, <= d_cap, summing to remaining,
        with at most (MAX_TERMS - depth) more terms.
        Returns list of additional denominators or None.
        """
        if time.time() > deadline:
            return None

        if remaining == 0:
            return []

        if depth >= MAX_TERMS:
            return None

        if remaining <= 0:
            return None

        # Upper bound check: can we still reach 'remaining' with the available terms?
        remaining_terms = MAX_TERMS - depth
        # The maximum sum we can achieve with 'remaining_terms' denominators starting from min_d
        # is sum_{i=0}^{remaining_terms-1} 1/(min_d + i), but capped by d_cap
        # Actually, the denominators are in [min_d, d_cap], distinct.
        # The max sum is achieved by taking the smallest denominators: min_d, min_d+1, ...
        # But we can only go up to d_cap.
        
        # Compute upper bound more carefully
        # We want the sum of 1/d for d in [min_d, d_cap], taking the largest 'remaining_terms' values of 1/d
        # i.e., the smallest denominators.
        # The smallest denominators are min_d, min_d+1, ..., min_d+remaining_terms-1
        # But we're capped at d_cap.
        
        ub = Fraction(0, 1)
        for i in range(remaining_terms):
            d = min_d + i
            if d > d_cap:
                break
            ub += Fraction(1, d)
        
        if ub < remaining:
            return None

        # Also: the minimum denominator we can use is min_d.
        # If 1/min_d > remaining, we can't use min_d, so we skip to min_d+1.
        # But we handle that in the loop below.

        # Determine the range of d to try
        # d must satisfy: 1/d <= remaining (otherwise 1/d > remaining, can't use)
        # So d >= ceil(1/remaining)
        d_min = max(min_d, (remaining.denominator + remaining.numerator - 1) // remaining.numerator)
        
        # d must be <= d_cap
        if d_min > d_cap:
            return None

        # Also, we need enough "room": the maximum sum with remaining_terms denominators
        # starting from d_min must be >= remaining. (Already checked above with min_d,
        # but d_min might be larger than min_d, so re-check)
        ub2 = Fraction(0, 1)
        for i in range(remaining_terms):
            d = d_min + i
            if d > d_cap:
                break
            ub2 += Fraction(1, d)
        if ub2 < remaining:
            return None

        # Try denominators from d_min upward
        # To minimize the maximum, we should try smaller d first? No, we want to find ANY solution
        # with max < best_max_den. The max of the solution is the largest d we pick.
        # To minimize the overall max, we should try to keep all d small.
        # So try d from d_min upward, and the first solution found will have a certain max.
        # But we want the MINIMUM possible max. So we should do iterative deepening on d_cap.
        
        # For now, just find any solution with d <= d_cap.
        # We'll call this with d_cap = best_max_den - 1, and if found, update best.
        
        for d in range(d_min, d_cap + 1):
            if time.time() > deadline:
                return None
            
            rem2 = remaining - Fraction(1, d)
            if rem2 < 0:
                # d too small, 1/d > remaining, skip to larger d
                continue
            if rem2 == 0:
                return [d]
            
            result = dfs(rem2, d + 1, depth + 1, current + [d], d_cap)
            if result is not None:
                return [d] + result
        
        return None

    # Iterative improvement: try to reduce best_max_den
    # Start with current best, try to find solution with max < best_max_den
    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        d_cap = best_max_den - 1
        if d_cap < 2:
            break
        
        # Try DFS with this cap
        result = dfs(target, 2, 0, [], d_cap)
        if result is not None:
            # Verify
            s = sum(Fraction(1, d) for d in result)
            if s == target and len(result) <= MAX_TERMS and all(1 <= d <= MAX_D_GLOBAL for d in result) and len(set(result)) == len(result):
                new_max = max(result)
                if new_max < best_max_den:
                    best_solution = result
                    best_max_den = new_max
                    improved = True
    
    # Final verification
    final = list(best_solution)
    s = sum(Fraction(1, d) for d in final)
    if s != target or len(final) > MAX_TERMS or any(d < 1 or d > MAX_D_GLOBAL for d in final) or len(set(final)) != len(final):
        final = list(baseline)

    return final

# EVOLVE-BLOCK-END
