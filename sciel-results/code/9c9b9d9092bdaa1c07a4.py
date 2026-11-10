"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
from functools import lru_cache

P = {'divisors': [6, 10, 14, 15, 21, 22, 26, 33, 35, 39], 'maximum': 1000000}

def _is_valid_answer(ans, divisors):
    """Check if answer is valid: 10 distinct positive ints <= 1e6, each divisible by corresponding divisor."""
    if not isinstance(ans, (list, tuple)):
        return False
    if len(ans) != len(divisors):
        return False
    seen = set()
    for i, (val, div) in enumerate(zip(ans, divisors)):
        if not isinstance(val, int) or isinstance(val, bool):
            return False
        if val < 1 or val > P['maximum']:
            return False
        if val % div != 0:
            return False
        if val in seen:
            return False
        seen.add(val)
    return True

def _span(ans):
    return max(ans) - min(ans)

def construct():
    """
    Find 10 distinct positive integers <= 1000000, aligned with divisors [6,10,14,15,21,22,26,33,35,39].
    Each representative must be divisible by its corresponding input.
    Minimize max(representatives) - min(representatives).
    """
    divisors = P['divisors']
    n = len(divisors)
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    # The divisors are: [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    # Let's compute LCM to understand the periodicity
    from math import gcd
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    l = 1
    for d in divisors:
        l = lcm(l, d)
    # LCM is 6*10*14*15*21*22*26*33*35*39 with common factors removed
    # Let's just compute it
    l = divisors[0]
    for d in divisors[1:]:
        l = lcm(l, d)
    
    # The key insight: we need to find 10 distinct multiples, one for each divisor,
    # that are all within a small interval [L, L+S].
    
    # For a given interval [L, L+S], for each divisor d, the valid multiples are
    # k*d where ceil(L/d) <= k <= floor((L+S)/d).
    
    # We can use a backtracking search with constraint propagation (like a CSP).
    # But first, let's try a more direct approach.
    
    # Since the divisors are small (max 39), the multiples are dense.
    # In any interval of length 39, there's at least one multiple of 39.
    # In any interval of length 38, there might not be.
    # So the span S must be at least... well, it depends on the alignment.
    
    # Actually, the minimum possible span is related to the distinctness constraint.
    # We need 10 distinct values, so span >= 9 (if they were consecutive).
    # But they must be multiples of specific divisors, so the span might be larger.
    
    # Let's use a binary search on S, and for each S, check if any window works.
    # For a given S, we need to find L such that for every divisor d,
    # there exists k with L <= k*d <= L+S, and all chosen values are distinct.
    
    # To check feasibility for a given (L, S), we use backtracking:
    # - For each divisor d, compute the set of valid k values.
    # - Assign values greedily or via backtracking, ensuring distinctness.
    
    # Optimization: process divisors in order of fewest options first.
    
    def check_window(L, S):
        """Check if we can assign distinct multiples for all divisors within [L, L+S]."""
        if L < 1:
            return None
        if L + S > P['maximum']:
            return None
        
        # For each divisor, find the range of valid k values
        ranges = []
        for d in divisors:
            k_min = (L + d - 1) // d  # ceil(L/d)
            k_max = (L + S) // d       # floor((L+S)/d)
            if k_min < 1:
                k_min = 1
            if k_min > k_max:
                return None  # No valid multiple for this divisor
            ranges.append((d, k_min, k_max))
        
        # Sort by number of options (most constrained first)
        ranges.sort(key=lambda x: x[2] - x[1])
        
        assigned = {}
        used_values = set()
        
        def backtrack(idx):
            if idx == n:
                return True
            d, k_min, k_max = ranges[idx]
            for k in range(k_min, k_max + 1):
                val = k * d
                if val not in used_values:
                    used_values.add(val)
                    assigned[d] = val
                    if backtrack(idx + 1):
                        return True
                    used_values.remove(val)
            return False
        
        if backtrack(0):
            # Reconstruct in original order
            result = []
            for d in divisors:
                result.append(assigned[d])
            return result
        return None
    
    # Now, let's search for the minimum span S.
    # We know S >= 9 (need 10 distinct integers).
    # Let's try S from 9 upward, and for each S, try various L values.
    
    # For a given S, which L values should we try?
    # The window [L, L+S] must contain at least one multiple of each divisor.
    # This is a feasibility condition.
    
    # Key insight: for the window to be feasible, for each divisor d,
    # we need floor((L+S)/d) - ceil(L/d) + 1 >= 1.
    # This is equivalent to: there exists k such that L <= k*d <= L+S.
    
    # Let's iterate S from 9 to some upper bound, and for each S,
    # iterate L over a reasonable range.
    
    # But the range of L is huge (up to 1e6). We need to be smarter.
    
    # Observation: if a window [L, L+S] works, then for any L' that is congruent
    # to L mod lcm(divisors), the window [L', L'+S] has the same feasibility pattern.
    # So we only need to check L in [1, lcm(divisors)].
    
    # But lcm(divisors) might be large. Let's compute it.
    # divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    # LCM = 6 * 10 * 14 * 15 * 21 * 22 * 26 * 33 * 35 * 39 / (common factors)
    
    # Let's just compute it
    l = divisors[0]
    for d in divisors[1:]:
        l = lcm(l, d)
    
    # l might be large, but let's see. If l is too large, we'll use a different strategy.
    
    best_ans = divisors[:]  # trivial fallback
    best_span = _span(best_ans)
    
    # Strategy 1: For small S, try L in [1, min(l, some_limit)]
    # For larger S, the window is more likely to be feasible, so we can be more aggressive.
    
    max_L_to_try = min(l, 50000)  # Limit to keep runtime reasonable
    
    # Actually, let's think about this differently.
    # For a given S, the window [L, L+S] is feasible iff for each divisor d,
    # the interval [L/d, (L+S)/d] contains an integer.
    
    # Let's try a different approach: iterate over possible "anchor" values.
    # Pick one divisor (say the largest, 39) and try each multiple of 39 as the minimum.
    # Then for the other divisors, find the smallest multiple >= anchor that's distinct.
    
    # This is a greedy approach that might work well.
    
    def greedy_from_anchor(anchor, divisors_order=None):
        """Given an anchor (minimum value), greedily assign multiples to each divisor."""
        if divisors_order is None:
            # Process in order of increasing divisor (more options first)
            divisors_order = sorted(range(n), key=lambda i: divisors[i])
        
        used = set()
        result = [None] * n
        used.add(anchor)
        
        # Find which divisor has anchor as a multiple
        for i, d in enumerate(divisors):
            if anchor % d == 0:
                result[i] = anchor
                break
        
        for i in divisors_order:
            if result[i] is not None:
                continue
            d = divisors[i]
            # Find smallest multiple of d >= anchor not in used
            k = (anchor + d - 1) // d
            if k < 1:
                k = 1
            val = k * d
            while val in used or val > P['maximum']:
                k += 1
                val = k * d
                if val > P['maximum']:
                    return None
            result[i] = val
            used.add(val)
        
        return result
    
    # Try greedy from various anchors
    # Anchors should be multiples of at least one divisor
    # Let's try multiples of the largest divisor (39) and also multiples of other divisors
    
    anchor_candidates = set()
    # Multiples of 39
    for k in range(1, 1000):
        anchor_candidates.add(k * 39)
    # Multiples of other divisors
    for d in divisors:
        for k in range(1, 100):
            anchor_candidates.add(k * d)
    
    for anchor in sorted(anchor_candidates):
        if time.time() > deadline:
            break
        if anchor > P['maximum']:
            break
        ans = greedy_from_anchor(anchor)
        if ans is not None and _is_valid_answer(ans, divisors):
            s = _span(ans)
            if s < best_span:
                best_span = s
                best_ans = ans[:]
    
    # Now try the systematic window search for small S
    # For S from 9 to best_span - 1, check if any window works
    
    for S in range(9, best_span):
        if time.time() > deadline:
            break
        
        # For this S, try L values
        # L should be such that the window is feasible
        # Let's try L in [1, min(l, 100000)] with step 1 for small S, larger step for larger S
        
        if S < 20:
            l_range = range(1, min(l, 50000) + 1, 1)
        elif S < 50:
            l_range = range(1, min(l, 50000) + 1, 2)
        else:
            l_range = range(1, min(l, 50000) + 1, 5)
        
        found = False
        for L in l_range:
            if time.time() > deadline:
                break
            if L + S > P['maximum']:
                break
            
            # Quick feasibility check
            feasible = True
            for d in divisors:
                k_min = (L + d - 1) // d
                k_max = (L + S) // d
                if k_min < 1:
                    k_min = 1
                if k_min > k_max:
                    feasible = False
                    break
            if not feasible:
                continue
            
            ans = check_window(L, S)
            if ans is not None:
                best_span = S
                best_ans = ans[:]
                found = True
                break
        
        if found:
            # We found a solution with span S, which is an improvement
            # Continue to see if we can find an even smaller S
            pass
    
    # Also try random local search
    if time.time() < deadline - 5:
        rng = random.Random(42)
        for _ in range(5000):
            if time.time() > deadline - 5:
                break
            
            # Pick a random anchor
            d = divisors[rng.randint(0, n-1)]
            k = rng.randint(1, P['maximum'] // d)
            anchor = k * d
            
            ans = greedy_from_anchor(anchor)
            if ans is not None and _is_valid_answer(ans, divisors):
                s = _span(ans)
                if s < best_span:
                    best_span = s
                    best_ans = ans[:]
    
    # Final validation
    if not _is_valid_answer(best_ans, divisors):
        best_ans = divisors[:]
    
    # Ensure it's a list of Python ints
    result = [int(v) for v in best_ans]
    return result

# EVOLVE-BLOCK-END
