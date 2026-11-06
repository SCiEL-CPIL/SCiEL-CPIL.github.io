import time
from math import factorial

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    target = factorial(12)

    # Get all divisors of target
    divisors = []
    for i in range(1, int(target**0.5) + 1):
        if target % i == 0:
            divisors.append(i)
            if i != target // i:
                divisors.append(target // i)

    # Filter valid divisors: > 12
    valid_divs = sorted([d for d in divisors if d > 12])
    n_divs = len(valid_divs)

    # Precompute a set for O(1) lookup
    div_set = set(valid_divs)

    def find_solution(M, max_factors=12):
        """Find a decomposition of target into distinct factors in (12, M]."""
        # Candidates are valid divisors <= M
        # Since valid_divs is sorted ascending, find the rightmost index where valid_divs[i] <= M
        # Use binary search
        lo, hi = 0, n_divs
        while lo < hi:
            mid = (lo + hi) // 2
            if valid_divs[mid] <= M:
                lo = mid + 1
            else:
                hi = mid
        # lo is the first index where valid_divs[lo] > M
        # So candidates are valid_divs[:lo]
        end_idx = lo

        if end_idx == 0:
            return None

        res = []

        def backtrack(idx, remaining, count):
            if remaining == 1:
                return True
            if count >= max_factors:
                return False
            if idx >= end_idx:
                return False

            # Pruning: if the smallest remaining factor (valid_divs[idx]) is greater than remaining, 
            # we can't use it or any larger factor.
            if valid_divs[idx] > remaining:
                return False

            # Pruning: if remaining is prime, we must pick it if it's a candidate and > 12
            # (Check if remaining is in div_set and valid_divs[idx] <= remaining <= M)
            
            # Iterate from smallest to largest? Or largest to smallest?
            # Trying smaller factors first might lead to more factors, which is good for minimizing the max?
            # No, we just need ANY valid set. The set of factors is determined by the divisors.
            # However, to find a solution quickly, we might want to try factors that divide remaining.
            
            # Let's iterate from end_idx-1 down to idx? 
            # If we pick large factors first, we reduce 'remaining' faster.
            # But we are constrained by M.
            
            # Let's try iterating from idx to end_idx.
            # We must ensure distinctness, which is handled by idx increasing.
            
            # Optimization: If remaining is in div_set and remaining <= M and remaining >= valid_divs[idx], 
            # we can check if it's the last factor needed? No, remaining might be composite.
            
            for i in range(idx, end_idx):
                c = valid_divs[i]
                if c > remaining:
                    break
                if remaining % c == 0:
                    res.append(c)
                    if backtrack(i + 1, remaining // c, count + 1):
                        return True
                    res.pop()
            
            return False

        if backtrack(0, target, 0):
            return res
        return None

    # Strategy:
    # 1. Try to find a solution with a small number of factors (e.g., 5, 6) which often have smaller max factors?
    #    Actually, for a fixed product, fewer factors means larger average factor.
    #    So we should aim for MORE factors to keep the max factor small?
    #    Wait. If we split into more factors, the factors get smaller.
    #    Example: 12! = 479001600.
    #    If we use 1 factor: 479001600.
    #    If we use 2 factors: min max is sqrt(12!) ~ 21886.
    #    If we use 5 factors: min max is (12!)^(1/5) ~ 137.
    #    So we WANT many factors to minimize the maximum.
    #    The constraint is distinct integers > 12.
    #    So we should try to find a solution with as many factors as possible, or at least a solution that exists for a small M.
    
    # We can binary search on M.
    # Lower bound: The smallest possible max factor.
    # If we use k factors, min max >= (12!)^(1/k).
    # Max k is 12. (12!)^(1/12) ~ 7.5, but factors must be > 12.
    # So the theoretical lower bound for M is at least 13.
    # But can we have 12 distinct integers > 12 with product 12!?
    # The smallest 12 distinct integers > 12 are 13, 14, ..., 24.
    # Product = 24! / 12! = 25 * 26 * ... * 24? No.
    # 13*14*...*24.
    # Let's check if 12! can be decomposed into 12 distinct integers > 12.
    # The product of 13..24 is much larger than 12!.
    # 12! = 479,001,600.
    # 13*14*...*24 = 24! / 12!.
    # 24! is huge.
    # So we cannot have 12 factors.
    
    # We should find the maximum number of factors possible, or just search for M.
    # Binary search on M is robust.
    
    low = 13
    high = target
    
    best_sol = None
    
    # First, do a quick check for very large M to ensure we have a fallback?
    # Actually, just binary search.
    
    # To speed up, we can first try to find a solution with a moderate M, say 1000, 10000, etc.
    # But binary search is fine.
    
    while low <= high:
        if time.time() > deadline - 5.0:
            break
        mid = (low + high) // 2
        sol = find_solution(mid)
        if sol is not None:
            best_sol = sol
            high = mid - 1
        else:
            low = mid + 1
    
    if best_sol is not None:
        return best_sol
    
    # Fallback if search fails (should not happen, as M=target always works)
    return [target]
