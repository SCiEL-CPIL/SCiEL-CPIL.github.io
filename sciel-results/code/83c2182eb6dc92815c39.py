import time
from math import factorial

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s limit, keep 10s buffer
    
    target = factorial(12)  # 479001600
    
    # We need 1..12 distinct integers all > 12, product = 12!, minimize the largest factor.
    # 
    # 12! = 479001600
    # Prime factorization: 2^10 * 3^5 * 5^2 * 7 * 11
    # 
    # Strategy: We want to minimize the maximum factor. 
    # If we use k factors, their product is 12!.
    # The geometric mean of k factors is (12!)^(1/k).
    # To minimize the max, we want the factors to be as balanced as possible, 
    # but they must be distinct integers > 12.
    #
    # Let's think about how many factors to use.
    # If we use 1 factor: max = 479001600
    # If we use 2 factors: min max is at least sqrt(479001600) ≈ 21886
    # If we use 3 factors: min max is at least cube root ≈ 782
    # If we use 4 factors: min max is at least 4th root ≈ 83
    # If we use 5 factors: min max is at least 5th root ≈ 34.4
    # If we use 6 factors: min max is at least 6th root ≈ 21.9
    # If we use 7 factors: min max is at least 7th root ≈ 16.2
    # If we use 8 factors: min max is at least 8th root ≈ 13.1
    # If we use 9 factors: min max is at least 9th root ≈ 11.5 (but all must be > 12)
    # If we use 10 factors: min max is at least 10th root ≈ 10.5 (but all must be > 12)
    #
    # Since all factors must be > 12, the minimum possible value for each factor is 13.
    # For k factors, the minimum product is 13 * 14 * 15 * ... * (12+k) (using consecutive integers starting from 13).
    # Let's check what k values are feasible:
    # k=1: min product = 13, max product = 479001600. Feasible if 479001600 > 12. Yes.
    # k=2: min product = 13*14=182. Feasible.
    # k=3: min product = 13*14*15=2730. Feasible.
    # k=4: min product = 13*14*15*16=43680. Feasible.
    # k=5: min product = 13*14*15*16*17=742560. Feasible.
    # k=6: min product = 13*14*15*16*17*18=13366080. Feasible.
    # k=7: min product = 13*14*15*16*17*18*19=253955520. Feasible.
    # k=8: min product = 13*14*15*16*17*18*19*20=5079110400 > 479001600. NOT feasible!
    #
    # So we can use at most 7 factors.
    #
    # Let's analyze the optimal solution for each k from 7 down to 1, and find the overall minimum max factor.
    #
    # For k=7: We need 7 distinct integers > 12 with product 479001600.
    # The minimum possible max factor for 7 numbers with product P is when they are as equal as possible.
    # (479001600)^(1/7) ≈ 16.17
    # So the optimal 7 factors would be around 13, 14, 15, 16, 16, 17, 18 or similar.
    # But they must be distinct.
    # Let's try to find 7 distinct integers > 12 with product 479001600, minimizing the max.
    # 
    # We can use a search: try all combinations of 7 distinct integers from 13 to some upper bound, 
    # but that's too slow. Instead, use a smarter approach.
    #
    # Alternatively, we can note that the factors must divide 12!.
    # Let's enumerate the divisors of 12! that are > 12 and try to find a set of 7 (or fewer) distinct ones 
    # whose product is 12!, minimizing the maximum.
    #
    # Given the time constraint, let's use a precomputed optimal solution.
    # From known results or careful computation:
    # 
    # Let's try k=7: 
    # We need 7 distinct factors > 12.
    # The product of 13*14*15*16*17*18*19 = 253955520
    # We need product 479001600.
    # 479001600 / 253955520 ≈ 1.886
    # So we need to replace some factors with larger ones.
    # 
    # Let me try a different approach: use the fact that the factors must be composed of the prime factors of 12!.
    # 12! = 2^10 * 3^5 * 5^2 * 7 * 11
    #
    # Let's try to find a good solution by hand or with a small search.
    # 
    # Actually, let's use a systematic search with pruning. We'll search over the number of factors k from 7 down to 1,
    # and for each k, try to find distinct factors > 12 with product 12!, minimizing the max.
    #
    # Given the complexity, let's hardcode a verified optimal solution.
    # After careful analysis, the optimal solution uses 7 factors:
    # 13, 14, 15, 16, 17, 18, 19 ... no, that product is too small.
    # 
    # Let me compute: 13*14*15*16*17*18*19 = 253955520
    # We need 479001600 / 253955520 = 1.886...
    # So we need to multiply by about 1.886.
    # If we replace 19 with 36: 13*14*15*16*17*18*36 = 482384640. Close but not exact.
    # 479001600 / (13*14*15*16*17*18) = 479001600 / 640237440... wait let me recalculate.
    # 13*14 = 182
    # 182*15 = 2730
    # 2730*16 = 43680
    # 43680*17 = 742560
    # 742560*18 = 13366080
    # 479001600 / 13366080 = 35.83...
    # So we need 1 more factor that is about 35.83, but we already used 13-18, so the 7th factor would be 36 (if distinct and > 18).
    # 13366080 * 36 = 481178880. Not 479001600.
    # 
    # Let me try a different combination.
    # 
    # This is getting complex. Let's use a programmatic search with a tight time budget.
    
    # Generate all divisors of target that are > 12
    divisors = []
    for i in range(13, int(target**0.5) + 1):
        if target % i == 0:
            divisors.append(i)
            other = target // i
            if other != i and other > 12:
                divisors.append(other)
    
    # Also add target itself if > 12
    if target > 12:
        divisors.append(target)
    
    divisors = sorted(set(divisors))
    
    # We'll search for the best solution using DFS with pruning.
    # We want to minimize the maximum factor.
    # We'll try all possible numbers of factors k from 7 down to 1.
    # For each k, we'll try to find k distinct divisors > 12 whose product is target.
    # We'll use a priority queue or iterative deepening to find the minimum max factor.
    
    # Given the time constraint, let's use a simpler approach:
    # Try all combinations of divisors (up to 7 at a time) and check if their product is target.
    # But this is O(n^7) which is too slow.
    
    # Better approach: use a memoized DFS that tracks the remaining product and the set of used factors.
    # But the state space is large.
    
    # Let's try a different strategy: iterate over possible maximum factor M, and check if there exists
    # a set of distinct divisors in [13, M] whose product is target.
    # We can do this with a DFS that picks factors in increasing order.
    
    # Find the minimum M such that there exist distinct divisors in [13, M] with product target.
    
    # First, let's find all divisors of target that are >= 13.
    divs = sorted(set(d for d in divisors if d >= 13))
    
    # We'll use a DFS that picks factors in increasing order, ensuring distinctness and product = target.
    # We'll track the current product and the index in divs to ensure we only pick larger factors.
    
    # To minimize M, we can iterate M from 13 upwards, and for each M, check if there's a valid combination
    # using only divisors in [13, M].
    
    # But this could be slow. Let's use a smarter approach.
    
    # Let's try to find a solution with a small number of factors first.
    
    # For k=7, the minimum possible max is at least 17 (since we need 7 distinct integers > 12).
    # Let's try M from 17 upwards.
    
    best_solution = None
    best_max = target  # Start with the worst case (single factor)
    
    # We'll use a DFS with memoization to find if we can form the product using distinct factors from a given range.
    # But the state (remaining_product, min_factor_index) is large.
    
    # Let's try a different approach: generate all subsets of divisors (up to size 7) and check their product.
    # But there are too many divisors.
    
    # Let's use a heuristic: try combinations of divisors that are "close" to the geometric mean.
    
    # For k factors, the ideal value is (target)^(1/k).
    # For k=7: (479001600)^(1/7) ≈ 16.17
    # For k=6: (479001600)^(1/6) ≈ 21.88
    # For k=5: (479001600)^(1/5) ≈ 34.4
    # For k=4: (479001600)^(1/4) ≈ 83.0
    # For k=3: (479001600)^(1/3) ≈ 782
    # For k=2: (479001600)^(1/2) ≈ 21886
    # For k=1: 479001600
    
    # Let's try to find a solution for k=7 first, as it has the smallest potential max factor.
    
    # We'll search for 7 distinct divisors in [13, M] with product target, for increasing M.
    
    # To make this efficient, we can use a recursive search that picks factors in increasing order.
    
    def can_form_product(rem, start_idx, max_idx, current_factors):
        if rem == 1:
            if len(current_factors) >= 1:
                return True
            return False
        
        if time.time() > deadline:
            return False
        
        # If we've picked 7 factors and rem != 1, fail
        if len(current_factors) >= 7:
            return False
        
        for i in range(start_idx, max_idx + 1):
            d = divs[i]
            if d > rem:
                break
            if rem % d == 0:
                current_factors.append(d)
                if can_form_product(rem // d, i + 1, max_idx, current_factors):
                    return True
                current_factors.pop()
        
        return False
    
    # Try to find a solution with at most 7 factors, minimizing the max factor.
    # We'll iterate over the maximum allowed factor M (index in divs).
    
    for max_idx in range(len(divs)):
        M = divs[max_idx]
        current_factors = []
        if can_form_product(target, 0, max_idx, current_factors):
            if len(current_factors) <= 7:
                max_factor = max(current_factors)
                if max_factor < best_max:
                    best_max = max_factor
                    best_solution = list(current_factors)
            break  # We found the minimum M for a valid solution
    
    if best_solution is not None:
        return sorted(best_solution)
    
    # If no solution found with the above, try a different approach or return a known valid solution.
    # Let's try k=6, 5, etc.
    
    # For now, return a hardcoded valid solution if the above didn't work.
    # After further analysis, a valid solution is:
    # 13, 14, 15, 16, 17, 18, 19 ... no.
    
    # Let me try: 14 * 15 * 16 * 17 * 18 * 19 * 20 * 21 ... too many.
    
    # Given the time, let's return a known valid decomposition.
    # 12! = 2^10 * 3^5 * 5^2 * 7 * 11
    # One valid decomposition: 13, 14, 15, 16, 17, 18, 19, 20 ... no, we established max 7 factors.
    
    # Let's use: 15, 16, 17, 18, 19, 20, 21, 22 ... no.
    
    # I'll return a simple valid solution: [target] (1 factor), which is always valid.
    return [target]
