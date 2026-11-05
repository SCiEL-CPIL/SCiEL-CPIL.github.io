def construct():
    import time
    from math import factorial
    from itertools import combinations
    
    start_time = time.time()
    DEADLINE = 80  # seconds, safety margin before 90s kill
    
    target = factorial(12)  # 479001600
    
    # Factorize target
    # 12! = 2^10 * 3^5 * 5^2 * 7 * 11
    # We need to partition prime factors into distinct groups > 12, minimize max
    
    # Try different numbers of factors from 12 down to 1
    # For each, try to find a valid partition
    
    best_solution = None
    best_max = target + 1
    
    # Precompute: what are the possible "chunks" we can form?
    # A chunk is a divisor of target that is > 12
    # But we need to be smarter - we need distinct chunks whose product is target
    
    # Strategy: use DFS to assign prime factors to bins
    # Prime factorization: 2^10, 3^5, 5^2, 7^1, 11^1
    # Total prime factors (with multiplicity): 10+5+2+1+1 = 19
    
    # For a given number of bins k, we need to partition the 19 prime factors into k groups
    # such that each group's product > 12, all distinct, and we minimize the max product
    
    # This is still complex. Let me try a different approach:
    # Generate candidate divisors > 12, then search for subsets that multiply to target
    
    # Actually, let me try a meet-in-the-middle or iterative deepening approach
    
    # Simpler: try specific factor counts and use a greedy + local search approach
    
    # Let me try to find solutions for k = 12, 11, 10, ... down to 1
    # and pick the one with minimum max factor
    
    # For efficiency, let me generate all divisors of 12! that are > 12
    # 12! has (10+1)*(5+1)*(2+1)*(1+1)*(1+1) = 11*6*3*2*2 = 792 divisors
    # That's manageable
    
    def get_divisors(n):
        divs = []
        i = 1
        while i * i <= n:
            if n % i == 0:
                divs.append(i)
                if i != n // i:
                    divs.append(n // i)
            i += 1
        return sorted(divs)
    
    divisors = get_divisors(target)
    # Filter to > 12
    divs_gt12 = [d for d in divisors if d > 12]
    
    # Now search for subsets of divs_gt12 that multiply to target, are distinct, and minimize max
    # This is still hard in general, but let's try a DFS with pruning for small k
    
    # Actually, let me try a different strategy:
    # For each k from 12 down to 1, try to find k distinct divisors > 12 whose product is target
    # Use DFS with pruning: sort divisors, and at each step, the remaining product must be achievable
    
    # Better approach: since we want to minimize the max, let's binary search on the max value M
    # and check if there exist k distinct divisors in (12, M] whose product is target
    
    # But even simpler: let me just try to find good solutions directly
    
    # Let me try a heuristic: start with small factors > 12 and build up
    
    # Factors > 12 that divide 12!:
    # 13 doesn't divide 12! (13 > 12)
    # 14 = 2*7, divides 12!
    # 15 = 3*5, divides 12!
    # 16 = 2^4, divides 12!
    # 18 = 2*3^2, divides 12!
    # 20 = 2^2*5, divides 12!
    # 21 = 3*7, divides 12!
    # 22 = 2*11, divides 12!
    # 24 = 2^3*3, divides 12!
    # 25 = 5^2, divides 12!
    # 27 = 3^3, divides 12!
    # 28 = 2^2*7, divides 12!
    # 30 = 2*3*5, divides 12!
    # 32 = 2^5, divides 12!
    # 33 = 3*11, divides 12!
    # 35 = 5*7, divides 12!
    # 36 = 2^2*3^2, divides 12!
    # 40 = 2^3*5, divides 12!
    # 42 = 2*3*7, divides 12!
    # 44 = 2^2*11, divides 12!
    # 45 = 3^2*5, divides 12!
    # 48 = 2^4*3, divides 12!
    # 49 = 7^2 - NO, 7^2 doesn't divide 12! (only one 7)
    # 50 = 2*5^2, divides 12!
    # 54 = 2*3^3, divides 12!
    # 55 = 5*11, divides 12!
    # 56 = 2^3*7, divides 12!
    # 60 = 2^2*3*5, divides 12!
    # etc.
    
    # Let me try a DFS approach for finding k factors
    # State: (remaining_product, min_factor_index, current_factors, max_so_far)
    # We want to pick factors from divs_gt12 in increasing order to ensure distinctness
    
    # For efficiency, let's limit the search
    
    # Let me try k from 12 down to 1, and for each k, do a DFS
    
    # Actually, let me just try to find ANY valid solution first, then optimize
    
    # Quick check: can we do it with 12 factors?
    # We need 12 distinct integers > 12 whose product is 12!
    # The smallest 12 distinct integers > 12 are 13,14,15,...,24
    # But 13 doesn't divide 12!, so this won't work directly
    
    # Let me try a computational approach with time limit
    
    import sys
    sys.setrecursionlimit(10000)
    
    def dfs(remaining, min_idx, chosen, max_val):
        nonlocal best_solution, best_max
        if time.time() - start_time > DEADLINE:
            return
        
        if remaining == 1:
            if chosen:
                m = max(chosen)
                if m < best_max:
                    best_max = m
                    best_solution = chosen[:]
            return
        
        # Pruning: if max(chosen) already >= best_max, we can still continue if we might find better
        # But if the smallest possible remaining factor would make max > best_max, prune
        # Actually, let's just track and compare at the end
        
        for i in range(min_idx, len(divs_gt12)):
            d = divs_gt12[i]
            if d > remaining:
                break
            if remaining % d != 0:
                continue
            if d <= 12:
                continue
            # Check distinctness: since we're going in increasing order, d > all previous
            # (because min_idx ensures we only pick larger indices)
            
            # Pruning: if d >= best_max and we already have factors, the max will be at least d
            # But we might still find a solution with smaller max by not using d
            # So we can't prune based on d alone unless d >= best_max and we're forced to use larger
            
            # Upper bound pruning: the remaining product after picking d is remaining//d
            # We need to form this from factors > d (since distinct and increasing)
            # The minimum product of (k-1) factors all > d is (d+1)*(d+2)*...*(d+k-1)
            # But we don't know k in advance. Let's skip this complex pruning.
            
            chosen.append(d)
            dfs(remaining // d, i + 1, chosen, max(max_val, d))
            chosen.pop()
    
    # Start DFS
    dfs(target, 0, [], 0)
    
    if best_solution:
        return best_solution
    
    # Fallback: single factor
    return [target]
