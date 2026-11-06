import time
from fractions import Fraction

def construct():
    deadline = time.time() + 50
    
    # Known solution: 5/31 = 1/8 + 1/31 + 1/248
    # Let's verify: 1/8 + 1/31 + 1/248 = 31/248 + 8/248 + 1/248 = 40/248 = 5/31. Correct.
    best = [8, 31, 248]
    best_max = 248
    
    target = Fraction(5, 31)
    
    # We want to find a set of distinct denominators d_1 < d_2 < ... < d_k
    # such that sum(1/d_i) = 5/31 and max(d_i) < best_max.
    # We will search for the minimum possible max denominator.
    
    # Precompute bounds to speed up DFS
    # max_sum(n, k, limit) = sum_{i=n}^{n+k-1} 1/i (upper bound on sum of k distinct terms >= n, < limit)
    # Actually, to maximize the sum, we pick the smallest available denominators.
    # So max_sum is sum of reciprocals of the k smallest integers >= n and < limit.
    
    # min_sum(n, k, limit) = sum of reciprocals of the k largest integers < limit and >= n.
    # To minimize the sum, we pick the largest available denominators.
    
    def max_possible_sum(n, k, limit):
        s = Fraction(0, 1)
        count = 0
        for i in range(n, limit):
            if count == k:
                break
            s += Fraction(1, i)
            count += 1
        return s

    def min_possible_sum(n, k, limit):
        s = Fraction(0, 1)
        count = 0
        for i in range(limit - 1, n - 1, -1):
            if count == k:
                break
            s += Fraction(1, i)
            count += 1
        return s

    def dfs(rem, min_d, max_terms, current, limit):
        nonlocal best, best_max
        
        if time.time() > deadline:
            return False
            
        if rem == 0:
            if current:
                m = max(current)
                if m < best_max:
                    best_max = m
                    best = current[:]
                    return True
            return False
            
        if max_terms == 0:
            return False
            
        # Pruning 1: Max sum
        if rem > max_possible_sum(min_d, max_terms, limit):
            return False
            
        # Pruning 2: Min sum
        if rem < min_possible_sum(min_d, max_terms, limit):
            return False
            
        # If only 1 term left
        if max_terms == 1:
            if rem.numerator == 1:
                d = rem.denominator
                if min_d <= d < limit:
                    current.append(d)
                    dfs(Fraction(0, 1), d + 1, 0, current, limit)
                    current.pop()
            return False

        # Iterate d from min_d to limit-1
        # Constraint: 1/d <= rem  => d >= 1/rem
        # Constraint: rem - 1/d <= max_possible_sum(d+1, max_terms-1, limit)
        #             1/d >= rem - max_possible_sum(d+1, max_terms-1, limit)
        #             d <= 1 / (rem - max_possible_sum(d+1, max_terms-1, limit))
        
        d_min = min_d
        d_max = limit - 1
        
        # Tighten d_min
        # d >= 1/rem
        if rem < 1:
            d_min = max(d_min, (rem.denominator + rem.numerator - 1) // rem.numerator)
        else:
            d_min = max(d_min, 1)
            
        if d_min > d_max:
            return False

        for d in range(d_min, d_max + 1):
            if time.time() > deadline:
                return False
                
            if Fraction(1, d) > rem:
                continue
                
            new_rem = rem - Fraction(1, d)
            if new_rem < 0:
                continue
                
            # Prune: new_rem must be <= max_possible_sum(d+1, max_terms-1, limit)
            if new_rem > max_possible_sum(d + 1, max_terms - 1, limit):
                continue
                
            # Prune: new_rem must be >= min_possible_sum(d+1, max_terms-1, limit)
            if new_rem < min_possible_sum(d + 1, max_terms - 1, limit):
                continue
                
            current.append(d)
            if dfs(new_rem, d + 1, max_terms - 1, current, limit):
                current.pop()
                return True
            current.pop()
            
        return False

    # We know 248 works. Let's see if we can do better.
    # The number of terms is likely small. 
    # 5/31 is approx 0.16129.
    # 1/6 = 0.1666 (too big)
    # 1/7 = 0.1428
    # 1/8 = 0.125
    # 
    # If we use 1 term, d = 31/5 = 6.2 (not integer)
    # If we use 2 terms: 1/d1 + 1/d2 = 5/31
    #   d1 > 31/5 = 6.2 => d1 >= 7
    #   1/d2 = 5/31 - 1/d1 = (5d1 - 31) / (31d1)
    #   d2 = 31d1 / (5d1 - 31)
    #   For d2 to be integer, 5d1 - 31 must divide 31d1.
    #   Let k = 5d1 - 31. Then d1 = (k+31)/5. k must be multiple of 5? No, d1 integer => k+31 div 5 => k div 5 (since 31 div 5 is 1).
    #   So k = 5m.
    #   d2 = 31(5m+31)/5 / 5m = 31(5m+31) / 25m.
    #   25m must divide 31(5m+31).
    #   31 is prime. 25 and 31 coprime. So m must divide 31.
    #   m = 1 or 31.
    #   If m=1: k=5, d1=(5+31)/5=7. d2 = 31(36)/25 = 1116/25 (not integer)
    #   If m=31: k=155, d1=(155+31)/5=37.2 (not integer)
    #   So no 2-term solution.
    
    # If we use 3 terms:
    #   We found [8, 31, 248]. Max 248.
    #   Can we find a 3-term solution with smaller max?
    #   1/d1 + 1/d2 + 1/d3 = 5/31
    #   d1 >= 7.
    #   If d1=7: 1/d2 + 1/d3 = 5/31 - 1/7 = (35-31)/217 = 4/217.
    #     d2 > 217/4 = 54.25 => d2 >= 55.
    #     1/d3 = 4/217 - 1/d2 = (4d2 - 217) / (217d2)
    #     d3 = 217d2 / (4d2 - 217)
    #     Let k = 4d2 - 217. d2 = (k+217)/4. k div 4.
    #     d3 = 217(k+217)/4 / k = 217(k+217) / 4k.
    #     4k must divide 217(k+217).
    #     217 = 7 * 31.
    #     k must divide 217*217 = 217^2? Not necessarily, but 4k | 217(k+217).
    #     Since gcd(k, k+217) divides 217, let g = gcd(k, 217).
    #     This search is small.
    #     Let's just run the DFS.
    
    # We will try to find a solution with max < 248.
    # We limit the number of terms to 6 (safe upper bound for this size).
    
    # Start with limit 247.
    if dfs(target, 1, 6, [], best_max - 1):
        return best

    # If no solution with max < 248, return 248.
    return best
