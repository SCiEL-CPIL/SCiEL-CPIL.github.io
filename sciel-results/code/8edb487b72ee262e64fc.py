import time
from fractions import Fraction
from math import gcd

def construct():
    deadline = time.time() + 55
    
    # Start with known solution
    best = [8, 31, 248]
    best_max = 248
    
    def is_valid_solution(dens):
        s = Fraction(0, 1)
        for d in dens:
            s += Fraction(1, d)
        return s == Fraction(5, 31)
    
    def max_sum_with_terms(min_d, max_d, count):
        """Maximum sum of 'count' distinct reciprocals in (min_d, max_d]."""
        # The max sum is 1/(min_d+1) + 1/(min_d+2) + ... + 1/(min_d+count)
        # But capped by max_d
        total = 0
        for i in range(count):
            d = min_d + 1 + i
            if d > max_d:
                break
            total += Fraction(1, d)
        return total
    
    def min_sum_with_terms(min_d, max_d, count):
        """Minimum sum of 'count' distinct reciprocals in (min_d, max_d]."""
        # The min sum is 1/max_d + 1/(max_d-1) + ... + 1/(max_d-count+1)
        total = 0
        for i in range(count):
            d = max_d - i
            if d <= min_d:
                break
            total += Fraction(1, d)
        return total
    
    def dfs(rem_num, rem_den, min_d, max_d, max_terms, current, best_found):
        """
        DFS to find Egyptian fraction expansion.
        rem_num/rem_den: remaining sum to achieve
        min_d: minimum denominator for next term (exclusive)
        max_d: maximum denominator allowed
        max_terms: maximum number of terms we can still add
        current: list of chosen denominators
        best_found: [max_denominator] - updated in place if better solution found
        Returns: list of denominators if solution found, else None
        """
        if time.time() > deadline:
            return None
        
        if rem_num == 0:
            if current:
                cur_max = max(current)
                if cur_max < best_found[0]:
                    best_found[0] = cur_max
                    return current[:]
            return None
        
        if rem_num < 0:
            return None
        
        if max_terms == 0:
            return None
        
        # Check if rem is exactly 1/d for some valid d
        if rem_num == 1:
            d = rem_den
            if min_d < d <= max_d and d not in current:
                cur_max = max(current + [d])
                if cur_max < best_found[0]:
                    best_found[0] = cur_max
                    return current + [d]
            return None
        
        # Pruning: check if remaining sum is achievable
        # Max possible sum with max_terms terms in (min_d, max_d]
        max_possible = max_sum_with_terms(min_d, max_d, max_terms)
        if Fraction(rem_num, rem_den) > max_possible:
            return None
        
        # Min possible sum with max_terms terms in (min_d, max_d]
        min_possible = min_sum_with_terms(min_d, max_d, max_terms)
        if Fraction(rem_num, rem_den) < min_possible:
            return None
        
        # Determine range for next denominator d
        # 1/d < rem => d > rem_den/rem_num
        # Also d > min_d
        # And d <= max_d
        
        # d must satisfy: 1/d < rem_num/rem_den => d > rem_den/rem_num
        # So d >= floor(rem_den/rem_num) + 1
        floor_val = rem_den // rem_num
        if rem_den % rem_num == 0:
            d_min = max(min_d + 1, floor_val + 1)
        else:
            d_min = max(min_d + 1, floor_val + 1)
        
        # Upper bound on d: we need the remaining (max_terms-1) terms to be able to
        # sum to rem - 1/d. The minimum sum of (max_terms-1) terms in (d, max_d] is
        # 1/max_d + 1/(max_d-1) + ... (max_terms-1 terms)
        # So rem - 1/d >= min_sum_with_terms(d, max_d, max_terms-1)
        # 1/d <= rem - min_sum_with_terms(d, max_d, max_terms-1)
        # This is hard to solve directly, so we just iterate.
        
        # Also, 1/d must be < rem, and rem - 1/d must be >= min_sum_with_terms(d, max_d, max_terms-1)
        # And rem - 1/d must be <= max_sum_with_terms(d, max_d, max_terms-1)
        
        for d in range(d_min, max_d + 1):
            if time.time() > deadline:
                return None
            
            if d in current:
                continue
            
            # Check 1/d < rem
            if rem_num * d <= rem_den:
                continue
            
            rem2_num = rem_num * d - rem_den
            rem2_den = rem_den * d
            
            # Simplify
            g = gcd(rem2_num, rem2_den)
            rn = rem2_num // g
            rd = rem2_den // g
            
            # Check if remaining sum is achievable
            if rn == 0:
                # We've exactly reached the target
                cur_max = max(current + [d])
                if cur_max < best_found[0]:
                    best_found[0] = cur_max
                    return current + [d]
                continue
            
            if max_terms - 1 == 0:
                continue
            
            # Pruning for remaining
            max_possible_rem = max_sum_with_terms(d, max_d, max_terms - 1)
            if Fraction(rn, rd) > max_possible_rem:
                # Since d is increasing, 1/d is decreasing, so rem2 = rem - 1/d is increasing
                # We can continue
                continue
            
            min_possible_rem = min_sum_with_terms(d, max_d, max_terms - 1)
            if Fraction(rn, rd) < min_possible_rem:
                # rem2 is too small, and as d increases, rem2 increases, so we can continue
                continue
            
            current.append(d)
            result = dfs(rn, rd, d, max_d, max_terms - 1, current, best_found)
            current.pop()
            
            if result is not None:
                return result
        
        return None
    
    # Binary search on C (max denominator)
    # Lower bound: we need at least 1 term, and 1/d <= 5/31 => d >= 31/5 = 6.2, so d >= 7
    # But we need multiple terms. The minimum possible max denominator...
    # Let's just start from a reasonable lower bound and binary search.
    
    # Actually, let's just do a direct search with the DFS, trying to minimize the max.
    # We'll use the DFS with a decreasing max_d bound.
    
    # Try max_d from 247 down to some lower bound
    # But this might be slow. Let's try a few values.
    
    # First, let's try with max_terms = 10 and max_d = 247
    best_found = [best_max]
    result = dfs(5, 31, 0, best_max - 1, 10, [], best_found)
    if result is not None:
        if is_valid_solution(result):
            best = result
            best_max = max(best)
    
    # Try with more terms if time permits
    if time.time() < deadline - 5:
        best_found = [best_max]
        result = dfs(5, 31, 0, best_max - 1, 15, [], best_found)
        if result is not None:
            if is_valid_solution(result):
                best = result
                best_max = max(best)
    
    if time.time() < deadline - 5:
        best_found = [best_max]
        result = dfs(5, 31, 0, best_max - 1, 20, [], best_found)
        if result is not None:
            if is_valid_solution(result):
                best = result
                best_max = max(best)
    
    return best
