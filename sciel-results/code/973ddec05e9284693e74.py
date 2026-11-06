import math
import time

def construct():
    T12 = math.factorial(12)
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    best_max = T12
    best_solution = [T12]
    
    # Precompute all divisors of 12! that are > 12
    def get_divisors(n):
        divisors = set()
        for i in range(13, int(math.isqrt(n)) + 1):
            if n % i == 0:
                divisors.add(i)
                divisors.add(n // i)
        return sorted(divisors)
    
    # For each k, find the minimum possible max factor
    # We'll use a meet-in-the-middle approach for small k, and DFS with strong pruning for larger k
    
    def can_form_product(remaining, min_val, count_needed, max_allowed):
        """Check if we can form 'remaining' as product of 'count_needed' distinct factors 
        each >= min_val and <= max_allowed"""
        if count_needed == 0:
            return remaining == 1
        if count_needed == 1:
            return min_val <= remaining <= max_allowed and remaining >= min_val
        
        # Lower bound: product of smallest 'count_needed' values >= min_val
        min_prod = 1
        for i in range(count_needed):
            min_prod *= (min_val + i)
            if min_prod > remaining:
                return False
        
        # Upper bound: product of largest possible values <= max_allowed
        max_prod = 1
        for i in range(count_needed):
            max_prod *= (max_allowed - i)
            if max_prod < remaining:
                return False
        
        return True
    
    def search_factors(remaining, min_val, count_needed, current_factors, max_allowed):
        """Recursively find 'count_needed' distinct factors >= min_val with product = remaining,
        all factors <= max_allowed"""
        nonlocal best_max, best_solution
        
        if time.time() > deadline:
            return False
            
        if count_needed == 0:
            if remaining == 1:
                max_f = max(current_factors)
                if max_f < best_max:
                    best_max = max_f
                    best_solution = list(current_factors)
                    return True
            return False
            
        if count_needed == 1:
            if min_val <= remaining <= max_allowed:
                current_factors.append(remaining)
                max_f = max(current_factors)
                if max_f < best_max:
                    best_max = max_f
                    best_solution = list(current_factors)
                current_factors.pop()
                return max_f < best_max
            return False
        
        # Pruning: check if it's even possible
        if not can_form_product(remaining, min_val, count_needed, max_allowed):
            return False
        
        # Try divisors of remaining that are >= min_val and <= max_allowed
        # Only try divisors that could lead to a valid solution
        divisors_to_try = []
        for d in range(min_val, min(max_allowed, remaining) + 1):
            if remaining % d == 0:
                # Check if remaining//d can be formed by (count_needed-1) factors > d
                remaining_after = remaining // d
                if can_form_product(remaining_after, d + 1, count_needed - 1, max_allowed):
                    divisors_to_try.append(d)
        
        # Try each divisor
        for d in divisors_to_try:
            current_factors.append(d)
            if search_factors(remaining // d, d + 1, count_needed - 1, current_factors, max_allowed):
                current_factors.pop()
                return True
            current_factors.pop()
            
        return False
    
    # Try different numbers of factors from 12 down to 1
    # More factors generally allow smaller maximum
    for k in range(12, 0, -1):
        if time.time() > deadline:
            break
            
        # Theoretical lower bound on max factor for k factors
        # Each factor must be > 12, so minimum possible product with k factors is 13*14*...*12+k
        # If this exceeds 12!, skip this k
        min_possible = 1
        for i in range(k):
            min_possible *= (13 + i)
            if min_possible > T12:
                break
        if min_possible > T12:
            continue
        
        # Try to find solution with k factors, minimizing max
        # Start with a loose bound and tighten
        # Initial bound: T12 (trivial)
        # Better initial bound: use the k-th root
        initial_max = int(T12 ** (1.0 / k)) + 100
        
        # Binary search on the maximum factor
        low = 13
        high = initial_max
        
        # First, check if any solution exists with high
        if not search_factors(T12, 13, k, [], high):
            # No solution with this max, try larger
            # But we want to minimize max, so if no solution with high, we need larger
            # Actually, we want the smallest max that allows a solution
            # So we should start with a small max and increase until we find a solution
            pass
        
        # Let's do a linear search from small to large max
        # Start from a reasonable lower bound
        lower_bound = int(T12 ** (1.0 / k)) + 1
        for max_val in range(lower_bound, initial_max + 1):
            if time.time() > deadline:
                break
                
            if search_factors(T12, 13, k, [], max_val):
                # Found a solution with max_val
                if max_val < best_max:
                    best_max = max_val
                break  # Found the minimum max for this k, move to next k
        
        # If we found a good solution, we can stop early
        # More factors (larger k) should give smaller max
        # So if k=12 works, that's likely optimal
        if k >= 10 and best_max < T12:
            break
    
    return best_solution
