import time
from math import factorial, isqrt

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill
    
    target = factorial(12)  # 479001600 = 2^10 * 3^5 * 5^2 * 7 * 11
    
    # Prime factorization: 2^10 * 3^5 * 5^2 * 7 * 11
    primes = [2, 3, 5, 7, 11]
    exponents = [10, 5, 2, 1, 1]
    total_prime_factors = sum(exponents)  # 19
    
    best = [target]
    best_max = target
    
    def get_divisors_from_powers(remaining, lo, hi):
        """Generate divisors of remaining in [lo, hi] using its prime factorization."""
        # Factor remaining
        temp = remaining
        factors = []
        for p in primes:
            e = 0
            while temp % p == 0:
                temp //= p
                e += 1
            factors.append(e)
        
        # Generate all divisors via recursion
        divs = []
        def gen(idx, current):
            if idx == len(primes):
                if lo <= current <= hi:
                    divs.append(current)
                return
            for e in range(factors[idx] + 1):
                gen(idx + 1, current * (primes[idx] ** e))
        gen(0, 1)
        return sorted(divs)
    
    def lower_bound_max(remaining, min_factor, num_factors_left, used_set):
        """Lower bound on the maximum factor needed."""
        if num_factors_left == 0:
            return 0
        if num_factors_left == 1:
            return remaining
        
        # We need num_factors_left distinct factors >= min_factor
        # whose product is 'remaining'
        # Minimum possible max: if we use the smallest possible factors
        # and one gets the rest
        
        # Simple bound: geometric mean
        import math
        gm = remaining ** (1.0 / num_factors_left)
        # But factors must be >= min_factor and distinct
        # If gm < min_factor, we need at least min_factor for the smallest
        # The max must be at least remaining / (product of (num_factors_left-1) smallest distinct factors >= min_factor)
        
        # Try to assign the smallest possible distinct factors
        # to minimize the last (largest) factor
        assigned = []
        cur_min = min_factor
        for i in range(num_factors_left - 1):
            # Find smallest divisor of remaining that's >= cur_min and not used
            # For a bound, just use cur_min as approximation
            assigned.append(cur_min)
            cur_min += 1
        
        # The remaining factor is remaining / product(assigned)
        # But this is a lower bound approximation
        prod_assigned = 1
        for a in assigned:
            prod_assigned *= a
        if prod_assigned > remaining:
            return min_factor  # can't even fit
        last = remaining // prod_assigned
        # But last must also be >= cur_min
        return max(last, cur_min)
    
    def search(remaining, min_factor, current_factors, max_so_far, used_set):
        nonlocal best, best_max, start_time
        
        if time.time() - start_time > DEADLINE:
            return
        
        if remaining == 1:
            if len(current_factors) >= 1:
                mx = max(current_factors)
                if mx < best_max:
                    best_max = mx
                    best = list(current_factors)
            return
        
        if len(current_factors) >= 12:
            return
        
        if max_so_far >= best_max:
            return
        
        min_possible = max(min_factor, 13)
        
        if remaining < min_possible:
            return
        
        # Lower bound pruning: if even the best case can't beat best_max
        num_left = 12 - len(current_factors)
        lb = lower_bound_max(remaining, min_possible, num_left, used_set)
        if lb >= best_max:
            return
        
        # Upper bound for this factor: must be < best_max and <= remaining
        hi = min(best_max - 1, remaining)
        
        if hi < min_possible:
            return
        
        # Get valid divisors
        divs = get_divisors_from_powers(remaining, min_possible, hi)
        
        # Filter out used
        divs = [d for d in divs if d not in used_set]
        
        if not divs:
            return
        
        # Sort: prefer smaller divisors to allow more splits and keep max low
        divs.sort()
        
        # Limit to speed up
        max_divs = 60
        divs = divs[:max_divs]
        
        for d in divs:
            new_max = max(max_so_far, d)
            if new_max >= best_max:
                continue
            
            # Additional pruning: check if remaining can be split
            new_remaining = remaining // d
            new_num_left = 12 - (len(current_factors) + 1)
            if new_num_left > 0:
                # Need new_num_left distinct factors >= d+1 with product new_remaining
                if new_remaining < (d + 1) ** new_num_left:
                    continue
                # Also check: can we fit new_num_left distinct factors >= d+1
                # whose product is new_remaining?
                # Simple check: new_remaining must be >= product of new_num_left smallest distinct ints >= d+1
                min_prod = 1
                for i in range(new_num_left):
                    min_prod *= (d + 1 + i)
                if new_remaining < min_prod:
                    continue
            
            new_used = used_set | {d}
            new_factors = current_factors + [d]
            new_min = d + 1
            
            search(new_remaining, new_min, new_factors, new_max, new_used)
            
            if time.time() - start_time > DEADLINE:
                return
    
    # Try to find a good initial solution to set a tight upper bound
    # Start with a reasonable guess and refine
    # For 12 factors, geometric mean is about 233
    # Let's try to find a solution with max around 300-400 first
    
    # Initial search with a loose bound
    search(target, 13, [], 0, set())
    
    return best
