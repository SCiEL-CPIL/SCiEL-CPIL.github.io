import time
from math import factorial

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s limit to be safe
    
    N = factorial(12)
    
    # Generate divisors of N greater than 12
    divisors = []
    i = 1
    while i * i <= N:
        if N % i == 0:
            d1 = i
            d2 = N // i
            if d1 > 12:
                divisors.append(d1)
            if d2 > 12 and d2 != d1:
                divisors.append(d2)
        i += 1
    
    divisors = sorted(set(divisors))
    
    best_max = N  # Initial upper bound
    best_factors = [N]
    
    # Helper to check if rem can be formed by at most k factors > last
    # This is a heuristic prune. 
    # Minimum product of m distinct integers > last is (last+1)(last+2)...(last+m)
    def min_product_gt(last, count):
        prod = 1
        for j in range(count):
            prod *= (last + 1 + j)
            if prod > N: # Overflow or just too big
                return prod
        return prod

    def dfs(rem, last, factors, depth):
        nonlocal best_max, best_factors
        
        if time.time() > deadline:
            return
            
        if rem == 1:
            if factors:
                m = max(factors)
                if m < best_max:
                    best_max = m
                    best_factors = factors[:]
            return
            
        if depth >= 12:
            return
            
        # Prune: if current max is already >= best_max, no point continuing
        # Note: max(factors) is last if we append in increasing order? 
        # Yes, we enforce d > last, so factors is sorted.
        if factors and factors[-1] >= best_max:
            return
            
        # Try to add a factor d
        # d must divide rem
        # d > last
        # d < best_max (strictly, to improve)
        # If d == rem, we can take it and finish
        # If d < rem, we need more factors
        
        # Optimization: iterate through divisors of rem that are > last and < best_max
        # Instead of iterating all divisors list, we can just check candidates from divisors list
        # Since divisors list is small (~700), this is fine.
        
        for d in divisors:
            if d <= last:
                continue
            if d >= best_max:
                break # Since divisors is sorted
            
            if rem % d != 0:
                continue
                
            # If we take d, new rem is rem // d
            new_rem = rem // d
            
            # Check if new_rem can be formed by remaining slots
            remaining_slots = 12 - depth - 1
            
            if new_rem == 1:
                # Valid solution found
                factors.append(d)
                m = max(factors) # which is d since sorted
                if m < best_max:
                    best_max = m
                    best_factors = factors[:]
                factors.pop()
                continue # No need to recurse deeper if rem becomes 1
            
            # If new_rem > 1, we need more factors
            # Check if min product of remaining_slots factors > d is <= new_rem
            # Actually, we need to form new_rem with factors > d.
            # The minimum product of k distinct factors > d is (d+1)(d+2)...(d+k)
            # If remaining_slots == 0, we can't form new_rem > 1.
            if remaining_slots == 0:
                continue
                
            # Heuristic prune:
            # If new_rem is large, but we have few slots, check if it's possible.
            # The max product we can make with remaining_slots factors < best_max is 
            # roughly (best_max-1)^remaining_slots. If new_rem > that, impossible.
            # But best_max is changing.
            
            # Simple check: if new_rem > 1 and remaining_slots > 0
            # We must be able to divide new_rem by factors > d.
            # So new_rem must have a divisor > d.
            # If new_rem is prime and new_rem <= d, impossible.
            # If new_rem is prime and new_rem > d, we can take new_rem as the next factor
            # (if it is < best_max).
            
            factors.append(d)
            dfs(new_rem, d, factors, depth + 1)
            factors.pop()

    dfs(N, 12, [], 0)
    
    return best_factors
