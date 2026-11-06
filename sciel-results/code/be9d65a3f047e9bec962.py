"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
from itertools import product

def construct():
    target = math.factorial(12)
    start_time = time.time()
    timeout = 80.0  # Safety margin before 90s kill
    
    # Prime factorization of 12! = 2^10 * 3^5 * 5^2 * 7 * 11
    # Split into two groups: Group A = {2^10}, Group B = {3^5 * 5^2 * 7 * 11}
    # Actually, better split: Group A = {2^10 * 3^5}, Group B = {5^2 * 7 * 11}
    # Let's try: Group A = {2^10}, Group B = {3^5 * 5^2 * 7 * 11}
    
    # Generate all divisors from each group
    def gen_divisors_from_prime_powers(prime_powers):
        """Generate all divisors from a list of (prime, exponent) pairs."""
        divisors = [1]
        for p, e in prime_powers:
            new_divs = []
            for d in divisors:
                for exp in range(e + 1):
                    new_divs.append(d * (p ** exp))
            divisors = new_divs
        return divisors
    
    # Split primes: A = {2^10}, B = {3^5, 5^2, 7^1, 11^1}
    group_a_divs = gen_divisors_from_prime_powers([(2, 10)])
    group_b_divs = gen_divisors_from_prime_powers([(3, 5), (5, 2), (7, 1), (11, 1)])
    
    # Filter: factors must be > 12
    valid_a = [d for d in group_a_divs if d > 12]
    valid_b = [d for d in group_b_divs if d > 12]
    
    # For meet-in-the-middle, we need to partition 12! into factors > 12
    # Each factor can have contributions from both A and B parts
    # A factor f = a * b where a | 2^10, b | 3^5*5^2*7*11
    # We need to find distinct factors f1, f2, ..., fk such that prod(fi) = 12!
    
    # Alternative approach: enumerate all valid composite factors > 12 of 12!
    # and use DFS with strong pruning
    
    # Get all divisors of 12!
    def get_all_divisors(n):
        divs = set()
        for i in range(1, int(math.isqrt(n)) + 1):
            if n % i == 0:
                divs.add(i)
                divs.add(n // i)
        return sorted(divs)
    
    all_divs = get_all_divisors(target)
    valid_divs = [d for d in all_divs if d > 12]
    
    # Sort by value for pruning
    valid_divs.sort()
    
    best_factors = None
    best_max = float('inf')
    
    def dfs(remaining, start_idx, current_factors, current_max):
        nonlocal best_factors, best_max, best_max
        
        if time.time() - start_time > timeout:
            return False
        
        if remaining == 1:
            if current_factors and current_max < best_max:
                best_max = current_max
                best_factors = current_factors[:]
            return True
        
        # Lower bound: we need at least one more factor
        # The smallest possible next factor is valid_divs[start_idx] if start_idx < len(valid_divs)
        if start_idx >= len(valid_divs):
            return False
        
        # Pruning: if remaining is prime and > 12, it must be the last factor
        # Check if remaining is in valid_divs and > all current factors
        if remaining in valid_divs and remaining > (current_factors[-1] if current_factors else 12):
            new_max = max(current_max, remaining)
            if new_max < best_max:
                best_max = new_max
                best_factors = current_factors + [remaining]
            return True
        
        for i in range(start_idx, len(valid_divs)):
            d = valid_divs[i]
            
            # Time check
            if time.time() - start_time > timeout:
                return False
            
            if d > remaining:
                break
            
            if remaining % d != 0:
                continue
            
            # Distinctness: d must not be in current_factors
            if d in current_factors:
                continue
            
            new_remaining = remaining // d
            new_max = max(current_max, d)
            
            if new_max >= best_max:
                break
            
            # Recursive call
            dfs(new_remaining, i + 1, current_factors + [d], new_max)
        
        return True
    
    # Try different numbers of factors
    # Start with smaller k to find good solutions quickly
    for k in range(2, 13):
        if time.time() - start_time > timeout:
            break
        
        # We can't easily constrain k in the above DFS, so let it find any valid solution
        # and track the best_max
        dfs(target, 0, [], 0)
        
        # If we found a very good solution, we can stop
        if best_max <= 20:  # Theoretical minimum is around 13-15
            break
    
    if best_factors is None:
        return [target]
    
    return best_factors
# EVOLVE-BLOCK-END
