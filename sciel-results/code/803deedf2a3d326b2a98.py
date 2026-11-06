"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
from functools import lru_cache

T12 = math.factorial(12)  # 479001600

# Prime factorization of 12!
# 12! = 2^10 * 3^5 * 5^2 * 7 * 11
PRIMES = [2, 3, 5, 7, 11]
INITIAL_EXP = (10, 5, 2, 1, 1)

def exp_to_val(exp):
    """Convert exponent tuple to integer value."""
    result = 1
    for i, e in enumerate(exp):
        if e > 0:
            result *= PRIMES[i] ** e
    return result

def val_to_exp(n):
    """Convert integer to exponent tuple for the known primes. Returns None if n has other prime factors."""
    exp = [0, 0, 0, 0, 0]
    for i, p in enumerate(PRIMES):
        while n % p == 0:
            exp[i] += 1
            n //= p
    if n != 1:
        return None
    return tuple(exp)

def solve():
    """
    Find distinct factors > 12 multiplying to 12! minimizing the largest factor.
    Uses DFS over exponent vectors.
    """
    deadline = time.time() + 85
    
    best_max = T12
    best_solution = [T12]
    
    # We will build factors in strictly increasing order.
    # State: (remaining_exp_tuple, last_factor_val)
    # We need to find a sequence of factors f1 < f2 < ... < fk such that:
    # 1. fi >= 13
    # 2. Product fi = 12!
    # 3. Minimize fk (the largest)
    
    # To minimize the largest, we can try to bound the largest factor.
    # Let M be the max factor. We want smallest M.
    # We can iterate M from a lower bound upwards?
    # Lower bound: (12!)^(1/2) is too high. 
    # Actually, if we use k factors, M >= (12!)^(1/k).
    # But distinctness and > 12 constraint.
    
    # Alternative: DFS that tracks the current max.
    # We prune if current_max >= best_max.
    # We also prune if the remaining product cannot be factored into distinct integers > last_factor with max < best_max.
    
    # Let's use a recursive function.
    # To make it fast, we represent the remaining product by its exponent vector.
    # We iterate over possible next factors f.
    # f must be a divisor of remaining product.
    # f > last_factor.
    # f >= 13.
    # f < best_max (if we want to improve).
    
    # How to enumerate divisors of remaining product efficiently?
    # The remaining product is determined by exp vector.
    # Number of divisors can be large but manageable for 12!.
    # Max divisors of 12! is (10+1)*(5+1)*(2+1)*(1+1)*(1+1) = 11*6*3*2*2 = 792.
    # So we can precompute or generate divisors on the fly.
    
    # However, we must ensure f > last_factor.
    # And we must ensure that the remaining product after dividing by f can be factored into distinct integers > f.
    
    # Let's implement a helper to generate all divisors of a given exp vector.
    # Then filter by value > last_factor and value < best_max.
    
    # For efficiency, we can sort divisors and iterate.
    
    # Let's use a global list to track the current path.
    
    def generate_divisors(exp):
        """Generate all divisors (as values) of the number represented by exp."""
        divs = [1]
        for i, e in enumerate(exp):
            if e > 0:
                p = PRIMES[i]
                powers = [1]
                for k in range(1, e + 1):
                    powers.append(powers[-1] * p)
                new_divs = []
                for d in divs:
                    for pow_val in powers:
                        new_divs.append(d * pow_val)
                divs = new_divs
        return divs

    # To avoid regenerating divisors at every step (which is expensive),
    # we can precompute divisors for all possible exp states?
    # Number of states is 792.
    # Let's cache divisors for each exp tuple.
    
    divisor_cache = {}
    
    def get_divisors(exp):
        if exp not in divisor_cache:
            divisor_cache[exp] = generate_divisors(exp)
        return divisor_cache[exp]

    # DFS
    def dfs(rem_exp, last_f, chosen, current_max):
        nonlocal best_max, best_solution
        
        if time.time() > deadline:
            return
            
        if rem_exp == (0, 0, 0, 0, 0):
            # All factors chosen
            if chosen:
                if current_max < best_max:
                    best_max = current_max
                    best_solution = list(chosen)
            return
        
        # Pruning:
        # If we have no more factors to choose, rem_exp must be 0.
        # If rem_exp != 0, we need to choose at least one more factor.
        
        # The next factor f must satisfy:
        # 1. f > last_f
        # 2. f >= 13
        # 3. f < best_max (to improve)
        # 4. f divides rem_val (represented by rem_exp)
        
        # Also, the remaining product after choosing f must be factorable into distinct integers > f.
        # This implies that the remaining product must be >= 1.
        # If remaining product is 1, we are done (handled above).
        # If remaining product > 1, we need to factor it.
        
        # A strong pruning:
        # The minimum possible value for the next factor is max(last_f + 1, 13).
        # The maximum possible value for the next factor is min(best_max - 1, rem_val).
        
        # Let's get the divisors of rem_val.
        rem_val = exp_to_val(rem_exp)
        
        # Optimization: If rem_val is prime (or has no divisors in range), we can only take rem_val itself.
        # But we need to check if rem_val > last_f and rem_val >= 13 and rem_val < best_max.
        
        divs = get_divisors(rem_exp)
        
        # Sort divisors to process smaller ones first? 
        # Or larger ones? 
        # If we process smaller ones first, we might find a solution with many small factors, 
        # which might allow a smaller max later? 
        # Actually, we want to minimize the MAX. 
        # Choosing a small factor now doesn't necessarily minimize the max.
        # But choosing a large factor now increases the current max.
        # So it's generally better to choose smaller factors first to keep current_max low, 
        # allowing more room for subsequent factors to be large (but < best_max).
        # Wait, if we choose a small factor, the remaining product is larger, 
        # which might force a larger factor later.
        # If we choose a large factor, the remaining product is smaller.
        # It's complex. Let's just iterate all valid divisors.
        
        # Filter divisors
        valid_divs = []
        for d in divs:
            if d > last_f and d >= 13 and d < best_max:
                # Also, we need to ensure that the remaining part can be factored.
                # The remaining part is rem_val // d.
                # It must be factorable into distinct integers > d.
                # This implies rem_val // d >= 1.
                # If rem_val // d == 1, it's a valid completion.
                # If rem_val // d > 1, we need to recurse.
                
                # Additional pruning:
                # The remaining product R = rem_val // d.
                # The next factor must be > d.
                # So R must be >= 1.
                # If R > 1, the smallest possible next factor is d+1.
                # So we need R >= d+1? Not necessarily, R could be 1.
                # If R > 1, and we need to factor R into distinct integers > d,
                # then R must be at least d+1 (if only one factor) or more.
                # Actually, if R > 1, the smallest factor is > d, so R must be > d.
                # Wait, if R is factored into k factors f1...fk with fi > d,
                # then R >= (d+1)^k.
                # This is a strong constraint.
                
                # For now, just recurse.
                valid_divs.append(d)
        
        # Sort valid_divs. 
        # Should we try smaller first or larger first?
        # If we try smaller first, we keep current_max low for longer.
        # This might help prune branches where current_max grows too fast.
        valid_divs.sort()
        
        for d in valid_divs:
            # Calculate new rem_exp
            d_exp = val_to_exp(d)
            if d_exp is None:
                continue # Should not happen if d is divisor of rem_val composed of known primes
            
            new_rem_exp = tuple(rem_exp[i] - d_exp[i] for i in range(5))
            if any(x < 0 for x in new_rem_exp):
                continue # Should not happen if d divides rem_val
            
            new_max = max(current_max, d)
            if new_max >= best_max:
                continue # Prune
            
            # Recurse
            dfs(new_rem_exp, d, chosen + [d], new_max)
            
            if time.time() > deadline:
                return

    # Initial call
    # rem_exp = INITIAL_EXP
    # last_f = 0 (so any f >= 13 is allowed)
    # chosen = []
    # current_max = 0
    
    dfs(INITIAL_EXP, 0, [], 0)
    
    return best_solution

def construct():
    """
    Return 1..12 DISTINCT integers all greater than 12, with product exactly 12!.
    Minimize the LARGEST factor.
    """
    try:
        solution = solve()
        # Verify
        product = 1
        for f in solution:
            product *= f
        if product == T12 and all(f > 12 for f in solution) and len(set(solution)) == len(solution) and 1 <= len(solution) <= 12:
            return solution
    except Exception:
        pass
    
    # Fallback
    return [T12]

# EVOLVE-BLOCK-END
