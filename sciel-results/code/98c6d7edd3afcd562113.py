def construct():
    from math import factorial
    import time
    
    target = factorial(12)
    start_time = time.time()
    deadline = start_time + 85  # safety margin
    
    # All divisors of 12! that are > 12
    # 12! = 2^10 * 3^5 * 5^2 * 7 * 11
    # Generate all divisors
    primes = [2, 3, 5, 7, 11]
    exponents = [10, 5, 2, 1, 1]
    
    divisors = [1]
    for p, e in zip(primes, exponents):
        new_divs = []
        for d in divisors:
            val = d
            for _ in range(e + 1):
                new_divs.append(val)
                val *= p
        divisors = new_divs
    
    # Filter: > 12, and < target (since we need at least one more factor)
    valid_divs = sorted(set(d for d in divisors if 12 < d < target))
    
    best = [target]  # fallback: single factor = 12!
    best_max = target
    
    # Precompute factorization of 12! for quick divisor generation
    # 12! = 2^10 * 3^5 * 5^2 * 7 * 11
    
    def get_divisors_of(n):
        """Generate all divisors of n that are > 12 and <= n, sorted."""
        if n == 1:
            return []
        # Factor n using the known prime factors of 12!
        factors = {}
        temp = n
        for p in [2, 3, 5, 7, 11]:
            count = 0
            while temp % p == 0:
                temp //= p
                count += 1
            if count > 0:
                factors[p] = count
        # If temp > 1, it should not happen since n divides 12!
        if temp > 1:
            return []
        
        divs = [1]
        for p, e in factors.items():
            new_divs = []
            for d in divs:
                val = d
                for _ in range(e + 1):
                    new_divs.append(val)
                    val *= p
            divs = new_divs
        
        # Filter: > 12
        return sorted(d for d in divs if 12 < d <= n)
    
    def dfs(remaining, current_factors, next_min, depth):
        nonlocal best, best_max
        
        if time.time() > deadline:
            return
        
        if remaining == 1:
            if current_factors:
                mx = max(current_factors)
                if mx < best_max:
                    best_max = mx
                    best = list(current_factors)
            return
        
        # Simple bound: if remaining < next_min, we can't pick any factor
        if remaining < next_min:
            return
        
        # Get divisors of remaining that are >= next_min
        divs = get_divisors_of(remaining)
        
        # Filter to only those >= next_min and not in current_factors
        current_set = set(current_factors)
        
        for f in divs:
            if f < next_min:
                continue
            if f in current_set:
                continue
            
            # Pruning: remaining/f must be decomposable
            new_remaining = remaining // f
            
            # If new_remaining == 1, we're done (handled at top of next call)
            # If new_remaining > 1, we need more factors
            # The next factor must be > f (to keep them distinct and in order)
            next_next_min = max(f + 1, 13)
            if new_remaining > 1 and new_remaining < next_next_min:
                continue
            
            current_factors.append(f)
            dfs(new_remaining, current_factors, next_next_min, depth + 1)
            current_factors.pop()
            
            if time.time() > deadline:
                return
        
    dfs(target, [], 13, 0)
    
    return best
