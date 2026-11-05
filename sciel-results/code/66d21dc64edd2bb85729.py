# EVOLVE-BLOCK-START
import time
from math import factorial, log

def construct():
    start_time = time.time()
    DEADLINE = 50.0  # seconds, safety margin before 90s kill
    
    target = factorial(12)  # 479001600
    # Prime factorization of 12!: 2^10 * 3^5 * 5^2 * 7^1 * 11^1
    
    def factorize(n):
        """Return dict of prime -> exponent for n."""
        factors = {}
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors[d] = factors.get(d, 0) + 1
                n //= d
            d += 1
        if n > 1:
            factors[n] = factors.get(n, 0) + 1
        return factors
    
    target_facs = factorize(target)
    
    def divisors_from_facs(facs):
        """Generate all divisors from prime factorization dict."""
        primes = sorted(facs.keys())
        divs = [1]
        for p in primes:
            e = facs[p]
            new_divs = []
            for d in divs:
                power = 1
                for _ in range(e + 1):
                    new_divs.append(d * power)
                    power *= p
            divs = new_divs
        return sorted(divs)
    
    def can_decompose(max_factor):
        """Check if target can be decomposed into distinct factors in (12, max_factor]."""
        # Generate all valid divisors of target that are > 12 and <= max_factor
        all_divs = divisors_from_facs(target_facs)
        valid_divs = [d for d in all_divs if 12 < d <= max_factor]
        
        # Sort descending to find larger factors first (pruning)
        valid_divs.sort(reverse=True)
        
        # Memoization: key = (remaining_target, last_used_index)
        memo = {}
        
        def dfs(rem, min_idx, used_factors, depth):
            if time.time() - start_time > DEADLINE - 2:
                return None
            if rem == 1:
                return used_factors
            if depth >= 12:
                return None
            # Pruning: if remaining product > 1, need at least one more factor
            # All remaining factors must be > 12, so rem must be expressible
            # as product of distinct factors each > 12 and <= max_factor
            # Lower bound: need factors whose product is rem, each > 12
            # So rem must be at least 13^(number of remaining factors needed)
            # But more simply: the smallest valid factor is 13, so rem >= 13 if we need >= 1 factor
            if rem <= 12:
                return None
            
            # Check memo
            key = (rem, min_idx)
            if key in memo:
                return memo[key]
            
            # Try each valid divisor of rem that is >= valid_divs[min_idx] (to ensure distinct & ordered)
            # Actually we need distinct factors, so track used set differently
            # Let me use a different approach: track the set of used factors
            
            for i in range(min_idx, len(valid_divs)):
                d = valid_divs[i]
                if d > rem:
                    continue
                if rem % d == 0:
                    res = dfs(rem // d, i + 1, used_factors + [d], depth + 1)
                    if res is not None:
                        memo[key] = res
                        return res
            
            memo[key] = None
            return None
        
        result = dfs(target, 0, [], 0)
        return result
    
    # Binary search on maximum factor M
    # Lower bound: smallest possible max factor
    # Upper bound: 12! (single factor)
    lo = 13
    hi = target
    best_result = None
    
    while lo <= hi:
        if time.time() - start_time > DEADLINE - 5:
            break
        mid = (lo + hi) // 2
        result = can_decompose(mid)
        if result is not None:
            best_result = result
            hi = mid - 1
        else:
            lo = mid + 1
    
    if best_result is None:
        # Fallback: single factor
        return [target]
    
    return best_result
# EVOLVE-BLOCK-END
