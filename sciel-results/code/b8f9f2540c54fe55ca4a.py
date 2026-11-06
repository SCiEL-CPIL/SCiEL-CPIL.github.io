"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def construct():
    target = math.factorial(12)
    deadline = time.time() + 80.0
    
    # Generate all divisors of target
    divisors = set()
    for i in range(1, int(math.isqrt(target)) + 1):
        if target % i == 0:
            divisors.add(i)
            divisors.add(target // i)
            
    # Filter divisors > 12 and sort
    candidates = sorted([d for d in divisors if d > 12])
    cand_set = set(candidates)
    
    # Precompute prime factors of target for pruning
    def prime_factors(n):
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
    
    target_pf = prime_factors(target)
    
    # Precompute for each candidate its prime factorization
    cand_pf = {}
    for c in candidates:
        cand_pf[c] = prime_factors(c)
    
    best_factors = [target]  # Fallback
    
    def exists_subset(remaining, start_idx, k_idx_limit, used_pf):
        """DFS to check if remaining can be formed by distinct candidates from start_idx to k_idx_limit.
        used_pf tracks the prime factor exponents used so far."""
        if remaining == 1:
            return True
        
        if start_idx > k_idx_limit:
            return False
        
        # Pruning: if remaining is prime, it must be exactly one of the available candidates
        if start_idx == k_idx_limit:
            # Only one candidate left
            f = candidates[start_idx]
            if remaining == f:
                return True
            return False
        
        # Pruning: if remaining < candidates[start_idx], can't form it
        if remaining < candidates[start_idx]:
            return False
        
        # Pruning: check if remaining has any prime factor not available in candidates[start_idx..k_idx_limit]
        # This is expensive to check fully, so skip for now or do a quick check
        
        # Pruning: if remaining is prime, check if it's in the available range
        # Quick primality test for small numbers
        if remaining > 1:
            # Check if remaining is prime
            is_prime = True
            if remaining < 2:
                is_prime = False
            else:
                for p in range(2, int(math.isqrt(remaining)) + 1):
                    if remaining % p == 0:
                        is_prime = False
                        break
            if is_prime:
                # remaining must be one of the candidates in [start_idx, k_idx_limit]
                # Binary search or linear scan
                # Since candidates is sorted, we can check if remaining is in the slice
                # But we need to check if it's actually in candidates[start_idx..k_idx_limit]
                # Quick check: if remaining > candidates[k_idx_limit], no
                if remaining > candidates[k_idx_limit]:
                    return False
                # Check if remaining is in cand_set and its index is in range
                if remaining in cand_set:
                    # Find its index - but this is O(n). Instead, we can just try to pick it
                    # Actually, for the existence check, if remaining is prime and in the range,
                    # we can pick it and return True (since remaining/f = 1)
                    # But we need to ensure it's at an index >= start_idx and <= k_idx_limit
                    # Let's do a binary search
                    lo, hi = start_idx, k_idx_limit
                    found = False
                    while lo <= hi:
                        mid = (lo + hi) // 2
                        if candidates[mid] == remaining:
                            found = True
                            break
                        elif candidates[mid] < remaining:
                            lo = mid + 1
                        else:
                            hi = mid - 1
                    if found:
                        return True
                return False
        
        for i in range(start_idx, k_idx_limit + 1):
            f = candidates[i]
            if f > remaining:
                break
            
            if remaining % f == 0:
                if exists_subset(remaining // f, i + 1, k_idx_limit, None):
                    return True
        
        return False
    
    def find_subset(remaining, start_idx, k_idx_limit):
        if remaining == 1:
            return []
        
        if start_idx > k_idx_limit:
            return None
        
        # Pruning
        if remaining < candidates[start_idx]:
            return None
        
        for i in range(start_idx, k_idx_limit + 1):
            f = candidates[i]
            if f > remaining:
                break
            
            if remaining % f == 0:
                res = find_subset(remaining // f, i + 1, k_idx_limit)
                if res is not None:
                    return [f] + res
        return None
    
    # Iterate K in increasing order
    for k_idx in range(len(candidates)):
        if time.time() > deadline:
            break
        
        K = candidates[k_idx]
        
        # Quick check: can target be formed with factors <= K?
        # The product of the largest possible distinct factors from candidates[0..k_idx] 
        # must be >= target. But this is a weak bound.
        
        if exists_subset(target, 0, k_idx, None):
            sol = find_subset(target, 0, k_idx)
            if sol:
                return sol
    
    return best_factors
# EVOLVE-BLOCK-END
