import math
import time

def construct():
    T12 = math.factorial(12)  # 479001600
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s limit

    # 12! = 2^10 * 3^5 * 5^2 * 7^1 * 11^1
    # Generate all divisors of 12! that are > 12
    primes = [2, 3, 5, 7, 11]
    exponents = [10, 5, 2, 1, 1]
    
    all_divisors = []
    
    def generate_divisors(idx, current_val):
        if idx == len(primes):
            if current_val > 12:
                all_divisors.append(current_val)
            return
        p = primes[idx]
        e = exponents[idx]
        val = 1
        for i in range(e + 1):
            generate_divisors(idx + 1, current_val * val)
            val *= p

    generate_divisors(0, 1)
    all_divisors.sort()
    
    # For each candidate M, check if we can find distinct divisors in [13, M] with product T12
    # Use backtracking with strong pruning
    
    def search_for_M(M):
        """Find distinct divisors of T12, all in [13, M], with product exactly T12."""
        valid = [d for d in all_divisors if d <= M]
        n = len(valid)
        
        if n == 0:
            return None
        
        # Quick check: product of all valid must be >= T12 (weak but fast)
        # Actually we need a subset, so this isn't directly useful.
        
        def dfs(remaining, min_idx, chosen):
            if time.time() > deadline:
                return None
            
            if remaining == 1:
                return list(chosen)
            
            if min_idx >= n:
                return None
            
            # Pruning: smallest available factor
            if remaining < valid[min_idx]:
                return None
            
            # Pruning: if remaining is prime and not in valid[min_idx:], fail
            # Check if remaining can be formed by factors >= valid[min_idx]
            # The minimum product we can still achieve with at least one more factor is valid[min_idx]
            # But we need remaining to be divisible by some d in valid[min_idx:]
            
            for i in range(min_idx, n):
                d = valid[i]
                if d > remaining:
                    break
                if remaining % d == 0:
                    chosen.append(d)
                    res = dfs(remaining // d, i + 1, chosen)
                    if res is not None:
                        return res
                    chosen.pop()
            
            return None
        
        return dfs(T12, 0, [])
    
    # Try M from 13 upward
    for M in range(13, 500):
        if time.time() > deadline:
            break
        sol = search_for_M(M)
        if sol is not None:
            return sol
    
    # Fallback: return single factor (valid but suboptimal)
    return [T12]
