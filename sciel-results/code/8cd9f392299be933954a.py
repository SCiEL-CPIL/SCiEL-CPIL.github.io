"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import factorial

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin
    
    N = factorial(12)  # 479001600
    # 12! = 2^10 * 3^5 * 5^2 * 7^1 * 11^1
    
    # Generate all divisors of N
    primes = [(2, 10), (3, 5), (5, 2), (7, 1), (11, 1)]
    
    def gen_divisors(idx, current):
        if idx == len(primes):
            yield current
            return
        p, exp = primes[idx]
        val = 1
        for e in range(exp + 1):
            yield from gen_divisors(idx + 1, current * val)
            val *= p
    
    all_divs = sorted(set(gen_divisors(0, 1)))
    # Filter > 12
    divs_gt12 = [d for d in all_divs if d > 12]
    
    best_result = None
    best_max = N + 1
    
    # Try candidate max values in increasing order
    for M in divs_gt12:
        if time.time() > deadline:
            break
        if M >= best_max:
            break
        
        # Available divisors: those <= M and > 12
        avail = [d for d in divs_gt12 if d <= M]
        
        if not avail:
            continue
        
        # DFS with memoization-like pruning
        # State: (remaining_product, start_index)
        # We want to find a subset of distinct elements from avail[start_index:] 
        # whose product equals remaining_product
        
        def dfs(remaining, start_idx, count):
            if time.time() > deadline:
                return None
            
            if remaining == 1:
                if 1 <= count <= 12:
                    return []
                return None
            
            if count >= 12:
                return None
            
            # Pruning: if remaining is not divisible by any available divisor, fail
            # Also: if remaining < avail[start_idx] (smallest available), and remaining != 1, fail
            if start_idx >= len(avail):
                return None
            
            if remaining < avail[start_idx]:
                return None
            
            # Try each divisor from start_idx onward
            for i in range(start_idx, len(avail)):
                d = avail[i]
                if d > remaining:
                    break
                if remaining % d == 0:
                    sub = dfs(remaining // d, i + 1, count + 1)
                    if sub is not None:
                        return [d] + sub
            
            return None
        
        # We need to track the actual factors, not just check existence
        # Let me redo this to collect factors
        
        def dfs_collect(remaining, start_idx, factors):
            if time.time() > deadline:
                return None
            
            if remaining == 1:
                if 1 <= len(factors) <= 12:
                    return list(factors)
                return None
            
            if len(factors) >= 12:
                return None
            
            if start_idx >= len(avail):
                return None
            
            if remaining < avail[start_idx]:
                return None
            
            for i in range(start_idx, len(avail)):
                d = avail[i]
                if d > remaining:
                    break
                if remaining % d == 0:
                    factors.append(d)
                    result = dfs_collect(remaining // d, i + 1, factors)
                    factors.pop()
                    if result is not None:
                        return result
            
            return None
        
        result = dfs_collect(N, 0, [])
        if result is not None:
            # Verify
            prod = 1
            for x in result:
                prod *= x
            if prod == N and len(set(result)) == len(result) and all(x > 12 for x in result) and len(result) <= 12:
                max_factor = max(result)
                if max_factor < best_max:
                    best_max = max_factor
                    best_result = result
                break  # M is increasing, first valid M gives optimal
    
    if best_result is not None:
        return best_result
    else:
        return [N]
# EVOLVE-BLOCK-END
