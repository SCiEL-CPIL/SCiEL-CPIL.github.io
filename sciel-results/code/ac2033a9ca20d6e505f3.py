"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
_start = time.time()
_TIMEOUT = 85  # safety margin before 90s kill

def construct():
    # Deadline guard
    if time.time() - _start > _TIMEOUT:
        return [1]
    
    n = 256
    
    # Sieve to find all primes up to 256
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n + 1, i):
                is_prime[j] = False
    primes = [p for p in range(2, n + 1) if is_prime[p]]
    
    from math import gcd
    
    # Strategy 1: Max Prime Powers
    set1 = [1]
    for p in primes:
        val = p
        while val * p <= n:
            val *= p
        set1.append(val)
    sum1 = sum(set1)
    
    # Strategy 2: Greedy Merge starting from prime powers
    current_set = list(set1)
    improved = True
    while improved:
        improved = False
        best_gain = 0
        best_pair = None
        
        s = sorted(current_set)
        for i in range(len(s)):
            if s[i] > 256:
                continue
            for j in range(i + 1, len(s)):
                prod = s[i] * s[j]
                if prod > 256:
                    break
                if prod in current_set:
                    continue
                # Check gcd
                if gcd(s[i], s[j]) != 1:
                    continue
                gain = prod - s[i] - s[j]
                if gain > best_gain:
                    best_gain = gain
                    best_pair = (s[i], s[j])
        
        if best_pair:
            a, b = best_pair
            current_set.remove(a)
            current_set.remove(b)
            current_set.append(a * b)
            improved = True
            
        if time.time() - _start > _TIMEOUT - 5:
            break
            
    sum2 = sum(current_set)
    
    # Strategy 3: Local search - try replacing pairs with better merges
    # Try un-merging and re-merging to escape local optima
    best_sum = max(sum1, sum2)
    best_result = set1 if sum1 >= sum2 else current_set
    
    # Additional local search: try swapping elements
    def try_improve(s):
        """Try to improve set s by local moves"""
        s = list(s)
        improved = True
        while improved:
            improved = False
            for i in range(len(s)):
                for j in range(i + 1, len(s)):
                    a, b = s[i], s[j]
                    if gcd(a, b) != 1:
                        continue
                    prod = a * b
                    if prod > 256:
                        continue
                    if prod in s:
                        continue
                    gain = prod - a - b
                    if gain > 0:
                        s.remove(a)
                        s.remove(b)
                        s.append(prod)
                        improved = True
                        break
                if improved:
                    break
        return s
    
    improved_set = try_improve(current_set)
    sum3 = sum(improved_set)
    if sum3 > best_sum:
        best_sum = sum3
        best_result = improved_set
    
    # Verify and return the better set
    def is_valid(s):
        if len(s) != len(set(s)):
            return False
        for i in range(len(s)):
            for j in range(i + 1, len(s)):
                if gcd(s[i], s[j]) != 1:
                    return False
        return True
    
    result = best_result
    
    if not is_valid(result):
        # Fallback to primes if something went wrong
        return [1] + primes

    return [int(x) for x in result]
# EVOLVE-BLOCK-END
