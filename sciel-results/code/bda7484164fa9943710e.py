"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
_start = time.time()
_TIMEOUT = 85  # safety margin before 90s kill

def construct():
    # Deadline guard
    if time.time() - _start > _TIMEOUT:
        return [1]
    
    # Sieve to find all primes up to 256
    n = 256
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n + 1, i):
                is_prime[j] = False
    primes = [p for p in range(2, n + 1) if is_prime[p]]
    
    # Initialize set with primes and 1
    current_set = list(primes) + [1]
    current_sum = sum(current_set)
    
    best_set = list(current_set)
    best_sum = current_sum
    
    # Greedy merge loop
    # We want to maximize sum by merging coprime pairs a,b into a*b
    # if a*b <= 256 and a*b not in set.
    # Gain is a*b - a - b.
    
    improved = True
    while improved:
        improved = False
        best_gain = 0
        best_pair = None
        
        # Sort to allow early termination in inner loop
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
                gain = prod - s[i] - s[j]
                if gain > best_gain:
                    best_gain = gain
                    best_pair = (s[i], s[j])
        
        if best_pair:
            a, b = best_pair
            # Remove a and b
            current_set.remove(a)
            current_set.remove(b)
            current_set.append(a * b)
            current_sum += best_gain
            
            if current_sum > best_sum:
                best_sum = current_sum
                best_set = list(current_set)
            
            improved = True
            
        # Time check
        if time.time() - _start > _TIMEOUT - 5:
            break
            
    # Verify pairwise coprime (sanity check)
    from math import gcd
    for i in range(len(best_set)):
        for j in range(i + 1, len(best_set)):
            if gcd(best_set[i], best_set[j]) != 1:
                # Fallback to primes if something went wrong
                return [1] + primes
                
    # Ensure types are Python int
    return [int(x) for x in best_set]
# EVOLVE-BLOCK-END
