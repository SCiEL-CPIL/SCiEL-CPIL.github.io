import math

def construct():
    # Find n <= 10^18 with maximum number of divisors
    # This is a heuristic improvement assuming more divisors leads to more coprime adjacent pairs
    LIMIT = 10**18
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113]
    
    best_n = 1
    best_divs = 1
    
    def search(idx, max_exp, current_n, current_divs):
        nonlocal best_n, best_divs
        if current_n > LIMIT:
            return
        
        if current_divs > best_divs:
            best_divs = current_divs
            best_n = current_n
            
        if idx >= len(primes):
            return
            
        p = primes[idx]
        # Try exponents from 1 to max_exp
        # We can also try exponent 0? No, if we skip a prime, we break the chain of non-increasing exponents?
        # Actually, we can skip primes. But for max divisors, we use smallest primes.
        # The standard algorithm assumes we use primes in order and exponents are non-increasing.
        # If we skip a prime, we are effectively setting its exponent to 0, which is <= max_exp.
        # But if we set exponent to 0, we should still be able to use later primes?
        # No, if we skip prime p_i, we can't use p_j (j>i) with exponent > 0 if we want to maintain the property that smaller primes have larger exponents?
        # Actually, the condition is just that the number is formed by primes.
        # The optimal number for max divisors always uses the first k primes with non-increasing exponents.
        
        val = current_n
        for e in range(1, max_exp + 1):
            val *= p
            if val > LIMIT:
                break
            search(idx + 1, e, val, current_divs * (e + 1))
            
    search(0, 60, 1, 1)
    return best_n
