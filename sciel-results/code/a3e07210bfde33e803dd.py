import math
import time
import numpy as np

def construct():
    """
    Returns an integer n <= 10^18 maximizing the number of adjacent
    coprime pairs in its divisor sequence.
    """
    start_time = time.time()
    TIME_LIMIT = 145  # Safety margin below 150s wall-clock kill
    
    # The optimal n is expected to use the smallest primes.
    # Product of first 15 primes: 2*3*5*...*47 ~ 6e17.
    # Product of first 16 primes: *53 ~ 3.4e19 > 10^18.
    # So k is at most 15.
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    max_val = 10**18
    
    best_n = 1
    best_score = -1
    
    def tau_perp_from_exponents(exponents, primes_subset):
        """
        Calculate the score tau_perp(n) given the prime factors and their exponents.
        """
        if not exponents:
            return 0
            
        divisors = [1]
        
        for i, p in enumerate(primes_subset):
            e = exponents[i]
            if e == 0:
                continue
            current_divs = list(divisors)
            factor = 1
            for _ in range(e):
                factor *= p
                for d in current_divs:
                    divisors.append(d * factor)
            
        if len(divisors) == 1:
            return 0
            
        divisors.sort()
        
        # Use numpy for vectorized GCD calculation
        # Convert to int64 to avoid overflow issues in intermediate steps if any, 
        # though divisors are <= 10^18 which fits in int64.
        arr = np.array(divisors, dtype=np.int64)
        gcds = np.gcd(arr[:-1], arr[1:])
        score = np.count_nonzero(gcds == 1)
        
        return int(score)

    def dfs(idx, max_exp, current_n, exponents_so_far, primes_used):
        nonlocal best_n, best_score
        
        if time.time() - start_time > TIME_LIMIT:
            return
            
        # If we have included at least one prime, it's a valid candidate
        # We evaluate all candidates found.
        if current_n > 1:
            # Calculate score using the known factorization
            score = tau_perp_from_exponents(exponents_so_far, primes_used)
            if score > best_score:
                best_score = score
                best_n = current_n
        
        if idx >= len(primes):
            return
            
        p = primes[idx]
        
        # Determine max exponent for this prime
        # current_n * p^e <= max_val
        # p^e <= max_val // current_n
        limit = max_val // current_n
        
        e_max_possible = 0
        temp_product = 1
        while temp_product * p <= limit and e_max_possible < max_exp:
            temp_product *= p
            e_max_possible += 1
        
        e_limit = min(e_max_possible, max_exp)
        
        # Optimization: Try e=1 first? Or high to low?
        # High exponents usually lead to fewer "coprime" neighbors because
        # divisors become more clustered around powers of p.
        # Squarefree (e=1) often allows more variety.
        # However, the current code tries high to low.
        # Let's try 1, 2, ... up to e_limit.
        # Why? Because if e=1 is good, we want to find it quickly to prune?
        # Actually, the upper bound is weak.
        # Let's stick to high-to-low as it might find "dense" structures faster?
        # No, let's try low-to-high (1, 2, ...) because squarefree is a strong candidate.
        # Wait, the current program scores 24.8.
        # Let's just swap the loop to iterate 1 to e_limit.
        
        for e in range(1, e_limit + 1):
            next_n = current_n * (p ** e)
            dfs(idx + 1, e, next_n, exponents_so_far + [e], primes_used + [p])
            
            if time.time() - start_time > TIME_LIMIT:
                return

    # Start DFS
    # Max exponent for 2 is log2(10^18) ~ 59
    # We pass max_exp = 60 to allow any reasonable start.
    dfs(0, 60, 1, [], [])
    
    return best_n

if __name__ == "__main__":
    # Optional: run a quick self-test
    n = construct()
    # print(f"Best n: {n}")
    # print(f"Score: {tau_perp(n)}")
    pass
