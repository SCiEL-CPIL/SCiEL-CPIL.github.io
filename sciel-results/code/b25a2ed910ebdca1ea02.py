"""Construct an integer n <= 10^18 maximizing the number of adjacent coprime divisor pairs."""
# EVOLVE-BLOCK-START
def construct():
    limit = 10**18
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    
    best_n = 1
    best_tau = 0
    
    def dfs(idx, current_n, current_tau, max_exp):
        nonlocal best_n, best_tau
        
        # Update best if we found a candidate with more divisors
        # Note: We assume higher tau correlates with higher score.
        if current_tau > best_tau:
            best_tau = current_tau
            best_n = current_n
            
        if idx >= len(primes):
            return
            
        p = primes[idx]
        
        # Try exponents from max_exp down to 1
        # We can stop if p ** e exceeds limit / current_n
        for e in range(max_exp, 0, -1):
            # Optimization: check if p**e fits
            # p**e can be large, but e is small.
            # Better: iterative multiplication
            power = p
            for exp in range(1, e + 1):
                if exp == e:
                    new_n = current_n * power
                    if new_n <= limit:
                        dfs(idx + 1, new_n, current_tau * (e + 1), e)
                    break
                power *= p
                if current_n * power > limit:
                    break
                    
    dfs(0, 1, 1, 60)
    
    return best_n
# EVOLVE-BLOCK-END
