"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt

P = {'n': 256}

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    n = P['n']
    
    # Generate primes up to n
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, isqrt(n) + 1):
        if is_prime[i]:
            for j in range(i*i, n + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, n + 1) if is_prime[i]]
    
    # Function to get prime factorization
    def factorize(x):
        factors = {}
        d = 2
        while d * d <= x:
            while x % d == 0:
                factors[d] = factors.get(d, 0) + 1
                x //= d
            d += 1
        if x > 1:
            factors[x] = factors.get(x, 0) + 1
        return factors
    
    # Generate candidate composites: square-free, >= 3 distinct prime factors, <= n
    # We only consider composites composed of primes <= 16 to allow k_p >= 2
    # Primes > 16 have p^2 > 256, so k_p must be 1, meaning v_p(z) must be 0.
    # So z cannot have any prime factor > 16.
    small_primes = [p for p in primes if p <= 16]
    
    candidates = []
    # Generate all square-free numbers <= n composed of small_primes
    # Max 6 primes. 2*3*5*7*11*13 = 30030 > 256.
    # So max 5 primes? 2*3*5*7*11 = 2310 > 256.
    # Max 4 primes: 2*3*5*7 = 210.
    
    def gen_combinations(prs, start, current_val, current_factors):
        if current_val > n:
            return
        if len(current_factors) >= 3:
            candidates.append(current_val)
        
        for i in range(start, len(prs)):
            p = prs[i]
            if current_val * p > n:
                break
            gen_combinations(prs, i + 1, current_val * p, current_factors + [p])
            
    gen_combinations(small_primes, 0, 1, [])
    
    # Filter candidates to ensure they are distinct and valid
    candidates = list(set(candidates))
    
    # Determine required k_p for each prime
    # k_p = 1 + max(v_p(z) for z in candidates if p|z)
    # Since candidates are square-free, v_p(z) is 0 or 1.
    # So k_p is 1 or 2.
    
    required_k = {p: 1 for p in primes}
    for z in candidates:
        facs = factorize(z)
        for p in facs:
            # v_p(z) = 1
            if required_k.get(p, 1) < 2:
                required_k[p] = 2
                
    # Check if p^k <= n
    valid_primes = []
    prime_powers = []
    for p in primes:
        k = required_k[p]
        val = p ** k
        if val <= n:
            valid_primes.append(p)
            prime_powers.append(val)
        else:
            # If p^k > n, we cannot satisfy the condition for any z containing p with v_p(z)=1
            # So we must remove all z containing p from candidates
            # This is a dependency.
            # For now, if this happens, we fall back to just primes.
            # But for p <= 16, p^2 <= 256 is true.
            # For p > 16, k_p is 1, so p^1 <= 256 is true.
            pass
            
    # Construct set
    S = set(prime_powers)
    
    # Add candidates
    # We need to verify that the chosen prime powers actually satisfy the condition for these candidates.
    # Condition: for all z in candidates, for all p|z, k_p > v_p(z).
    # Since k_p is set to 2 if p|z, and v_p(z)=1, this holds.
    
    # However, we must also check the "two composites" condition.
    # Let's just add them and verify the whole set.
    
    S.update(candidates)
    
    # Verify the set
    def is_valid(S):
        S_list = list(S)
        n_elem = len(S_list)
        for i in range(n_elem):
            a = S_list[i]
            for j in range(i + 1, n_elem):
                b = S_list[j]
                for k in range(j + 1, n_elem):
                    c = S_list[k]
                    if (b * c) % a == 0:
                        return False
        return True
        
    if is_valid(S):
        return list(S)
    else:
        # Fallback: just primes
        return primes

# EVOLVE-BLOCK-END
