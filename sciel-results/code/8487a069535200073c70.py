import math
import time
import numpy as np
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit

    # Strategy: Constructive search
    # We look for collisions in M = k * sigma(k) where k is restricted to specific prime factorizations.
    # This avoids the O(N log N) sieve and allows exploring higher k values (up to 10^9 or more)
    # by iterating over exponents of small primes.

    # We will search for k of the form 2^a * 3^b * 5^c * 7^d ...
    # and compute M = k * sigma(k).
    # We store M in a dict: M -> list of k.
    # We look for M with the most k's.

    # To ensure k <= sqrt(M), we check this condition at the end.
    # Note: For k > 1, sigma(k) >= k + 1 > k, so M = k*sigma(k) > k^2.
    # Thus k < sqrt(M) is always true for k > 1.
    # For k=1, M=1, sqrt(M)=1, k<=sqrt(M) holds.
    # So we just need to ensure M <= 10^18.

    best_M = None
    best_k_list = []
    max_count = 0

    # Define primes to include in the construction
    # Including more primes increases the number of possible k values
    # but also increases the risk of M > 10^18.
    # Let's start with 2, 3, 5, 7.
    
    # We'll iterate over exponents a, b, c, d
    # k = 2^a 3^b 5^c 7^d
    # sigma(k) = (2^(a+1)-1) * (3^(b+1)-1)/2 * (5^(c+1)-1)/4 * (7^(d+1)-1)/6
    # M = k * sigma(k)
    
    # To keep M <= 10^18, we need to bound the exponents.
    # log10(M) <= 18.
    # log10(k) + log10(sigma(k)) <= 18.
    # Since sigma(k) ~ k * (log log k), M ~ k^2 log log k.
    # k^2 ~ 10^18 => k ~ 10^9.
    # 2^a 3^b 5^c 7^d <= 10^9.
    
    # We can generate all such k and compute M.
    # Number of such k is manageable.
    
    # Let's use a recursive generation or nested loops.
    # Given the small number of primes, nested loops are fine.
    
    # Primes: 2, 3, 5, 7
    # Max exponent for 2: 2^30 ~ 10^9, so a in [0, 30]
    # Max exponent for 3: 3^18 ~ 4*10^8, so b in [0, 18]
    # Max exponent for 5: 5^13 ~ 1.2*10^9, so c in [0, 13]
    # Max exponent for 7: 7^10 ~ 2.8*10^8, so d in [0, 10]
    
    # Total iterations: 31 * 19 * 14 * 11 ~ 90,000. Very fast.
    
    m_to_k = defaultdict(list)
    
    # Precompute sigma parts for efficiency? Or just calculate.
    # Calculating powers is fast.
    
    for a in range(31):
        if time.time() > deadline - 10:
            break
        p2 = 2 ** a
        s2 = 2 ** (a + 1) - 1
        
        for b in range(19):
            p3 = 3 ** b
            s3 = (3 ** (b + 1) - 1) // 2
            
            base_k = p2 * p3
            if base_k > 10**9: # k must be <= 10^9 roughly
                break
                
            base_sigma = s2 * s3
            
            for c in range(14):
                p5 = 5 ** c
                s5 = (5 ** (c + 1) - 1) // 4
                
                k2 = base_k * p5
                if k2 > 10**9:
                    break
                    
                sigma2 = base_sigma * s5
                
                for d in range(11):
                    p7 = 7 ** d
                    s7 = (7 ** (d + 1) - 1) // 6
                    
                    k = k2 * p7
                    if k > 10**9:
                        break
                        
                    sigma_k = sigma2 * s7
                    M = k * sigma_k
                    
                    if M <= 10**18:
                        m_to_k[M].append(k)
                    else:
                        # Since k increases with d, M increases. We can break?
                        # Yes, M is increasing with d.
                        break

    # Find the M with the most solutions
    for M, k_list in m_to_k.items():
        if len(k_list) > max_count:
            max_count = len(k_list)
            best_M = M
            best_k_list = k_list
            
            # Optional: Sort k_list
            # best_k_list.sort()
            
            # Early exit if we find a very high count?
            # Unlikely to find > 10 in this space, but 4-5 is good.
            if max_count >= 10:
                break

    if best_M and max_count > 0:
        # Verify solutions
        valid_k = []
        for k in best_k_list:
            if k <= math.isqrt(best_M):
                s = _sigma_fast(k)
                if k * s == best_M:
                    valid_k.append(k)
        
        if len(valid_k) > 0:
            return {'M': best_M, 'k': valid_k}

    # Fallback to previous logic if constructive search fails or finds few
    # But constructive search is very fast, so we can run the sieve as a second pass if time permits?
    # Or just return the best found.
    
    if best_M:
        return {'M': best_M, 'k': best_k_list}

    # If even this fails (shouldn't happen), fallback to known solution
    return {'M': 333312, 'k': [336, 372, 434]}

def _sigma_fast(n):
    """Compute sigma(n) by trial division, optimized."""
    if n == 1:
        return 1
    result = 1
    temp = n
    # Check 2
    if temp % 2 == 0:
        exp = 0
        while temp % 2 == 0:
            temp //= 2
            exp += 1
        result *= (2 ** (exp + 1) - 1)
    # Check odd primes
    p = 3
    while p * p <= temp:
        if temp % p == 0:
            exp = 0
            while temp % p == 0:
                temp //= p
                exp += 1
            result *= (p ** (exp + 1) - 1) // (p - 1)
        p += 2
    if temp > 1:
        result *= (temp * temp - 1) // (temp - 1)
    return result
