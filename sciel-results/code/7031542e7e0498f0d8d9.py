import time
import numpy as np
from math import gcd
from itertools import product

def construct():
    start_time = time.time()
    DEADLINE = 75  # safety margin before 90s kill
    
    MAX_N = 1000001  # we need up to 1000000, and n+1 can be 1000001
    
    # Sieve to find primes up to MAX_N
    is_prime = np.ones(MAX_N + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(MAX_N**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    primes = np.where(is_prime)[0]
    
    # Initialize powerful_part array
    pp = np.ones(MAX_N + 1, dtype=np.int64)
    
    # For each prime, accumulate the full prime power into multiples
    for p in primes:
        if p * p > MAX_N:
            break
        p2 = p * p
        # Find the highest power p^e <= MAX_N
        powers = []
        pe = p2
        while pe <= MAX_N:
            powers.append(pe)
            pe_next = pe * p
            if pe_next > MAX_N:
                break
            pe = pe_next
        
        # Apply p^2 to all multiples of p^2
        multiples_p2 = np.arange(p2, MAX_N + 1, p2)
        pp[multiples_p2] *= p2
        
        # Apply additional p factors for higher powers
        pe = p2 * p  # p^3
        while pe <= MAX_N:
            multiples_pe = np.arange(pe, MAX_N + 1, pe)
            pp[multiples_pe] *= p
            pe *= p
        
        if time.time() - start_time > DEADLINE:
            break
    
    # Now compute the product pp[n] * pp[n+1] for n in [10000, 1000000]
    # and find the n that maximizes it
    products = pp[10000:1000001] * pp[10001:1000002]
    
    # Find the index of maximum
    max_idx = np.argmax(products)
    best_n = 10000 + max_idx
    best_val = int(products[max_idx])
    
    # Verify
    if best_n < 10000 or best_n > 1000000:
        best_n = 10000
        best_val = int(pp[10000] * pp[10001])
    
    # CRT-based targeted search: look for pairs (P, Q) of prime powers
    # where P divides n and Q divides n+1, with P*Q large
    # We want n ≡ 0 (mod P) and n ≡ -1 (mod Q), i.e., n ≡ P-1 (mod lcm(P,Q))
    # Actually: n ≡ 0 mod P, n ≡ -1 mod Q
    # By CRT: n = P * k, and P*k ≡ -1 (mod Q), so k ≡ -P^{-1} (mod Q)
    
    # Generate candidate prime powers
    prime_powers = []
    for p in primes:
        if p * p > MAX_N:
            break
        pe = p * p
        while pe <= MAX_N:
            prime_powers.append(pe)
            pe *= p
            if pe > MAX_N:
                break
    
    # Sort by value descending to prioritize large prime powers
    prime_powers.sort(reverse=True)
    
    # Try pairs of prime powers from different primes
    # For efficiency, only try top K prime powers
    K = min(200, len(prime_powers))
    top_pp = prime_powers[:K]
    
    # Track best found
    best_crt_n = best_n
    best_crt_val = best_val
    
    # Try pairs (P, Q) where gcd(P, Q) = 1
    # n ≡ 0 (mod P), n ≡ -1 (mod Q)
    # Solution: n = P * t where P*t ≡ -1 (mod Q)
    # t ≡ -P^{-1} (mod Q)
    # So t = (Q - (P^{-1} mod Q)) mod Q, and n = P * t
    # Then n + Q*P*m for m = 0, 1, 2, ...
    
    # Precompute modular inverses for efficiency
    # For each pair, check if gcd(P,Q)==1, compute solution, check range
    
    for i in range(len(top_pp)):
        if time.time() - start_time > DEADLINE - 5:
            break
        P = top_pp[i]
        for j in range(i + 1, len(top_pp)):
            Q = top_pp[j]
            if gcd(P, Q) != 1:
                continue
            
            # Find t such that P*t ≡ -1 (mod Q)
            # i.e., P*t ≡ Q-1 (mod Q)
            # t ≡ (Q-1) * P^{-1} (mod Q)
            
            # Compute P^{-1} mod Q using extended Euclidean
            try:
                # pow(P, -1, Q) works in Python 3.8+
                P_inv = pow(P, -1, Q)
            except ValueError:
                continue
            
            t = ((Q - 1) * P_inv) % Q
            if t == 0:
                t = Q  # n = P*Q, check if in range
            
            n0 = P * t
            
            # Check if n0 is in [10000, 1000000]
            # Also check n0 + k*lcm(P,Q) for small k
            lcm_PQ = P * Q  # since gcd(P,Q)=1
            
            # Find the smallest m such that n0 + m*lcm_PQ >= 10000
            if n0 >= 10000 and n0 <= 1000000:
                # Check this n
                val = int(pp[n0] * pp[n0 + 1])
                if val > best_crt_val:
                    best_crt_val = val
                    best_crt_n = n0
            
            # Check nearby values
            m_start = max(0, (10000 - n0 + lcm_PQ - 1) // lcm_PQ)
            m_end = min(5, (1000000 - n0) // lcm_PQ)
            for m in range(m_start, m_end + 1):
                n_check = n0 + m * lcm_PQ
                if 10000 <= n_check <= 1000000:
                    val = int(pp[n_check] * pp[n_check + 1])
                    if val > best_crt_val:
                        best_crt_val = val
                        best_crt_n = n_check
    
    if best_crt_val > best_val:
        best_n = best_crt_n
        best_val = best_crt_val
    
    return int(best_n)
