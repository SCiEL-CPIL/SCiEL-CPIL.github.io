"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from fractions import Fraction
import numpy as np

MAX_N = 10_000_000
TARGET = Fraction(3, 2)
DEADLINE_SECONDS = 85  # safety margin before 90s kill

def construct():
    t0 = time.time()
    N = MAX_N
    
    # 1. Compute Smallest Prime Factor (SPF) array using NumPy
    # This is O(N log log N) and fast in NumPy
    spf = np.zeros(N + 1, dtype=np.int64)
    # Initialize spf for primes
    # We can use a sieve-like approach
    # spf[i] = i if i is prime
    # For composites, spf[i] is the smallest prime factor
    
    # Efficient SPF sieve
    # Initialize with 0
    for i in range(2, N + 1):
        if spf[i] == 0:
            # i is prime
            spf[i::i] = i
    
    # 2. Function to compute sigma(n) using SPF
    # sigma(n) is multiplicative.
    # If n = p1^e1 * p2^e2 * ...
    # sigma(n) = (1+p1+...+p1^e1) * (1+p2+...+p2^e2) * ...
    
    def sigma(n):
        if n == 1:
            return 1
        res = 1
        while n > 1:
            p = spf[n]
            # p is smallest prime factor
            power_sum = 1
            power = 1
            while n % p == 0:
                n //= p
                power *= p
                power_sum += power
            res *= power_sum
        return res

    best_err = None
    best_pair = (11, 6)
    
    # Check seed
    if sigma(11) == sigma(6):
        best_err = abs(Fraction(11, 6) - TARGET)
        best_pair = (11, 6)

    # 3. Iterate k = 2a - 3b
    # We want to minimize k / (2b)
    # So for small k, we want large b.
    # a = (3b + k) / 2
    # a <= N => 3b + k <= 2N => b <= (2N - k) / 3
    # b >= 1
    
    # Parity: 3b + k must be even.
    # If k is even, b is even.
    # If k is odd, b is odd.
    
    # We iterate k from 1 to K_MAX.
    # For each k, we check the largest possible b, then b-2, b-4, etc.
    # How many b's to check per k?
    # If we check too many, it might be slow.
    # If we check too few, we might miss the pair for that k.
    # However, sigma collisions are not that rare for large numbers?
    # Actually, sigma(n) can be large.
    # Let's check a reasonable number of candidates, say 1000 per k.
    
    K_MAX = 1000
    CHECKS_PER_K = 1000
    
    for k in range(1, K_MAX + 1):
        if time.time() - t0 > DEADLINE_SECONDS - 5:
            break
            
        # Determine max b
        max_b = (2 * N - k) // 3
        if max_b < 1:
            break
            
        # Adjust max_b to correct parity
        # 3b + k is even => b parity is determined by k
        # 3b % 2 == b % 2. So b % 2 == k % 2.
        if max_b % 2 != k % 2:
            max_b -= 1
            
        # Check b = max_b, max_b - 2, ...
        # We stop if error becomes larger than current best?
        # Error = k / (2b). As b decreases, error increases.
        # So if k / (2b) > best_err, we can stop checking smaller b for this k.
        # Also, for larger k, if k / (2 * max_b) > best_err, we can potentially break entirely?
        # Not necessarily, because a larger k with a much larger b? No, b is bounded by N.
        # Max b for any k is approx 2N/3.
        # So min error for k is k / (2 * (2N/3)) = 3k / 4N.
        # If 3k / 4N > best_err, then no larger k can improve.
        
        # Check lower bound for k
        # best_err is Fraction.
        # 3k / (4N) > best_err => k > 4N * best_err / 3
        
        if best_err is not None:
            # Check if we can break
            # 3k / 4N > best_err
            # k > 4N * best_err / 3
            # Use float approximation for speed, but be careful
            min_k_needed = (4 * N * float(best_err)) / 3.0
            if k > min_k_needed:
                break
                
        for _ in range(CHECKS_PER_K):
            if time.time() - t0 > DEADLINE_SECONDS - 5:
                break
                
            b = max_b
            a = (3 * b + k) // 2
            
            if a > N or a < 1:
                break
                
            # Check sigma
            if sigma(a) == sigma(b):
                err = abs(Fraction(a, b) - TARGET)
                if best_err is None or err < best_err:
                    best_err = err
                    best_pair = (a, b)
                    
            # Move to next candidate b
            # b decreases by 2 to maintain parity
            max_b -= 2
            if max_b < 1:
                break
                
    # Verify final result
    a_final, b_final = best_pair
    # Re-verify sigma just in case
    if sigma(a_final) != sigma(b_final):
        return [11, 6]
        
    err_final = abs(Fraction(a_final, b_final) - TARGET)
    if err_final == 0:
        return [11, 6]
        
    return [int(a_final), int(b_final)]

# EVOLVE-BLOCK-END
