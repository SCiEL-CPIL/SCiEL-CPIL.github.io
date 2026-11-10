import time
import sys
import numpy as np
from fractions import Fraction

# EVOLVE-BLOCK-START
import math

MAX_N = 10_000_000
DEADLINE_SECONDS = 85

def construct():
    t0 = time.time()
    
    def time_left():
        return DEADLINE_SECONDS - (time.time() - t0)

    N = MAX_N

    # Step 1: Linear Sieve for Smallest Prime Factor (SPF)
    # spf[i] is the smallest prime factor of i
    spf = np.zeros(N + 1, dtype=np.int64)
    primes = []
    
    # Initialize spf for odd numbers? No, linear sieve is simple.
    # spf[1] = 1
    spf[1] = 1
    
    for i in range(2, N + 1):
        if time_left() < 10:
            return [11, 6]
            
        if spf[i] == 0:
            spf[i] = i
            primes.append(i)
            
        for p in primes:
            if p > spf[i] or i * p > N:
                break
            spf[i * p] = p

    if time_left() < 5:
        return [11, 6]

    # Step 2: Fast sigma(n) computation using spf
    # sigma(n) = product (p^(k+1)-1)/(p-1)
    # We can compute this on the fly.
    
    def calc_sigma(n):
        if n == 1:
            return 1
        s = 1
        while n > 1:
            p = spf[n]
            power = 0
            sum_term = 1
            p_power = 1
            while n % p == 0:
                n //= p
                power += 1
                p_power *= p
                sum_term += p_power
            s *= sum_term
        return s

    best_err = float('inf')
    best_pair = (11, 6)
    
    # Step 3: Search over k = |2a - 3b|
    # We iterate k from 1 upwards.
    # For each k, we find the largest b such that a is integer and 1 <= a,b <= N.
    # Then we check if sigma(a) == sigma(b).
    
    # We can stop if k / (2 * 1) > best_err is not a good bound.
    # The minimal error for a given k is achieved at max b.
    # Max b is roughly 2N/3.
    # So min_err_k ~ k / (4N/3) ~ 3k / 4N.
    # If 3k / 4N > best_err, then even the best pair for this k is worse than current best.
    # Since k increases, if this condition holds, it will hold for all larger k.
    # So we can break.
    
    k_limit = 100000 # Safety limit for k, though pruning should kick in earlier
    
    for k in range(1, k_limit + 1):
        if time_left() < 2:
            break
            
        # Pruning: if the theoretical minimum error for this k is > best_err, stop
        # min error ~ k / (2 * max_b)
        # max_b ~ 2N/3
        # min_err ~ k / (4N/3) = 3k / 4N
        if best_err < float('inf') and (3.0 * k) / (4.0 * N) >= best_err:
            break
            
        # Case 1: 2a - 3b = k
        # a = (3b + k) / 2
        # Conditions:
        # 1. 3b + k is even => b % 2 == k % 2
        # 2. a <= N => 3b + k <= 2N => b <= (2N - k) // 3
        # 3. a >= 1 => 3b + k >= 2 => b >= max(1, (2-k)//3)
        
        max_b_1 = (2 * N - k) // 3
        if max_b_1 >= 1:
            # Find largest b <= max_b_1 with b % 2 == k % 2
            b = max_b_1
            if b % 2 != k % 2:
                b -= 1
            
            if b >= 1:
                a = (3 * b + k) // 2
                if 1 <= a <= N:
                    sa = calc_sigma(a)
                    sb = calc_sigma(b)
                    if sa == sb:
                        err = abs(a / b - 1.5)
                        if err < best_err:
                            best_err = err
                            best_pair = (a, b)
                            
        # Case 2: 3b - 2a = k
        # a = (3b - k) / 2
        # Conditions:
        # 1. 3b - k is even => b % 2 == k % 2
        # 2. a >= 1 => 3b - k >= 2 => b >= (k + 2) // 3
        # 3. a <= N => 3b - k <= 2N => b <= (2N + k) // 3
        
        min_b_2 = (k + 2) // 3
        max_b_2 = (2 * N + k) // 3
        
        if max_b_2 >= min_b_2:
            # Find largest b <= max_b_2 with b % 2 == k % 2 and b >= min_b_2
            b = max_b_2
            if b % 2 != k % 2:
                b -= 1
            
            if b >= min_b_2:
                a = (3 * b - k) // 2
                if 1 <= a <= N:
                    sa = calc_sigma(a)
                    sb = calc_sigma(b)
                    if sa == sb:
                        err = abs(a / b - 1.5)
                        if err < best_err:
                            best_err = err
                            best_pair = (a, b)

    # Verify result
    a_final, b_final = best_pair
    if not (1 <= a_final <= N and 1 <= b_final <= N):
        return [11, 6]
    if a_final == b_final:
        return [11, 6]
    
    # Re-verify sigma equality to be safe
    if calc_sigma(a_final) != calc_sigma(b_final):
        return [11, 6]
        
    err_final = abs(Fraction(a_final, b_final) - Fraction(3, 2))
    if err_final == 0:
        return [11, 6]
        
    return [int(a_final), int(b_final)]

# EVOLVE-BLOCK-END
