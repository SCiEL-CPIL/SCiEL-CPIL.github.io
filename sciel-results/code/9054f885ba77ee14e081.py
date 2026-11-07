import math
import time
import sys
from collections import defaultdict

def construct():
    t0 = time.time()
    DEADLINE = 85.0
    N = 10000000

    # 1. Sieve for Smallest Prime Factor (SPF)
    # This is necessary for fast factorization and sigma calculation.
    spf = list(range(N + 1))
    for i in range(2, int(N**0.5) + 1):
        if spf[i] == i:
            for j in range(i*i, N + 1, i):
                if spf[j] == j:
                    spf[j] = i

    # Helper to compute sigma(n) using SPF
    def get_sigma(n):
        if n <= 1:
            return n
        res = 1
        while n > 1:
            p = spf[n]
            power_sum = 1
            power = 1
            while n % p == 0:
                n //= p
                power *= p
                power_sum += power
            res *= power_sum
        return res

    # Fallback solution: sigma(6)=12, sigma(11)=12. Ratio 11/6 = 1.833. Error ~0.333.
    # We expect to find much better ones.
    best_val = float('inf')
    best_pair = [11, 6]

    # Strategy:
    # We want a/b ~ 3/2.
    # Let a = 3k + r_a, b = 2k + r_b? No, simple scaling.
    # Let's iterate over b.
    # For a given b, a should be close to 1.5 * b.
    # Specifically, a = (3b + d) / 2.
    # If b is odd, 3b is odd, so d must be odd for a to be integer. Smallest |d| is 1.
    #   a1 = (3b + 1) / 2  => error = 1/(2b)
    #   a2 = (3b - 1) / 2  => error = 1/(2b)
    # If b is even, 3b is even, so d must be even. Smallest non-zero |d| is 2 (d=0 gives a=1.5b, error 0, invalid).
    #   a1 = (3b + 2) / 2  => error = 2/(2b) = 1/b
    #   a2 = (3b - 2) / 2  => error = 2/(2b) = 1/b
    
    # So for odd b, the best possible error is 1/(2b).
    # For even b, the best possible error is 1/b.
    
    # We iterate b from N down to 1.
    # We can stop if the theoretical minimum error for the current b is greater than best_val.
    # For odd b: min_err = 1/(2b). If 1/(2b) > best_val, then for all smaller b, min_err will be larger.
    # So if we reach an odd b where 1/(2b) > best_val, we can stop.
    # Since b is decreasing, 1/(2b) is increasing.
    # So once 1/(2b) > best_val, we can break.
    
    # Note: We check odd b first? No, we just iterate down.
    # But we can optimize the stop condition.
    # If best_val is small, we stop early.
    
    # To speed up, we can group numbers by sigma.
    # But sigma values can be large.
    # Let's try the direct search first, it might be fast enough if we find a small error early.
    # If we find a pair with error 1/(2*10^7) = 5e-8, that's very small.
    # Is it likely that for large b, there exists a with sigma(a)=sigma(b) and a close to 1.5b?
    # Probably not for the very first large b we check.
    # But we can check a few candidates for each b.
    
    # Let's refine the loop.
    # We will check b from N down to 1.
    # For each b, compute s_b = sigma(b).
    # Then check candidate a's.
    
    # Optimization:
    # Instead of checking ALL b, maybe we can restrict to b with certain properties?
    # No obvious property.
    
    # Let's just run the loop with the stop condition.
    
    for b in range(N, 0, -1):
        if time.time() - t0 > DEADLINE:
            break
            
        # Stop condition check
        # If b is odd, min possible error is 1/(2b).
        # If 1/(2b) > best_val, then no smaller b can beat best_val.
        # Since b is decreasing, this condition will hold for all subsequent b.
        # So we can break.
        if b % 2 == 1:
            if 1.0 / (2.0 * b) > best_val:
                break
        
        s_b = get_sigma(b)
        
        if b % 2 == 1:
            # Candidates: (3b+1)/2, (3b-1)/2
            a1 = (3 * b + 1) // 2
            a2 = (3 * b - 1) // 2
            
            if 1 <= a1 <= N and a1 != b:
                if get_sigma(a1) == s_b:
                    val = 1.0 / (2.0 * b)
                    if val < best_val:
                        best_val = val
                        best_pair = [a1, b]
                        # Since this is the minimum possible error for this b,
                        # and we are iterating b downwards,
                        # and the stop condition is based on this min error,
                        # we can return immediately?
                        # Wait, if we find a valid pair for odd b, its error is 1/(2b).
                        # Any smaller b' will have min error 1/(2b') > 1/(2b).
                        # So this is the global optimum!
                        return [int(a1), int(b)]
            
            if 1 <= a2 <= N and a2 != b:
                if get_sigma(a2) == s_b:
                    val = 1.0 / (2.0 * b)
                    if val < best_val:
                        best_val = val
                        best_pair = [a2, b]
                        return [int(a2), int(b)]
                        
        else:
            # Candidates: (3b+2)/2, (3b-2)/2
            a1 = (3 * b + 2) // 2
            a2 = (3 * b - 2) // 2
            
            if 1 <= a1 <= N and a1 != b:
                if get_sigma(a1) == s_b:
                    val = 1.0 / b
                    if val < best_val:
                        best_val = val
                        best_pair = [a1, b]
                        # Can we return?
                        # No, because for odd b' < b, the min error is 1/(2b').
                        # If b' is close to b, 1/(2b') < 1/b.
                        # Example: b=10, err=0.1. b'=9, min err=1/18=0.055.
                        # So we cannot return immediately for even b.
                        # We must continue searching until the stop condition is met.
                        pass
            
            if 1 <= a2 <= N and a2 != b:
                if get_sigma(a2) == s_b:
                    val = 1.0 / b
                    if val < best_val:
                        best_val = val
                        best_pair = [a2, b]
                        pass

    return [int(best_pair[0]), int(best_pair[1])]
