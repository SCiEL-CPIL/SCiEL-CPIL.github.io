import math
import time

def construct():
    N = 123457
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill

    # Sieve of Eratosthenes to find smallest prime factor (SPF)
    # This allows O(log n) factorization for any number up to N
    spf = list(range(N + 1))
    for i in range(2, int(math.isqrt(N)) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, N + 1, i):
                if spf[j] == j:
                    spf[j] = i

    # Precompute Largest Prime Factor (LPF) for all numbers up to N
    # Using the SPF array, we can compute LPF efficiently
    lpf = [1] * (N + 1)
    for i in range(2, N + 1):
        n = i
        max_pf = 1
        while n > 1:
            p = spf[n]
            if p > max_pf:
                max_pf = p
            # Divide out all occurrences of p
            while n % p == 0:
                n //= p
        lpf[i] = max_pf

    # Search for a, b such that a + b = N
    # minimizing max(lpf[a], lpf[b])
    # We only need to check up to N // 2 because a and b are symmetric
    
    best_a = 1
    best_b = N - 1
    best_val = max(lpf[best_a], lpf[best_b])

    for a in range(2, N // 2 + 1):
        if time.time() > deadline:
            break
        
        b = N - a
        pa = lpf[a]
        pb = lpf[b]
        current_max = pa if pa > pb else pb
        
        if current_max < best_val:
            best_val = current_max
            best_a = a
            best_b = b
            
            # Early termination: 
            # Since N is odd, one of a,b is even, the other is odd.
            # P(even) >= 2 (if even > 1), P(odd) >= 3 (if odd > 1).
            # If a=1, P(1)=1. If b=1, P(1)=1.
            # The absolute minimum possible for max(P(a), P(b)) is:
            # If we can find a=1, b=N-1. P(1)=1, P(N-1).
            # If we can find a such that P(a)=2 (power of 2) and P(b)=3 (power of 3)?
            # 2^x + 3^y = 123457.
            # 123457 is not divisible by 2 or 3.
            # Let's check if best_val can be 3.
            # If best_val == 3, then both a and b must have largest prime factor <= 3.
            # This means a and b are of the form 2^i * 3^j.
            # Since a+b is odd, one is even, one is odd.
            # The odd one must be 3^j (since it's odd and only factors 2,3).
            # The even one must be 2^i * 3^k.
            # So 3^j + 2^i * 3^k = 123457.
            # 123457 is not divisible by 3 (sum of digits 22).
            # If j > k, LHS is divisible by 3^k. 123457 not div by 3. So k=0.
            # Then 3^j + 2^i = 123457.
            # 123457 - 3^j must be a power of 2.
            # Max power of 3 < 123457: 3^10 = 59049, 3^11 = 177147.
            # So j <= 10.
            # We can check these 11 cases.
            # If we find a solution with max=3, it's the theoretical minimum (since P(odd)>=3).
            # Actually, if a=1, P(a)=1. Then we need P(b) minimized.
            # But 1 + 123456. P(123456) is large.
            # So the minimum is likely > 3.
            # However, if we find best_val == 3, we can break.
            # If we find best_val == 4? P(a)=4 is impossible (4 is not prime). P(a) is always prime or 1.
            # So next possible values are 5, 7, etc.
            
            if best_val == 3:
                break

    return [best_a, best_b]
