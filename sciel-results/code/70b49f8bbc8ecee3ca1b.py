import math

def construct():
    N = 123457
    # Sieve of Eratosthenes to find smallest prime factor
    spf = list(range(N + 1))
    for i in range(2, int(math.isqrt(N)) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, N + 1, i):
                if spf[j] == j:
                    spf[j] = i

    def largest_prime_factor(n):
        if n == 1:
            return 1
        max_pf = 1
        while n > 1:
            p = spf[n]
            max_pf = max(max_pf, p)
            while n % p == 0:
                n //= p
        return max_pf

    best_a = 1
    best_b = N - 1
    best_val = max(largest_prime_factor(best_a), largest_prime_factor(best_b))
    
    # We only need to check up to N // 2 because a and b are symmetric
    for a in range(2, N // 2 + 1):
        b = N - a
        pa = largest_prime_factor(a)
        pb = largest_prime_factor(b)
        current_max = max(pa, pb)
        if current_max < best_val:
            best_val = current_max
            best_a = a
            best_b = b
            # If we reach a very low value, we can stop early? 
            # The minimum possible value is 2 (if both are powers of 2, but sum is odd, impossible).
            # Next is 3 (one is power of 2, other power of 3? 2^a + 3^b = 123457).
            # If we find a solution with max prime factor 7, it's likely very good.
            # But let's just run through all to be sure, it's fast enough.
            
    return [best_a, best_b]
