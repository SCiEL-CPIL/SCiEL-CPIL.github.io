"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def construct():
    deadline = time.time() + 85  # safety margin before 90s kill
    N = 123457

    def largest_prime_factor(n):
        if n == 1:
            return 1
        max_pf = 1
        # trial by 2
        while n % 2 == 0:
            max_pf = 2
            n //= 2
        # trial by odd numbers
        f = 3
        while f * f <= n:
            while n % f == 0:
                max_pf = f
                n //= f
            f += 2
        if n > 1:
            max_pf = n
        return max_pf

    best_pair = (1, N - 1)
    best_score = max(largest_prime_factor(best_pair[0]), largest_prime_factor(best_pair[1]))

    # We want to minimize max(P(a), P(N-a)).
    # A good heuristic is to find a and N-a that are both "smooth" (have small largest prime factors).
    # Since N is fixed, we can search for a such that a and N-a have small P.
    # Let's iterate a from 1 to N//2.
    # But 123457 is large enough that a full scan might be slow if we do full factorization each time.
    # However, 123457 is only ~1.2e5. A full scan with optimized factorization should be well within 60s.
    # Let's optimize the factorization.
    
    # Precompute smallest prime factor (SPF) up to N?
    # N = 123457. SPF array of size 123458 is fine.
    # This allows O(log n) factorization for any n <= N.
    
    limit = N
    spf = list(range(limit + 1))
    for i in range(2, int(math.isqrt(limit)) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i*i, limit + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    def lpf_fast(n):
        if n == 1:
            return 1
        max_pf = 1
        while n > 1:
            p = spf[n]
            max_pf = p
            while n % p == 0:
                n //= p
        return max_pf

    # Re-evaluate best with fast LPF
    best_score = max(lpf_fast(best_pair[0]), lpf_fast(best_pair[1]))

    # Search
    # To speed up, we can skip even numbers if N is odd?
    # N = 123457 is odd. So a and N-a have opposite parity.
    # One is even, one is odd.
    # The even one will have P at least 2.
    # Let's just iterate all a from 1 to N//2.
    
    # We can also prune: if lpf_fast(a) >= best_score, we don't need to check N-a?
    # No, because max(P(a), P(N-a)) could be smaller if P(N-a) is very small?
    # Wait, max(x, y) >= x. So if P(a) >= best_score, then max(P(a), P(N-a)) >= best_score.
    # So we can skip if P(a) >= best_score.
    # Similarly, if P(N-a) >= best_score, we can skip.
    
    # Let's iterate.
    for a in range(1, N // 2 + 1):
        if time.time() > deadline:
            break
        pa = lpf_fast(a)
        if pa >= best_score:
            continue
        b = N - a
        pb = lpf_fast(b)
        if pb >= best_score:
            continue
        score = max(pa, pb)
        if score < best_score:
            best_score = score
            best_pair = (a, b)
            # Early termination if best_score is 1? Impossible for N>2.
            # If best_score is 2, then both are powers of 2? But N is odd, so one is odd.
            # If one is odd and P(odd)=1, it must be 1.
            # So a=1, b=N-1=123456. P(1)=1, P(123456)=?
            # 123456 = 2^4 * 3 * 5143? 5143 is prime?
            # If best_score is small, we can stop?
            # The theoretical minimum is at least 2 (since one number is even, P>=2, and the other is odd, if it's >1, P>=3? No, P(1)=1. But N-a=1 => a=N-1. P(N-1) could be large).
            # Actually, if a=1, P(a)=1. P(b)=P(123456).
            # Let's see if we can get lower.
            # Since one number is even, P(even) >= 2.
            # If the odd number is 1, P(odd)=1. Max is P(even).
            # If the odd number is >1, P(odd) >= 3 (smallest odd prime).
            # So max(P(a), P(b)) >= 2.
            # Can we achieve 2? Only if both P(a)=2 and P(b)=2.
            # This means a=2^k, b=2^m.
            # a+b = 2^k + 2^m = 123457 (odd).
            # Sum of two powers of 2 is odd only if one is 1 (2^0) and the other is odd? No, powers of 2 are even except 1.
            # 1 + 2^m = 123457 => 2^m = 123456.
            # Is 123456 a power of 2? No.
            # So best_score > 2.
            
    return [best_pair[0], best_pair[1]]
# EVOLVE-BLOCK-END
