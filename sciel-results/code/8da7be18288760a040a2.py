"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def _isqrt(n):
    return math.isqrt(n)

def _sigma_by_factorization(n):
    if n <= 0:
        return 0
    result = 1
    temp = n
    if temp % 2 == 0:
        s = 1
        p_power = 1
        while temp % 2 == 0:
            temp //= 2
            p_power *= 2
            s += p_power
        result *= s
    f = 3
    while f * f <= temp:
        if temp % f == 0:
            s = 1
            p_power = 1
            while temp % f == 0:
                temp //= f
                p_power *= f
                s += p_power
            result *= s
        f += 2
    if temp > 1:
        result *= (1 + temp)
    return result

def _verify(M, k):
    if k < 1 or k > _isqrt(M):
        return False
    return k * _sigma_by_factorization(k) == M

def construct():
    t0 = time.time()
    deadline = t0 + 110.0

    # M = P * sigma(P), P = product of n distinct primes
    # Every divisor d of P: d*sigma(d) = M
    # Count = 2^n, constraint M <= 10^18, k <= isqrt(M)
    # Since sigma(P) > P for P>1, P < sqrt(M), so all divisors valid

    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113]

    best_M = None
    best_k = None
    best_count = 0

    # Try consecutive smallest primes, find max n that fits
    for n in range(len(primes), 2, -1):
        if time.time() > deadline - 20:
            break
        subset = primes[:n]
        M = 1
        for p in subset:
            M *= p * (1 + p)
        if M > 10**18:
            continue
        if M < 1:
            continue

        # Generate divisors of P (product of first n primes)
        P = 1
        for p in subset:
            P *= p

        # 2^n divisors; for n > 20 this is too many to list (cap at 1000)
        if (1 << n) > 1000:
            # We can only list at most 1000, so count is capped
            # But we still want the best M. Let's just use n where 2^n <= 1000
            # 2^9 = 512, 2^10 = 1024 > 1000
            # So max useful n is 9 for full listing
            # But wait - the task says "at most 1000 distinct ints"
            # So we can only report 1000. Let's find best n where 2^n <= 1000
            pass

        # Generate all divisors
        divisors = [1]
        for p in subset:
            new_divs = [d * p for d in divisors]
            divisors = new_divs + divisors

        sqrt_M = _isqrt(M)
        valid_k = [d for d in divisors if 1 <= d <= sqrt_M]

        # Cap at 1000
        if len(valid_k) > 1000:
            valid_k = valid_k[:1000]

        # Verify all (or sample if large)
        ok = True
        if len(valid_k) <= 2000:
            for k in valid_k:
                if not _verify(M, k):
                    ok = False
                    break
        else:
            ok = True  # trust the math for large sets

        if ok and len(valid_k) > best_count:
            best_count = len(valid_k)
            best_M = M
            best_k = valid_k

        # If 2^n > 1000, we can't list them all, so n=9 (512) is max for full list
        # But maybe we can get more with a different approach
        if (1 << n) > 1000:
            # Try n=9 (512 divisors)
            break

    # If we broke early due to cap, explicitly try n=9
    if best_count < 512:
        n = 9
        subset = primes[:n]
        M = 1
        for p in subset:
            M *= p * (1 + p)
        if M <= 10**18:
            P = 1
            for p in subset:
                P *= p
            divisors = [1]
            for p in subset:
                new_divs = [d * p for d in divisors]
                divisors = new_divs + divisors
            sqrt_M = _isqrt(M)
            valid_k = [d for d in divisors if 1 <= d <= sqrt_M]
            if len(valid_k) <= 1000:
                ok = all(_verify(M, k) for k in valid_k)
                if ok and len(valid_k) > best_count:
                    best_count = len(valid_k)
                    best_M = M
                    best_k = valid_k

    if best_M is None or best_count < 3:
        return {'M': 333312, 'k': [336, 372, 434]}

    return {'M': int(best_M), 'k': [int(k) for k in best_k]}
# EVOLVE-BLOCK-END
