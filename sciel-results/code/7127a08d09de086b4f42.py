import time
import numpy as np

def construct():
    deadline = time.time() + 75.0  # safety margin before 90s kill
    MAX_N = 10_000_000
    TARGET_NUM, TARGET_DEN = 3, 2

    # Precompute sigma(n) for all n up to MAX_N using a sieve.
    # sigma[n] = sum of all positive divisors of n.
    # Method: for each d from 1 to MAX_N, add d to all multiples of d.
    # This is O(N log N) but with NumPy vectorization it's fast.
    # Actually, a better approach: use the multiplicative formula via prime factorization.
    # But the sieve approach with NumPy:
    # sigma = np.zeros(MAX_N+1, dtype=np.int64)
    # for d in range(1, MAX_N+1):
    #     sigma[d::d] += d
    # This is too slow in pure Python loop. Let's use a different approach.

    # Better: compute sigma via the formula sigma(n) = product over p^k || n of (p^(k+1)-1)/(p-1)
    # We can do this with a modified sieve that tracks the largest prime power contribution.

    # Actually, let's use the standard O(N log log N) sieve for sigma:
    # sigma[1] = 1
    # For each prime p, for each multiple m of p, we can update sigma.
    # But this requires tracking exponents.

    # Simplest fast approach: use the divisor sum sieve with NumPy in chunks.
    # sigma[n] = sum_{d|n} d
    # We can compute this by: for d = 1 to N, for k = 1, 2, ... while d*k <= N: sigma[d*k] += d
    # In NumPy: for d in range(1, N+1): sigma[d::d] += d
    # The inner operation is vectorized, so each d takes O(N/d) time but vectorized.
    # Total operations ~ N * H(N) ~ N * ln(N) ~ 10^7 * 16 ~ 1.6e8, which might be too slow.

    # Alternative: use the multiplicative property with a linear sieve (O(N)).
    # Let's implement a linear sieve for sigma.

    N = MAX_N
    sigma = np.zeros(N + 1, dtype=np.int64)
    sigma[1] = 1
    is_prime = np.ones(N + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    primes = []

    # We need: for each n, track the largest prime power p^k dividing n, and sigma(n) / sigma(p^k)
    # Let's use arrays:
    # sp[n] = largest prime power p^k dividing n
    # sig[n] = sigma(n)
    # We can compute using: if n = p^k * m where p doesn't divide m,
    # then sigma(n) = sigma(p^k) * sigma(m)

    # Linear sieve approach:
    # For each i from 2 to N:
    #   for each prime p in primes (while i*p <= N):
    #     mark i*p as composite
    #     if i % p == 0:
    #       # p divides i, so the largest prime power of i*p is p * sp[i]
    #       sp[i*p] = sp[i] * p
    #       sig[i*p] = sig[i] * (1 + p + ... + p^(k+1)) / (1 + p + ... + p^k) * ... 
    #       Actually: if sp[i] = p^k, then sp[i*p] = p^(k+1)
    #       sigma(p^(k+1)) = sigma(p^k) * (1 + p) ... no.
    #       sigma(p^k) = (p^(k+1)-1)/(p-1)
    #       sigma(p^(k+1)) = (p^(k+2)-1)/(p-1) = sigma(p^k) * (1 + p) ... let's verify:
    #       sigma(p^k) * (1+p) = (p^(k+1)-1)(1+p)/(p-1) = (p^(k+2)+p^(k+1)-p-1)/(p-1)
    #       sigma(p^(k+1)) = (p^(k+2)-1)/(p-1)
    #       These are not equal. Let me just compute directly.

    # Let me use a simpler approach: precompute using the divisor sieve but with NumPy acceleration.
    # Actually, let me just use the straightforward approach and see if it's fast enough.

    # Let me try a different strategy: only compute sigma for candidates we actually need.
    # We can compute sigma(n) on demand using trial division up to sqrt(n).
    # For each candidate pair, we compute sigma(a) and sigma(b) via factorization.
    # If we have ~10^6 candidates, and each factorization takes ~sqrt(10^7) ~ 3000 ops worst case,
    # that's ~3e9 ops, too slow.

    # Better: precompute smallest prime factor (SPF) up to N, then compute sigma(n) in O(log n) per n.
    # SPF sieve is O(N log log N) and fast.

    # Step 1: Compute SPF up to N
    spf = np.zeros(N + 1, dtype=np.int64)
    for i in range(2, N + 1):
        if spf[i] == 0:
            # i is prime
            spf[i] = i
            if i * i <= N:
                for j in range(i * i, N + 1, i):
                    if spf[j] == 0:
                        spf[j] = i

    # Step 2: Compute sigma(n) for all n using SPF
    sigma = np.zeros(N + 1, dtype=np.int64)
    sigma[1] = 1
    for n in range(2, N + 1):
        p = spf[n]
        m = n // p
        if m % p == 0:
            # p divides m, so n = p * m where p | m
            # Let m = p^k * r where p doesn't divide r
            # We need to track the power of p in n
            # Actually, let's use a different recurrence.
            # sigma(n) where n = p^k * m, p not dividing m:
            # sigma(n) = sigma(p^k) * sigma(m)
            # We can precompute sigma_prime_power or use:
            # Let's define: for each n, let p^k be the highest power of spf[n] dividing n.
            # This is getting complicated. Let me just do trial division with SPF for each candidate.
            pass
        else:
            # p does not divide m, so n = p * m, gcd(p, m) = 1
            # sigma(n) = sigma(p) * sigma(m) = (1 + p) * sigma(m)
            sigma[n] = (1 + p) * sigma[m]

    # The above doesn't handle the case where p divides m correctly.
    # Let me redo this properly.

    # Proper linear-time sigma computation using SPF:
    # For each n, write n = p^k * m where p = spf[n] and p does not divide m.
    # sigma(n) = sigma(p^k) * sigma(m)
    # We need to know k and m. We can compute:
    #   p_power = 1
    #   temp = n
    #   while temp % p == 0:
    #       p_power *= p
    #       temp //= p
    #   m = temp
    #   sigma(n) = sigma(p_power) * sigma(m)
    # But sigma(p_power) = (p^(k+1) - 1) / (p - 1)

    # Let me just compute sigma for all n using this method.
    sigma = np.zeros(N + 1, dtype=np.int64)
    sigma[1] = 1
    for n in range(2, N + 1):
        p = spf[n]
        # Find the highest power of p dividing n
        p_power = 1
        temp = n
        while temp % p == 0:
            p_power *= p
            temp //= p
        m = temp
        # sigma(p_power) = (p^(k+1) - 1) / (p - 1) where p_power = p^k
        # p^(k+1) = p_power * p
        sigma_p_power = (p_power * p - 1) // (p - 1)
        sigma[n] = sigma_p_power * sigma[m]

    # Now search for pairs (a, b) with sigma(a) = sigma(b) and a/b close to 3/2.
    # For each b from 1 to N, compute a = round(3*b/2), check if a <= N and a > 0.
    # Check if sigma[a] == sigma[b] and a != b (to avoid a/b = 1 which is not 3/2).
    # Actually, a/b = 3/2 exactly would mean 2a = 3b, i.e., a = 3b/2.
    # If b is even, a = 3b/2 is integer. Check if sigma(a) == sigma(b).
    # If b is odd, a is not integer for exact 3/2, so we check a = round(3b/2).

    best_err = float('inf')
    best_a, best_b = 11, 6  # fallback: sigma(11)=12, sigma(6)=12, |11/6 - 3/2| = |11/6 - 9/6| = 2/6 = 1/3

    # First check exact 3/2: b even, a = 3b/2
    for b in range(2, N + 1, 2):
        a = 3 * b // 2
        if a > N:
            break
        if a == b:
            continue
        if sigma[a] == sigma[b]:
            err = abs(a / b - 3 / 2)
            # err should be 0 if 2a == 3b, but let's compute exactly
            # a/b - 3/2 = (2a - 3b) / (2b) = 0 if 2a = 3b
            # Since a = 3b/2 and b is even, 2a = 3b exactly, so err = 0.
            # But the task says raw value of exactly 0 is invalid!
            # So we need a/b != 3/2 exactly.
            # This means we should NOT use exact 3/2 pairs.
            pass

    # So we need a/b close to but not equal to 3/2.
    # For each b, consider a = floor(3b/2) and a = ceil(3b/2) (if different from 3b/2).
    # Actually, for b even, 3b/2 is integer, so we check a = 3b/2 - 1 and a = 3b/2 + 1.
    # For b odd, 3b/2 is not integer, so a = floor(3b/2) or a = ceil(3b/2).

    for b in range(1, N + 1):
        if time.time() > deadline - 10:
            break
        a_base = 3 * b // 2
        candidates_a = set()
        # Add neighbors around 3b/2
        for da in range(-5, 6):
            a = a_base + da
            if 1 <= a <= N and a != b:
                candidates_a.add(a)
        for a in candidates_a:
            if sigma[a] == sigma[b]:
                # Compute |a/b - 3/2| = |2a - 3b| / (2b)
                num = abs(2 * a - 3 * b)
                if num == 0:
                    continue  # exact 3/2, invalid
                den = 2 * b
                err = num / den
                if err < best_err:
                    best_err = err
                    best_a, best_b = a, b

    # Also try a wider search for small b where we can check more a values
    for b in range(1, min(100000, N + 1)):
        if time.time() > deadline - 10:
            break
        a_base = 3 * b // 2
        # Check a in a wider range
        for da in range(-20, 21):
            a = a_base + da
            if 1 <= a <= N and a != b:
                if sigma[a] == sigma[b]:
                    num = abs(2 * a - 3 * b)
                    if num == 0:
                        continue
                    den = 2 * b
                    err = num / den
                    if err < best_err:
                        best_err = err
                        best_a, best_b = a, b

    return [int(best_a), int(best_b)]
