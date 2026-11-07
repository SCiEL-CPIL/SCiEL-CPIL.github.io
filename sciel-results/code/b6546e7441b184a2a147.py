import math
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # 15s margin before 150s kill

    # Phase 1: Linear sieve for sigma(k) using NumPy
    N = 12000000

    sigma = np.zeros(N + 1, dtype=np.int64)
    p_pow = np.zeros(N + 1, dtype=np.int64)
    sigma_rest = np.zeros(N + 1, dtype=np.int64)
    spf = np.zeros(N + 1, dtype=np.int32)

    sigma[1] = 1
    p_pow[1] = 1
    sigma_rest[1] = 1

    primes = [2]
    spf[2] = 2
    sigma[2] = 3
    p_pow[2] = 2
    sigma_rest[2] = 1

    for i in range(3, N + 1):
        if spf[i] == 0:
            spf[i] = i
            primes.append(i)
            sigma[i] = 1 + i
            p_pow[i] = i
            sigma_rest[i] = 1

        for p in primes:
            if p > spf[i] or i * p > N:
                break
            ip = i * p
            spf[ip] = p
            if p == spf[i]:
                p_pow[ip] = p_pow[i] * p
                sigma_rest[ip] = sigma_rest[i]
                sigma[ip] = sigma[i] + p_pow[i] * p * sigma_rest[i]
                break
            else:
                p_pow[ip] = p
                sigma_rest[ip] = sigma[i]
                sigma[ip] = sigma[i] * (1 + p)

        if (i & 0x3FFFF) == 0 and time.time() > deadline - 50.0:
            break

    # Find the actual max k processed
    max_k_processed = N
    while max_k_processed > 1 and sigma[max_k_processed] == 0:
        max_k_processed -= 1

    # Group by k*sigma(k) using a dict
    groups = {}
    for k in range(1, max_k_processed + 1):
        c = k * sigma[k]
        if c not in groups:
            groups[c] = []
        groups[c].append(k)
        if (k & 0x3FFFF) == 0 and time.time() > deadline - 30.0:
            break

    # Find best group
    best_group = []
    best_c = 0
    for c, xs in groups.items():
        if len(xs) > len(best_group):
            best_group = xs
            best_c = c

    best_count = len(best_group)
    best_M = int(best_c)

    # Phase 2: Smooth number search for k > max_k_processed
    def sigma_individual(k):
        """Compute sigma(k) by trial division"""
        if k <= max_k_processed and sigma[k] > 0:
            return int(sigma[k])
        n = k
        result = 1
        d = 2
        while d * d <= n:
            if n % d == 0:
                power = 0
                while n % d == 0:
                    n //= d
                    power += 1
                result *= (d**(power+1) - 1) // (d - 1)
            d += 1 if d == 2 else 2
        if n > 1:
            result *= (n * n + n)
        return result

    if time.time() < deadline - 25.0:
        small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]

        def gen_smooth(lo, hi, primes_list, max_depth=8):
            if max_depth == 0:
                if lo <= 1 <= hi:
                    yield 1
                return
            p = primes_list[0]
            rest = primes_list[1:]
            power = 1
            while power <= hi:
                for sub in gen_smooth(max(lo // power, 1), hi // power, rest, max_depth - 1):
                    val = power * sub
                    if lo <= val <= hi:
                        yield val
                power *= p
                if power > hi:
                    break

        lo = max_k_processed + 1
        hi = min(30000000, 10**9)

        count_checked = 0
        for k in gen_smooth(lo, hi, small_primes, 8):
            if time.time() > deadline - 20.0:
                break
            s = sigma_individual(k)
            c = k * s
            if c not in groups:
                groups[c] = []
            groups[c].append(k)
            count_checked += 1
            if count_checked % 50000 == 0 and time.time() > deadline - 20.0:
                break

        # Re-find best group
        for c, xs in groups.items():
            if len(xs) > len(best_group):
                best_group = xs
                best_c = c
                best_count = len(xs)
                best_M = int(c)

    # Phase 3: Targeted divisor search for top M values
    if time.time() < deadline - 12.0:
        top_M_values = []
        for c, xs in groups.items():
            if len(xs) >= 3:
                top_M_values.append((len(xs), c))
        top_M_values.sort(reverse=True)

        for _, M_val in top_M_values[:30]:
            if time.time() > deadline - 10.0:
                break
            if len(groups[M_val]) < 4:
                continue

            sqrt_Mv = math.isqrt(M_val)
            existing = set(groups[M_val])
            new_ks = []
            for d in range(1, sqrt_Mv + 1):
                if M_val % d == 0:
                    if d not in existing:
                        s = sigma_individual(d)
                        if d * s == M_val:
                            new_ks.append(d)
                    other = M_val // d
                    if other != d and other <= sqrt_Mv and other not in existing:
                        s2 = sigma_individual(other)
                        if other * s2 == M_val:
                            new_ks.append(other)

            if new_ks:
                groups[M_val].extend(new_ks)
                if len(groups[M_val]) > len(best_group):
                    best_group = groups[M_val]
                    best_c = M_val
                    best_count = len(best_group)
                    best_M = int(M_val)

    # Verify and return
    if best_count >= 3:
        sqrt_M = math.isqrt(best_M)
        valid_ks = [int(x) for x in best_group if x <= sqrt_M]
        if len(valid_ks) >= 3:
            return {'M': best_M, 'k': valid_ks}

    # Fallback
    return {'M': 333312, 'k': [336, 372, 434]}
