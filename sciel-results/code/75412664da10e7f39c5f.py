import time
import math
from collections import defaultdict
import numpy as np


def construct():
    start = time.time()
    deadline = start + 110.0  # Safety margin before 150s kill

    # Fallback known solution: 336*sigma(336) = 372*sigma(372) = 434*sigma(434) = 333312
    best_M = 333312
    best_k = [336, 372, 434]
    best_count = 3

    # ---------- helpers ----------
    def factorize(n):
        """Factorize n via trial division. Returns list of (p, exp)."""
        if n == 1:
            return []
        factors = []
        d = 2
        while d * d <= n:
            if n % d == 0:
                cnt = 0
                while n % d == 0:
                    n //= d
                    cnt += 1
                factors.append((d, cnt))
            d += 1 if d == 2 else 2
        if n > 1:
            factors.append((n, 1))
        return factors

    def sigma_from_factors(factors):
        """Compute sigma from factor list [(p, exp), ...]."""
        result = 1
        for p, e in factors:
            result *= (p ** (e + 1) - 1) // (p - 1)
        return result

    def sigma_exact(n):
        if n == 1:
            return 1
        return sigma_from_factors(factorize(n))

    def gen_divisors(factors):
        """Generate all divisors from factor list."""
        divs = [1]
        for p, e in factors:
            new_divs = []
            for d in divs:
                pe = 1
                for _ in range(e + 1):
                    new_divs.append(d * pe)
                    pe *= p
            divs = new_divs
        return divs

    # ---------- Phase 1: NumPy-vectorized sigma sieve, group by M ----------
    # Try progressively larger N until time runs low
    sieve_Ns = [5_000_000, 10_000_000, 20_000_000, 40_000_000]
    for N in sieve_Ns:
        if time.time() > deadline - 30:
            break
        try:
            # NumPy-vectorized sigma computation
            sigma_arr = np.zeros(N + 1, dtype=np.int64)
            for i in range(1, N + 1):
                sigma_arr[i::i] += i

            if time.time() > deadline - 20:
                del sigma_arr
                break

            # Compute M = k * sigma[k] for all k, find collisions
            # Use a dict to group k by M value
            m_to_k = defaultdict(list)
            sigma_list = sigma_arr.tolist()
            for k in range(2, N + 1):
                M_val = k * sigma_list[k]
                if M_val <= 10**18:
                    m_to_k[M_val].append(k)

            del sigma_arr
            del sigma_list

            for M_val, ks in m_to_k.items():
                if len(ks) > best_count:
                    best_count = len(ks)
                    best_M = M_val
                    best_k = ks[:]

            del m_to_k

        except MemoryError:
            break
        except Exception:
            break

    # ---------- Phase 2: Polish pass on current best_M via full divisor enumeration ----------
    if time.time() < deadline - 15 and best_M is not None:
        try:
            factors_M = factorize(best_M)
            divisors = gen_divisors(factors_M)
            isqrt_M = math.isqrt(best_M)
            valid_k = []
            for k in divisors:
                if 1 < k <= isqrt_M:
                    try:
                        if k * sigma_exact(k) == best_M:
                            valid_k.append(k)
                    except Exception:
                        continue
            if len(valid_k) > best_count:
                best_count = len(valid_k)
                best_k = valid_k
        except Exception:
            pass

    # ---------- Phase 3: Meet-in-the-middle using multiplicative structure ----------
    # Decompose k = a * b where gcd(a,b)=1, sigma(k) = sigma(a)*sigma(b)
    # M = a*sigma(a) * b*sigma(b)
    if time.time() < deadline - 20:
        try:
            # Generate 2,3-smooth numbers up to limit
            max_comp = 5_000_000
            smooth_23 = set()
            a = 1
            while a <= max_comp:
                b = 1
                while a * b <= max_comp:
                    smooth_23.add(a * b)
                    b *= 3
                a *= 2
            smooth_23 = list(smooth_23)

            # Generate 5,7-smooth numbers up to limit
            smooth_57 = set()
            c = 1
            while c <= max_comp:
                d = 1
                while c * d <= max_comp:
                    smooth_57.add(c * d)
                    d *= 7
                c *= 5
            smooth_57 = list(smooth_57)

            # Precompute sigma and M contribution for smooth components
            def compute_m_contrib(nums):
                contrib = {}
                for n in nums:
                    s = sigma_exact(n)
                    contrib[n] = n * s
                return contrib

            m_contrib_23 = compute_m_contrib(smooth_23)
            m_contrib_57 = compute_m_contrib(smooth_57)

            # Group m_contrib_57 by M value
            m57_to_bs = defaultdict(list)
            for b in smooth_57:
                m57_to_bs[m_contrib_57[b]].append(b)

            # Sort smooth_23 by M contribution for early pruning
            smooth_23_sorted = sorted(smooth_23, key=lambda x: m_contrib_23[x])

            m_to_ks = defaultdict(set)
            limit_k = 10**14

            for a in smooth_23_sorted:
                if time.time() > deadline - 8:
                    break
                ma = m_contrib_23[a]
                for m57, bs in m57_to_bs.items():
                    M_val = ma * m57
                    if M_val > 10**18:
                        continue
                    for b in bs:
                        if a * b > limit_k:
                            continue
                        if math.gcd(a, b) == 1:
                            k = a * b
                            if k <= math.isqrt(M_val):
                                m_to_ks[M_val].add(k)

            for M_val, ks in m_to_ks.items():
                if len(ks) > best_count:
                    best_count = len(ks)
                    best_M = M_val
                    best_k = list(ks)

        except Exception:
            pass

    # ---------- Final verification ----------
    if best_M is not None and best_k:
        isqrt_M = math.isqrt(best_M)
        valid_k = []
        for k in best_k:
            if not isinstance(k, int) or isinstance(k, bool):
                continue
            if k < 1 or k > isqrt_M:
                continue
            try:
                if k * sigma_exact(k) == best_M:
                    valid_k.append(k)
            except Exception:
                continue
        valid_k = list(set(valid_k))
        if len(valid_k) >= 2:
            best_k = valid_k
            best_count = len(valid_k)
        else:
            best_M = 333312
            best_k = [336, 372, 434]
            best_count = 3

    best_M = int(best_M)
    best_k = [int(k) for k in best_k]

    return {'M': best_M, 'k': best_k}
