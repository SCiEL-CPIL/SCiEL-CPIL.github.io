import math
import time
from math import isqrt
from collections import defaultdict

# ---------- exact integer arithmetic helpers ----------

def _factor(n):
    """Trial-division factorization for n up to ~1e18 (fast for smooth n)."""
    factors = {}
    if n <= 1:
        return factors
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,
              53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109,
              113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179,
              181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241,
              251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313,
              317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389,
              397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461,
              463, 467, 479, 487, 491, 499):
        if p * p > n:
            break
        if n % p == 0:
            cnt = 0
            while n % p == 0:
                n //= p
                cnt += 1
            factors[p] = cnt
    if n > 1:
        factors[n] = 1
    return factors


def _sigma(n):
    """Exact sigma(n) via factorization."""
    if n <= 0:
        return 0
    if n == 1:
        return 1
    f = _factor(n)
    result = 1
    for p, e in f.items():
        # sum_{i=0}^{e} p^i
        result *= (p ** (e + 1) - 1) // (p - 1)
    return result


def _verify(k, M):
    """Check k*sigma(k)==M and k<=isqrt(M)."""
    if k < 1 or k > isqrt(M):
        return False
    s = _sigma(k)
    return k * s == M


# ---------- sigma via direct divisor sum (for small k) ----------
def _sigma_small(n, limit=200000):
    """sigma(n) by iterating divisors up to sqrt; fast for n < ~1e6."""
    if n <= 0:
        return 0
    if n == 1:
        return 1
    total = 1
    r = isqrt(n)
    for d in range(2, r + 1):
        if n % d == 0:
            total += d
            if d != n // d:
                total += n // d
    total += n
    return total


# ---------- generate highly composite / divisor-rich candidates ----------
def _gen_candidates(max_k, primes):
    """Generate k values with many divisors (smooth numbers over small primes)."""
    cands = set()
    cands.add(1)

    def _dfs(idx, cur, max_val):
        if cur > max_val:
            return
        if idx >= len(primes):
            if 1 < cur <= max_k:
                cands.add(cur)
            return
        p = primes[idx]
        v = cur
        while v <= max_val:
            _dfs(idx + 1, v, max_val)
            if v > max_val // p:
                break
            v *= p

    _dfs(0, 1, max_k)
    return cands


def construct():
    t0 = time.time()
    DEADLINE = 130  # safety margin before 150s kill

    # Prime list for generating smooth numbers
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,
                    53, 59, 61, 67, 71, 73, 79, 83, 89, 97]

    # We'll do multiple passes with increasing max_k
    # Pass 1: small k (fast)
    # Pass 2: medium k
    # Pass 3: large k (if time permits)

    best_M = 333312
    best_k_list = [336, 372, 434]
    best_count = 3

    # Group: M -> list of k
    groups = defaultdict(list)

    # Pre-seed with known solutions
    for k in [336, 372, 434]:
        groups[333312].append(k)

    # Pass 1: k up to 10000
    max_k1 = 10000
    cands1 = _gen_candidates(max_k1, small_primes[:10])
    for k in sorted(cands1):
        if time.time() - t0 > DEADLINE - 30:
            break
        if k == 1:
            continue
        s = _sigma_small(k)
        M = k * s
        if M <= 10**18 and k <= isqrt(M):
            if k not in groups[M]:
                groups[M].append(k)

    # Pass 2: k up to 100000 (more candidates)
    if time.time() - t0 < DEADLINE - 20:
        max_k2 = 100000
        cands2 = _gen_candidates(max_k2, small_primes[:12])
        for k in sorted(cands2):
            if time.time() - t0 > DEADLINE - 20:
                break
            if k == 1 or k in cands1:
                continue
            s = _sigma_small(k)
            M = k * s
            if M <= 10**18 and k <= isqrt(M):
                if k not in groups[M]:
                    groups[M].append(k)

    # Pass 3: k up to 1000000 (if time)
    if time.time() - t0 < DEADLINE - 10:
        max_k3 = 1000000
        cands3 = _gen_candidates(max_k3, small_primes[:14])
        for k in sorted(cands3):
            if time.time() - t0 > DEADLINE - 10:
                break
            if k == 1 or k in cands1 or k in cands2:
                continue
            s = _sigma_small(k)
            M = k * s
            if M <= 10**18 and k <= isqrt(M):
                if k not in groups[M]:
                    groups[M].append(k)

    # Pass 4: targeted search — look for k with specific structure
    # k = 2^a * 3^b * 5^c * 7^d with various exponents
    if time.time() - t0 < DEADLINE - 5:
        seen_k = set(groups.keys() and [k for ks in groups.values() for k in ks])
        # Actually build a set of all k seen
        all_k_seen = set()
        for ks in groups.values():
            for k in ks:
                all_k_seen.add(k)

        # Generate more structured candidates
        for a in range(0, 20):
            if time.time() - t0 > DEADLINE - 5:
                break
            for b in range(0, 15):
                if time.time() - t0 > DEADLINE - 5:
                    break
                k_ab = (2 ** a) * (3 ** b)
                if k_ab > 10**7:
                    break
                for c in range(0, 12):
                    if time.time() - t0 > DEADLINE - 5:
                        break
                    k_abc = k_ab * (5 ** c)
                    if k_abc > 10**7:
                        break
                    for d in range(0, 10):
                        if time.time() - t0 > DEADLINE - 5:
                            break
                        k = k_abc * (7 ** d)
                        if k > 10**7 or k == 1 or k in all_k_seen:
                            if k > 10**7:
                                break
                            continue
                        s = _sigma_small(k)
                        M = k * s
                        if M <= 10**18 and k <= isqrt(M):
                            if k not in groups[M]:
                                groups[M].append(k)
                                all_k_seen.add(k)
                        elif k > 10**7:
                            break

    # Find the best M
    for M, ks in groups.items():
        if len(ks) > best_count:
            # Verify all solutions
            valid = []
            for k in ks:
                if _verify(k, M):
                    valid.append(k)
            if len(valid) > best_count:
                best_M = M
                best_k_list = valid
                best_count = len(valid)

    # Final verification of best result
    final_k = []
    for k in best_k_list:
        if _verify(k, best_M):
            final_k.append(k)

    if not final_k:
        # Fallback
        return {'M': 333312, 'k': [336, 372, 434]}

    # Ensure types are exact Python int
    M_out = int(best_M)
    k_out = [int(k) for k in sorted(set(final_k))]

    # Final safety check
    if len(k_out) == 0 or M_out < 1 or M_out > 10**18:
        return {'M': 333312, 'k': [336, 372, 434]}

    return {'M': M_out, 'k': k_out}
