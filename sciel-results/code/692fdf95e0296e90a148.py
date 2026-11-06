"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from math import isqrt

# Cache for is_powerful
_powerful_cache = {}

def is_powerful(n):
    """Check if n is powerful: every prime factor appears with exponent >= 2.
    1 is powerful. Uses trial division up to cube root of n."""
    if n in _powerful_cache:
        return _powerful_cache[n]
    if n == 1:
        _powerful_cache[n] = True
        return True
    # Factor out small primes
    temp = n
    for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]:
        if temp % (p * p) == 0:
            # p^2 divides temp, so p appears with exponent >= 2
            while temp % p == 0:
                temp //= p
        elif temp % p == 0:
            # p appears with exponent 1 -> not powerful
            _powerful_cache[n] = False
            return False
    # After removing small primes, check remaining temp
    if temp == 1:
        _powerful_cache[n] = True
        return True
    # temp must be a prime with exponent >= 2, or product of primes each with exp >= 2
    # Since we removed all primes up to 97, if temp > 1, it's either:
    # - a prime > 97 (then temp must be a perfect square of that prime, i.e., temp = q^2 for prime q)
    # - or a product of such primes
    # Check if temp is a perfect square
    r = isqrt(temp)
    if r * r == temp:
        # temp = r^2; need r to be powerful? No, we need each prime in r to appear with exp >= 2 in temp.
        # In temp = r^2, each prime in r appears with exp 2*exp_in_r. We need 2*exp_in_r >= 2, i.e., exp_in_r >= 1.
        # So we need r to have no prime factor with exponent 0, which is always true.
        # But wait: we need to verify that r itself doesn't have a prime with exponent 1 that would make temp have exponent 2 (which is fine).
        # Actually, if temp = r^2 and r is any integer, then every prime factor of temp has even exponent >= 2.
        # So temp is powerful iff temp is a perfect square? No!
        # Counterexample: temp = 12 = 2^2 * 3^1. 12 is not a perfect square. But 12 is not powerful because 3 has exp 1.
        # If temp is a perfect square, say temp = s^2, then every prime factor of temp has exponent 2*exp_in_s >= 2.
        # So yes, if temp is a perfect square, it is powerful.
        # But we also need to check: could temp be powerful but not a perfect square?
        # If temp is powerful, all exponents are >= 2. If all exponents are even, temp is a perfect square.
        # If some exponent is odd (but >= 2, e.g., 3, 5, ...), then temp is not a perfect square but still powerful.
        # Example: temp = 8 = 2^3 is powerful but not a perfect square.
        # So we need a more careful check.
        # Since we've removed all primes <= 97, the remaining temp has prime factors > 97.
        # For temp to be powerful, each prime factor must have exponent >= 2.
        # Let's factor temp by trial division up to sqrt(temp) but only need to check primes > 97.
        # However, temp could be large. Let's do:
        # If temp is a perfect square, it's powerful (as argued).
        # If not, check if temp = p^k for prime p > 97 and k >= 2, or product of such.
        # Simpler: do trial division on temp by integers from 101 up to sqrt(temp).
        # But temp could be up to 10^18, sqrt up to 10^9, which is too slow.
        # Alternative: since we only removed primes <= 97, and temp has no small prime factors,
        # if temp is powerful, it must be of the form p1^e1 * p2^e2 * ... where each pi > 97 and ei >= 2.
        # The smallest such prime is 101, so 101^2 = 10201. If temp < 10201, then temp must be 1 (but temp > 1 here) or a square of a prime > 97.
        # Actually, if temp < 101^2 = 10201 and temp > 1, and temp has no prime factor <= 97, then temp itself must be a prime > 97.
        # But a prime > 97 has exponent 1 in temp, so temp is not powerful.
        # So for temp < 10201: not powerful (unless temp=1, but temp>1).
        # For temp >= 10201: we need to check.
        # Let's try: is temp a perfect power?
        # Check if temp = m^k for k >= 2.
        max_exp = int(np.log2(temp)) + 1
        found = False
        for k in range(2, max_exp + 1):
            # Check if temp is a perfect k-th power
            r = int(round(temp ** (1.0 / k)))
            # Check r, r-1, r+1
            for cand in [r - 1, r, r + 1]:
                if cand >= 1 and cand ** k == temp:
                    # temp = cand^k; powerful if cand is powerful? No.
                    # temp = cand^k. Each prime factor of cand has exponent e in cand, so 2e in temp.
                    # We need 2e >= 2, i.e., e >= 1. Always true.
                    # But wait: we need to verify that cand itself doesn't introduce issues.
                    # Actually, if temp = cand^k with k >= 2, then every prime factor of temp has exponent k * e_p where e_p >= 1.
                    # So exponent >= k >= 2. Thus temp is powerful.
                    # But this is only valid if cand is an integer and the factorization is exact.
                    # However, we also need to ensure that no prime factor appears with exponent < 2 in temp.
                    # Since k >= 2 and e_p >= 1, k*e_p >= 2. So yes.
                    # But: what if cand has a prime factor with e_p = 0? That's impossible.
                    # So if temp is a perfect k-th power for some k >= 2, temp is powerful.
                    _powerful_cache[n] = True
                    return True
            found = True
            break
        if not found:
            # temp is not a perfect power. For temp to be powerful, it must be a product of primes each with exponent >= 2.
            # But if it's not a perfect power, it could still be powerful, e.g., 2^2 * 3^3 = 108.
            # 108 is not a perfect power (108 = 2^2 * 3^3, gcd(2,3)=1, so not a perfect power).
            # So we need to factor temp.
            # Since temp has no prime factors <= 97, we do trial division from 101.
            # But this could be slow. However, in practice, for the numbers we encounter,
            # temp will often be 1 or a small number.
            # Let's do trial division up to min(isqrt(temp), 10**6) to keep it fast.
            # If we can't fully factor, assume not powerful (conservative).
            t = temp
            for p in range(101, min(isqrt(t) + 1, 10**6)):
                if t % p == 0:
                    exp = 0
                    while t % p == 0:
                        t //= p
                        exp += 1
                    if exp < 2:
                        _powerful_cache[n] = False
                        return False
            if t > 1:
                # t is a prime > 10^6 or t = 1
                # If t > 1, it's a prime factor with exponent 1 (since we divided out all p <= 10^6)
                # So not powerful
                _powerful_cache[n] = False
                return False
            _powerful_cache[n] = True
            return True
        else:
            # We found a perfect power, already returned
            pass
    else:
        # temp is not a perfect square; could still be powerful (e.g., 8 = 2^3)
        # But we already removed all primes <= 97, so temp has no small prime factors.
        # Check if temp is a perfect power
        max_exp = int(np.log2(temp)) + 1
        for k in range(2, max_exp + 1):
            r = int(round(temp ** (1.0 / k)))
            for cand in [r - 1, r, r + 1]:
                if cand >= 1 and cand ** k == temp:
                    _powerful_cache[n] = True
                    return True
        # Not a perfect power. Do trial division
        t = temp
        for p in range(101, min(isqrt(t) + 1, 10**6)):
            if t % p == 0:
                exp = 0
                while t % p == 0:
                    t //= p
                    exp += 1
                if exp < 2:
                    _powerful_cache[n] = False
                    return False
        if t > 1:
            _powerful_cache[n] = False
            return False
        _powerful_cache[n] = True
        return True

def generate_powerful_numbers(limit):
    """Generate all powerful numbers <= limit using a^2 * b^3 form."""
    if limit < 1:
        return []
    powerfuls = set()
    max_b = int(limit ** (1/3)) + 2
    for b in range(1, max_b + 1):
        b3 = b * b * b
        if b3 > limit:
            break
        max_a = isqrt(limit // b3)
        for a in range(1, max_a + 1):
            n = a * a * b3
            if n <= limit:
                powerfuls.add(n)
    return sorted(powerfuls)

def construct():
    start_time = time.time()
    deadline = start_time + 108  # safety margin vs 150s wall-clock limit

    # ---- Phase 1: find candidate Ns using small powerful list ----
    L1 = 10**8
    pow_list_1 = generate_powerful_numbers(L1)
    n1 = len(pow_list_1)
    
    if n1 == 0:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    P1 = np.array(pow_list_1, dtype=np.int64)
    i_idx, j_idx = np.triu_indices(n1)
    sums = P1[i_idx] + P1[j_idx]
    max_sum = int(sums.max()) if n1 else 0
    counts = np.bincount(sums, minlength=max_sum + 1)
    if counts.size > 0:
        counts[0] = 0
    
    topk = min(200, counts.size)
    order = np.argsort(-counts, kind='stable')
    candidates = [int(x) for x in order[:topk] if counts[x] > 0]
    
    # ---- Phase 2: for each candidate, count pairs using is_powerful ----
    best_N = 0
    best_pairs = []
    
    for N in candidates:
        if time.time() > deadline - 10:
            break
        if N < 2 or N > 10**18:
            continue
        pairs = []
        half = N // 2
        # Iterate a from 1 to half
        # Use step to skip non-powerful candidates? No, just iterate.
        # But 10^18 / 2 = 5*10^17 iterations is too many.
        # Instead, iterate over powerful numbers a <= half and check if N-a is powerful.
        # But we don't have a list of powerful numbers up to half.
        # Alternative: for a in range(1, half+1), check is_powerful(a) and is_powerful(N-a).
        # This is O(N) which is too slow for N ~ 10^18.
        
        # Better: generate powerful numbers up to half on the fly? Too slow.
        # Use the fact that powerful numbers are sparse. Density ~ sqrt(x)/x = 1/sqrt(x).
        # So up to x, there are ~ sqrt(x) powerful numbers.
        # For x = 10^18, sqrt(x) = 10^9, still too many.
        
        # Alternative: iterate a over powerful numbers generated up to some limit, and check N-a.
        # But we need a <= N/2, and N could be up to 10^18.
        
        # Let's change strategy: for each candidate N, generate powerful numbers up to N/2
        # and check if N-a is powerful. But generating powerful numbers up to 10^18 is infeasible.
        
        # Revised: Use the small powerful list to get candidate pairs, then verify.
        # For a in pow_list_1: if a <= N/2, check if N-a is powerful.
        # This misses pairs where a > L1. But for the top candidates from Phase 1,
        # most pairs will have both a and b <= L1 (since they were counted in Phase 1).
        # For additional pairs with a > L1, we'd need to search more.
        
        # Let's do: iterate a over pow_list_1 (up to L1), and also try a few values near N/2.
        for a in pow_list_1:
            if a > half:
                break
            b = N - a
            if b >= a and is_powerful(b):
                pairs.append([a, b])
        
        # Also check if N/2 is powerful (for a = b)
        if N % 2 == 0:
            a = N // 2
            if a not in [p[0] for p in pairs] and is_powerful(a):
                pairs.append([a, a])
        
        if len(pairs) > len(best_pairs):
            best_N = N
            best_pairs = pairs
    
    # ---- Fallback ----
    if not best_pairs:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    if best_N < 2 or best_N > 10**18:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    return {'N': best_N, 'pairs': best_pairs}
# EVOLVE-BLOCK-END
