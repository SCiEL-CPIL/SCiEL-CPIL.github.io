"""Square-product extension for 210: find distinct ints in (210,10210]
whose product * 210 is a perfect square, minimizing max - 210."""

import math
import time
from collections import defaultdict

N = 210
MAX_N = 10210
MAX_K = 100
SMALL_PRIMES = [2, 3, 5, 7]


def _is_square(n):
    r = math.isqrt(n)
    return r * r == n


def _sieve_primes(limit):
    sieve = [True] * (limit + 1)
    if limit >= 0:
        sieve[0] = False
    if limit >= 1:
        sieve[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            for j in range(i * i, limit + 1, i):
                sieve[j] = False
    return [i for i in range(2, limit + 1) if sieve[i]]


def _factorize(n, primes):
    d = {}
    for p in primes:
        if p * p > n:
            break
        if n % p == 0:
            e = 0
            while n % p == 0:
                n //= p
                e += 1
            d[p] = e
    if n > 1:
        d[n] = 1
    return d


def construct():
    t_start = time.time()
    DEADLINE = t_start + 75.0  # safety margin before 90s kill

    primes = _sieve_primes(MAX_N)

    # Precompute parity for all candidates
    # Parity vector: (small_mask, frozenset_of_large_odd_primes)
    parity = {}
    for n in range(211, MAX_N + 1):
        d = _factorize(n, primes)
        mask = 0
        for i, p in enumerate(SMALL_PRIMES):
            if d.get(p, 0) & 1:
                mask |= (1 << i)
        large = frozenset(p for p, e in d.items() if p not in SMALL_PRIMES and e & 1)
        parity[n] = (mask, large)

    TARGET_MASK = 0b1111  # 210 = 2*3*5*7, all odd
    TARGET_LARGE = frozenset()
    TARGET_VECTOR = (TARGET_MASK, TARGET_LARGE)

    # Phase 1: Quick check for 1-element solutions
    for n in range(211, MAX_N + 1):
        if time.time() > DEADLINE:
            break
        if parity[n] == TARGET_VECTOR:
            return [n]

    # Phase 2: Search for 2-element solutions with small max
    # Build reverse map: vec -> list of n with that vec (sorted)
    vec_to_ns = defaultdict(list)
    for n in range(211, MAX_N + 1):
        vec_to_ns[parity[n]].append(n)

    best_pair = None
    for a in range(211, MAX_N + 1):
        if time.time() > DEADLINE:
            break
        va = parity[a]
        req_mask = TARGET_MASK ^ va[0]
        req_large = TARGET_LARGE.symmetric_difference(va[1])
        req_vec = (req_mask, req_large)
        
        if req_vec in vec_to_ns:
            for b in vec_to_ns[req_vec]:
                if b > a:
                    if best_pair is None or max(a, b) < max(best_pair):
                        best_pair = (a, b)
                    break

    if best_pair is not None:
        return list(best_pair)

    # Phase 3: Search for 3-element solutions with incremental approach
    single_dict = {}
    pair_dict = {}
    
    for limit in range(211, MAX_N + 1):
        if time.time() > DEADLINE:
            break
        
        v = parity[limit]
        
        req_mask = TARGET_MASK ^ v[0]
        req_large = TARGET_LARGE.symmetric_difference(v[1])
        req = (req_mask, req_large)
        
        if req in pair_dict:
            n1, n2 = pair_dict[req]
            return [n1, n2, limit]
        
        if v not in single_dict:
            single_dict[v] = limit
        
        for prev_n in range(211, limit):
            v_prev = parity[prev_n]
            m_xor = v_prev[0] ^ v[0]
            if v_prev[1] == v[1]:
                lg_xor = frozenset()
            else:
                lg_xor = v_prev[1].symmetric_difference(v[1])
            pair_key = (m_xor, lg_xor)
            if pair_key not in pair_dict:
                pair_dict[pair_key] = (prev_n, limit)

    # Fallback: shouldn't happen, but return something valid
    # 210 * 840 = 210 * 210 * 4 = (210*2)^2 = 420^2
    return [840]


if __name__ == "__main__":
    t0 = time.time()
    result = construct()
    t1 = time.time()
    print(f"Result: {result}")
    print(f"Max - 210: {max(result) - 210}")
    prod = 210
    for x in result:
        prod *= x
    print(f"Product is square: {_is_square(prod)}")
    print(f"Time: {t1 - t0:.2f}s")
