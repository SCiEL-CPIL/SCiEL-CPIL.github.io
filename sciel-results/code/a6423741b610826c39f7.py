import math
import time
import bisect

P = {'n': 210, 'maximum_window': 10000, 'maximum_size': 100}

def _primes_upto(n):
    if n < 2:
        return []
    sieve = bytearray([1]) * (n + 1)
    sieve[0:2] = b'\x00\x00'
    import math
    for i in range(2, int(math.isqrt(n)) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    return [i for i in range(2, n+1) if sieve[i]]

def _parity_vectors(maxM):
    # returns list parity[i] for n in 211..maxM as int bitsets over primes<=maxM
    primes = _primes_upto(maxM)
    pindex = {p: i for i, p in enumerate(primes)}
    vecs = []
    for n in range(211, maxM+1):
        x = n
        v = 0
        for i, p in enumerate(primes):
            if p * p > x:
                break
            c = 0
            while x % p == 0:
                x //= p
                c ^= 1
            if c:
                v |= (1 << i)
        if x > 1:
            j = pindex[x]
            v |= (1 << j)
        vecs.append(v)
    return vecs, primes

def _target_parity(primes):
    t = 210
    v = 0
    pindex = {p: i for i, p in enumerate(primes)}
    for i, p in enumerate(primes):
        c = 0
        while t % p == 0:
            t //= p
            c ^= 1
        if c:
            v |= (1 << i)
    return v

def _span_combo(vectors, target, deadline):
    basis = {}
    for pos, v in enumerate(vectors):
        if time.time() > deadline:
            return None
        vec = v
        combo = 1 << pos
        while vec:
            pivot = vec.bit_length() - 1
            if pivot in basis:
                bvec, bcombo = basis[pivot]
                vec ^= bvec
                combo ^= bcombo
            else:
                basis[pivot] = (vec, combo)
                break
    vec = target
    combo = 0
    while vec:
        pivot = vec.bit_length() - 1
        if pivot in basis:
            bvec, bcombo = basis[pivot]
            vec ^= bvec
            combo ^= bcombo
        else:
            return None
    return combo

def _reduce_size(combo, vectors, target, deadline):
    # vectors indexed by position (0-based over 211..)
    # combo: bitset of positions currently selected, XOR = target
    # try to reduce to <=100 by removing zero-xor subsets
    while combo.bit_count() > 100 and time.time() < deadline:
        # find a non-empty zero combination within selected positions
        sel = [pos for pos in range(combo.bit_length()) if (combo >> pos) & 1]
        # elimination over sel vectors targeting 0
        basis = {}
        found = None
        for pos in sel:
            vec = vectors[pos]
            combo2 = 1 << pos
            while vec:
                pivot = vec.bit_length() - 1
                if pivot in basis:
                    bv, bc = basis[pivot]
                    vec ^= bv
                    combo2 ^= bc
                else:
                    basis[pivot] = (vec, combo2)
                    break
        # now reduce 0: always in span trivially; we need a *non-trivial* zero combo
        # find any non-empty combo XORing to 0: take any two basis? No.
        # Instead: find a dependency: since we have |sel| vectors and rank r < |sel|, there's a dependency.
        # Reconstruct a zero-combination: process all, track; when a vector reduces to 0 with nonzero combo, that's a zero combo.
        basis2 = {}
        zero_combo = None
        for pos in sel:
            vec = vectors[pos]
            combo2 = 1 << pos
            while vec:
                pivot = vec.bit_length() - 1
                if pivot in basis2:
                    bv, bc = basis2[pivot]
                    vec ^= bv
                    combo2 ^= bc
                else:
                    basis2[pivot] = (vec, combo2)
                    break
            if vec == 0 and combo2 != 0:
                zero_combo = combo2
                break
        if zero_combo is None or zero_combo == combo:
            break
        combo ^= zero_combo
    return combo

def construct():
    start = time.time()
    deadline = start + 80  # safety margin under 90s
    maxM = 10210
    # precompute up to maxM once (vectors depend on primes<=maxM; but for smaller M the higher primes just don't appear, bitsets still valid)
    # Actually parity vector of n only uses primes <= n <= maxM, so computing with primes<=maxM is fine for all M.
    vecs, primes = _parity_vectors(maxM)
    target = _target_parity(primes)
    # binary search minimal M in [211, maxM] such that target in span(vecs[0..M-211])
    lo, hi = 211, maxM
    ans = None
    while lo <= hi and time.time() < deadline:
        mid = (lo + hi) // 2
        sub = vecs[:mid-210]
        combo = _span_combo(sub, target, deadline)
        if combo is not None:
            ans = (mid, combo)
            hi = mid - 1
        else:
            lo = mid + 1
    if ans is None:
        # fallback: just return [4*210]
        return [4*P['n']]
    M, combo = ans
    # reduce size
    combo = _reduce_size(combo, vecs, target, deadline)
    selected = [211 + pos for pos in range(combo.bit_length()) if (combo >> pos) & 1]
    # verify
    prod = 210
    for s in selected:
        prod *= s
    r = math.isqrt(prod)
    if r*r != prod or len(selected) > 100 or len(set(selected)) != len(selected) or any(s <= 210 or s > 10210 for s in selected) or len(selected) < 1:
        # fallback to a known valid: [4*210]
        return [4*P['n']]
    return list(selected)
