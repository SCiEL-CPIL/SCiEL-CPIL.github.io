# EVOLVE-BLOCK-START
import time
from math import isqrt

LO = 210
MAXV = 10210
MAX_SIZE = 100
TARGET = 210  # 2*3*5*7


def _is_square(n):
    if n < 0:
        return False
    r = isqrt(n)
    return r * r == n


def _sieve_sfp(limit):
    """Squarefree part for 1..limit via a smallest-prime-factor sieve."""
    spf = list(range(limit + 1))
    for i in range(2, int(limit ** 0.5) + 1):
        if spf[i] == i:
            for j in range(i * i, limit + 1, i):
                if spf[j] == j:
                    spf[j] = i
    sfp = [0] * (limit + 1)
    sfp[1] = 1
    for x in range(2, limit + 1):
        v = x
        r = 1
        while v > 1:
            p = spf[v]
            cnt = 0
            while v % p == 0:
                v //= p
                cnt ^= 1
            if cnt:
                r *= p
        sfp[x] = r
    return sfp


def _sfp_of(n):
    r = 1
    p = 2
    while p * p <= n:
        cnt = 0
        while n % p == 0:
            n //= p
            cnt ^= 1
        if cnt:
            r *= p
        p += 1
    if n > 1:
        r *= n
    return r


def _verify(vals):
    if not vals:
        return False
    if len(set(vals)) != len(vals):
        return False
    if len(vals) > MAX_SIZE:
        return False
    for v in vals:
        if not isinstance(v, int) or isinstance(v, bool):
            return False
        if v <= LO or v > MAXV:
            return False
    prod = LO
    for v in vals:
        prod *= v
    return _is_square(prod)


def construct():
    t0 = time.time()
    deadline = t0 + 82.0

    sfp = _sieve_sfp(MAXV)
    sfp_target = sfp[TARGET]  # 210

    # ---- Fast direct search: single, then pairs, then triples ----
    # Single
    for x in range(LO + 1, MAXV + 1):
        if sfp[x] == sfp_target and _verify([x]):
            return [x]

    # Pairs: sfp(sfp[x]*sfp[y]) == target  <=>  sfp[x]*sfp[y] == target * g^2, g=gcd
    for i in range(LO + 1, MAXV + 1):
        if time.time() > deadline - 3:
            break
        a = sfp[i]
        for j in range(i + 1, MAXV + 1):
            b = sfp[j]
            g = math_gcd(a, b)
            if a * b == sfp_target * g * g and _verify([i, j]):
                return [i, j]

    # Triples: sfp(sfp[x]*sfp[y]*sfp[z]) == target
    for i in range(LO + 1, MAXV + 1):
        if time.time() > deadline - 3:
            break
        a = sfp[i]
        for j in range(i + 1, MAXV + 1):
            b = sfp[j]
            ab = _sfp_of(a * b)
            for k in range(j + 1, MAXV + 1):
                c = sfp[k]
                if _sfp_of(ab * c) == sfp_target and _verify([i, j, k]):
                    return [i, j, k]

    # ---- GF(2) Gaussian-elimination sweep for guaranteed minimal max ----
    # Map each prime to a bit; parity vector of n = set of primes with odd exponent.
    # target_vec = parity of 210.
    primes = []
    prime_seen = set()
    for x in range(LO + 1, MAXV + 1):
        v = x
        p = 2
        while p * p <= v:
            cnt = 0
            while v % p == 0:
                v //= p
                cnt ^= 1
            if cnt and p not in prime_seen:
                prime_seen.add(p)
                primes.append(p)
            p += 1
        if v > 1 and v not in prime_seen:
            prime_seen.add(v)
            primes.append(v)
    for p in (2, 3, 5, 7):
        if p not in prime_seen:
            prime_seen.add(p)
            primes.append(p)
    prime_index = {p: i for i, p in enumerate(primes)}
    D = len(primes)

    # parity bitvector for each value
    parity = {}
    for x in range(LO + 1, MAXV + 1):
        bits = 0
        v = x
        p = 2
        while p * p <= v:
            cnt = 0
            while v % p == 0:
                v //= p
                cnt ^= 1
            if cnt:
                bits |= (1 << prime_index[p])
            p += 1
        if v > 1:
            bits |= (1 << prime_index[v])
        parity[x] = bits

    target_bits = 0
    for p in (2, 3, 5, 7):
        target_bits |= (1 << prime_index[p])

    # Incremental basis over GF(2) with back-substitution data.
    pivot_to_row = {}   # pivot bit -> index into rows
    rows = []           # each: [bits, list_of_source_indices]
    row_to_pivot = {}   # row index -> pivot bit

    def add_row(bits, src):
        nonlocal rows
        b = bits
        # eliminate against existing basis (high pivot to low)
        for pbit, ridx in sorted(pivot_to_row.items(), reverse=True):
            if (b >> pbit) & 1:
                rb = rows[ridx][0]
                b ^= rb
                lst = rows[ridx][1]
                src = list(src)
                for s in lst:
                    src.remove(s) if s in src else None
        # find new pivot
        pivot = -1
        for j in range(D - 1, -1, -1):
            if (b >> j) & 1:
                pivot = j
                break
        if pivot < 0 or not src:
            return None
        rows.append([b, src])
        ridx = len(rows) - 1
        pivot_to_row[pivot] = ridx
        row_to_pivot[ridx] = pivot
        # eliminate this new row from existing rows to keep reduced form
        for pbit2, ridx2 in list(pivot_to_row.items()):
            if ridx2 == ridx:
                continue
            if (rows[ridx2][0] >> pivot) & 1:
                rows[ridx2][0] ^= b
                lst2 = rows[ridx2][1]
                for s in src:
                    if s in lst2:
                        lst2.remove(s)
                    else:
                        lst2.append(s)
        return ridx

    def in_span_and_extract(target):
        # check membership and recover a subset
        t = target
        for pbit, ridx in sorted(pivot_to_row.items(), reverse=True):
            if (t >> pbit) & 1:
                t ^= rows[ridx][0]
                # accumulate
        if t != 0:
            return None
        # rebuild: we need the combination of rows whose XOR == target.
        # Do a clean back-substitution collecting row indices.
        t = target
        chosen_rows = []
        for pbit, ridx in sorted(pivot_to_row.items(), reverse=True):
            if (t >> pbit) & 1:
                t ^= rows[ridx][0]
                chosen_rows.append(ridx)
        if t != 0:
            return None
        srcs = []
        for ridx in chosen_rows:
            for s in rows[ridx][1]:
                if s not in srcs:
                    srcs.append(s)
        return srcs

    # Sweep M from LO+1 upward
    for M in range(LO + 1, MAXV + 1):
        if time.time() > deadline:
            break
        add_row(parity[M], [M])
        srcs = in_span_and_extract(target_bits)
        if srcs is None:
            continue
        # cap to <= MAX_SIZE: drop from the end (largest values) while keeping span valid
        # Simplest: if too many, just take a subset that still works via re-check.
        # We'll try to reduce: remove largest elements while product stays square.
        srcs = sorted(srcs)
        while len(srcs) > MAX_SIZE:
            # remove the largest element that keeps the condition
            removed = False
            for idx in range(len(srcs) - 1, -1, -1):
                cand = srcs[:idx] + srcs[idx + 1:]
                if _verify(cand):
                    srcs = cand
                    removed = True
                    break
            if not removed:
                # cannot reduce; take first MAX_SIZE and hope (shouldn't happen for small M)
                srcs = srcs[:MAX_SIZE]
                break
        if _verify(srcs):
            return [int(v) for v in srcs]

    # Fallback: known valid single answer
    return [840]


def math_gcd(a, b):
    while b:
        a, b = b, a % b
    return a
# EVOLVE-BLOCK-END
