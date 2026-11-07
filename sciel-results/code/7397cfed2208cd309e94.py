import time
from math import isqrt

def _is_perfect_square(n):
    if n < 0:
        return False
    r = isqrt(n)
    return r * r == n

def construct():
    start = time.time()
    deadline = start + 85.0

    # Compute smallest prime factor up to a reasonable limit
    MAX_N = 5000
    spf = list(range(MAX_N + 1))
    for i in range(2, int(MAX_N ** 0.5) + 1):
        if spf[i] == i:
            for j in range(i * i, MAX_N + 1, i):
                if spf[j] == j:
                    spf[j] = i

    # Collect primes
    primes = [i for i in range(2, MAX_N + 1) if spf[i] == i]
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    nprimes = len(primes)

    def get_parity_bits(n):
        """Return bitmask of primes with odd exponent in n."""
        if n <= 1:
            return 0
        bits = 0
        x = n
        while x > 1:
            p = spf[x]
            cnt = 0
            while x % p == 0:
                x //= p
                cnt ^= 1
            if cnt:
                idx = prime_to_idx.get(p)
                if idx is not None:
                    bits |= (1 << idx)
                else:
                    pass
        return bits

    # Target: parity of 210 = 2*3*5*7
    target = (1 << 0) | (1 << 1) | (1 << 2) | (1 << 3)

    # Precompute parity bits for all n in [211, MAX_N]
    parity = [0] * (MAX_N + 1)
    for n in range(211, MAX_N + 1):
        parity[n] = get_parity_bits(n)

    # Incremental Gaussian elimination with subset tracking
    # We'll add vectors for n = 211, 212, 213, ... and after each, check if target is in span
    # Key insight: we want the MINIMAL max element, so we check in order and stop at first valid solution

    best_solution = None

    # Try incremental approach
    basis = {}  # pivot -> (vec, frozenset_of_indices)

    for n in range(211, MAX_N + 1):
        if time.time() > deadline - 5.0:
            break

        v = parity[n]
        idx = n - 211

        if v == 0:
            continue

        # Add to basis
        cur = v
        used = {idx}
        hb = cur.bit_length() - 1
        while hb >= 0 and hb in basis:
            bv, bused = basis[hb]
            cur ^= bv
            used = used.symmetric_difference(set(bused))
            hb = cur.bit_length() - 1 if cur else -1

        if cur != 0:
            hb = cur.bit_length() - 1
            basis[hb] = (cur, frozenset(used))

        # Check if target is now in span
        cur_check = target
        used_check = set()
        hb_check = cur_check.bit_length() - 1 if cur_check else -1
        while hb_check >= 0 and hb_check in basis:
            bv, bused = basis[hb_check]
            cur_check ^= bv
            used_check = used_check.symmetric_difference(set(bused))
            hb_check = cur_check.bit_length() - 1 if cur_check else -1

        if cur_check == 0 and used_check:
            # Convert indices back to actual numbers
            selected = [211 + i for i in sorted(used_check)]
            # Verify
            prod = 210
            for x in selected:
                prod *= x
            if _is_perfect_square(prod):
                best_solution = selected
                break

    if best_solution is None:
        # Fallback: hardcoded known solution
        best_solution = [224, 240]

    result = [int(x) for x in best_solution]
    return result
