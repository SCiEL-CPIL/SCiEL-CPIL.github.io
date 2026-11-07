import time
from math import isqrt

def _is_perfect_square(n):
    if n < 0:
        return False
    r = isqrt(n)
    return r * r == n

def construct():
    start = time.time()
    deadline = start + 80.0

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
                    # p > MAX_N, shouldn't happen for n <= MAX_N
                    pass
        return bits

    # Target: parity of 210 = 2*3*5*7
    # 2->idx0, 3->idx1, 5->idx2, 7->idx3
    target = (1 << 0) | (1 << 1) | (1 << 2) | (1 << 3)

    # Precompute parity bits for all n in [211, MAX_N]
    parity = [0] * (MAX_N + 1)
    for n in range(211, MAX_N + 1):
        parity[n] = get_parity_bits(n)

    # Incremental Gaussian elimination with subset tracking
    # basis: dict pivot_bit -> (reduced_vec, frozenset_of_original_indices)
    # We'll add vectors for n = 211, 212, 213, ... and after each, check if target is in span

    basis = {}  # pivot -> (vec, frozenset)
    # Also track: for each original index, its contribution to basis (for backtracking)
    # Actually, let's just rebuild the solution when we find it

    def try_solve(target_bits, basis_dict, max_idx):
        """Try to express target_bits using basis_dict. Return frozenset of indices or None."""
        if target_bits == 0:
            # Need non-empty dependent set - handle separately
            return None
        cur = target_bits
        used = set()
        hb = cur.bit_length() - 1 if cur else -1
        while hb >= 0 and hb in basis_dict:
            bv, bused = basis_dict[hb]
            cur ^= bv
            used = used.symmetric_difference(set(bused))
            hb = cur.bit_length() - 1 if cur else -1
        if cur != 0:
            return None
        if not used:
            return None
        return frozenset(used)

    best_solution = None

    # Incrementally add vectors
    for n in range(211, MAX_N + 1):
        if time.time() > deadline - 3.0:
            break

        v = parity[n]
        idx = n - 211  # 0-based index in our list

        if v == 0:
            # Zero vector, skip for basis but note it
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
        sol = try_solve(target, basis, idx)
        if sol is not None:
            # Convert indices back to actual numbers
            selected = [211 + i for i in sorted(sol)]
            # Verify
            prod = 210
            for x in selected:
                prod *= x
            if _is_perfect_square(prod):
                best_solution = selected
                break

    if best_solution is None:
        # Fallback: try a known small solution
        # Let's try to find one by brute force for small M
        for M in range(211, 300):
            if time.time() > deadline - 2.0:
                break
            # Build basis from scratch for [211, M]
            b = {}
            for n in range(211, M + 1):
                v = parity[n]
                idx = n - 211
                if v == 0:
                    continue
                cur = v
                used = {idx}
                hb = cur.bit_length() - 1
                while hb >= 0 and hb in b:
                    bv, bused = b[hb]
                    cur ^= bv
                    used = used.symmetric_difference(set(bused))
                    hb = cur.bit_length() - 1 if cur else -1
                if cur != 0:
                    hb = cur.bit_length() - 1
                    b[hb] = (cur, frozenset(used))
            sol = try_solve(target, b, M - 211)
            if sol is not None:
                selected = [211 + i for i in sorted(sol)]
                prod = 210
                for x in selected:
                    prod *= x
                if _is_perfect_square(prod):
                    best_solution = selected
                    break

    if best_solution is None:
        # Hardcoded fallback
        best_solution = [224, 240]

    result = [int(x) for x in best_solution]
    return result
