import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin

    n = 210
    max_val = 10210

    # Factorize 210: 2^1 * 3^1 * 5^1 * 7^1
    # Target: XOR of selected parity vectors must equal parity of 210
    # Parity of 210: bits for primes 2,3,5,7 are set

    # Sieve for primes up to max_val
    is_prime = [True] * (max_val + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(max_val ** 0.5) + 1):
        if is_prime[i]:
            for j in range(i * i, max_val + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, max_val + 1) if is_prime[i]]
    prime_index = {p: i for i, p in enumerate(primes)}
    num_primes = len(primes)

    def get_parity_mask(x):
        """Return bitmask where bit i is set if prime primes[i] appears to odd power in x."""
        mask = 0
        temp = x
        for idx, p in enumerate(primes):
            if p * p > temp:
                break
            if temp % p == 0:
                count = 0
                while temp % p == 0:
                    temp //= p
                    count ^= 1
                if count:
                    mask |= (1 << idx)
        if temp > 1:
            if temp in prime_index:
                mask |= (1 << prime_index[temp])
        return mask

    # Precompute masks for all numbers up to max_val
    masks = [0] * (max_val + 1)
    for i in range(2, max_val + 1):
        masks[i] = get_parity_mask(i)

    # Parity mask of 210
    target = masks[210]

    # Basis: dict mapping highest-bit position -> (vector, history_mask)
    basis = {}

    for M in range(211, max_val + 1):
        if time.time() > deadline:
            break

        v = masks[M]
        h = 1 << (M - 211)  # history bit for this candidate

        # Insert v into basis with history h
        tv = v
        th = h
        while tv:
            p = tv.bit_length() - 1
            if p in basis:
                bv, bh = basis[p]
                tv ^= bv
                th ^= bh
            else:
                basis[p] = (tv, th)
                break

        # Check if target is in span
        tv = target
        th = 0
        while tv:
            p = tv.bit_length() - 1
            if p in basis:
                bv, bh = basis[p]
                tv ^= bv
                th ^= bh
            else:
                break

        if tv == 0:
            # Extract solution from history mask th
            sol = []
            hh = th
            while hh:
                lsb = hh & -hh
                idx = lsb.bit_length() - 1
                sol.append(211 + idx)
                hh ^= lsb

            # Verify size constraint
            if len(sol) <= 100:
                # Verify the product condition
                prod = 210
                for x in sol:
                    prod *= x
                s = math.isqrt(prod)
                if s * s == prod:
                    return [int(x) for x in sol]
            # If size > 100, continue to next M (larger max might allow smaller subset)
        # Continue to next M

    # Fallback (should not be reached)
    return [840]
