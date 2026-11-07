"""Square-product extension for 210.

Find distinct ints in (210, 10210] (<=100 of them) whose product times 210 is a
perfect square, minimizing max(selected)-210.

Strategy (GF(2) exact):
- Represent each candidate n by its square-free parity bitvector (bit i =
  exponent of prime_i mod 2).  "210 * prod(selected) is a square"  <=>
  XOR of the selected vectors == parity(210).
- Feasibility for a given upper bound M (some non-empty subset of {211..M}
  XORs to the target) is monotone in M, so we binary-search the smallest
  feasible M  ->  this is exactly the minimum possible max(selected).
- A single GF(2) elimination pass stores, per pivot, both the reduced vector
  and the integer bitmask of the *original* rows that form it, so the exact
  satisfying subset is recovered in the same pass.
"""

import math
import time

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
    for i in range(2, int(limit ** 0.5) + 1):
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
    DEADLINE = t_start + 78.0  # safety margin well under the 90 s kill

    primes = _sieve_primes(MAX_N)
    prime_index = {p: i for i, p in enumerate(primes)}

    # Precompute the square-free parity bitvector (Python int) for 211..MAX_N.
    parity_vec = {}
    for n in range(211, MAX_N + 1):
        d = _factorize(n, primes)
        v = 0
        for p, e in d.items():
            if e & 1:
                v |= (1 << prime_index[p])
        parity_vec[n] = v

    # Target = parity(210) = 2^1 * 3^1 * 5^1 * 7^1
    target = 0
    for p in SMALL_PRIMES:
        target |= (1 << prime_index[p])

    def _feasible(M):
        """Return (ok, selected) for a non-empty subset of [211..M] whose
        parity XOR equals target.  Single GF(2) elimination pass that also
        tracks which original rows compose each pivot (integer bitmask)."""
        if M < 211:
            return (False, None)

        # basis[pivot_bit] = (reduced_vector, original_row_mask)
        basis = {}
        for idx, n in enumerate(range(211, M + 1)):
            x = parity_vec[n]
            om = 1 << idx
            while x:
                pb = x.bit_length() - 1
                if pb in basis:
                    x ^= basis[pb][0]
                    om ^= basis[pb][1]
                else:
                    basis[pb] = (x, om)
                    break

        # Reduce target against the basis; accumulate the selected mask.
        x = target
        sel_mask = 0
        while x:
            pb = x.bit_length() - 1
            if pb in basis:
                x ^= basis[pb][0]
                sel_mask ^= basis[pb][1]
            else:
                return (False, None)

        if sel_mask == 0:
            return (False, None)

        selected = []
        i = 0
        while sel_mask:
            if sel_mask & 1:
                selected.append(211 + i)
            sel_mask >>= 1
            i += 1
        return (True, selected)

    # --- Binary search the minimal feasible M in [211, MAX_N] ---
    best_M = None
    best_selected = None

    # Sanity: is a solution available at all up to MAX_N?
    ok, sel = _feasible(MAX_N)
    if ok:
        best_M = MAX_N
        best_selected = sel

    lo, hi = 211, MAX_N
    while lo < hi and time.time() < DEADLINE:
        mid = (lo + hi) // 2
        ok, sel = _feasible(mid)
        if ok:
            best_M = mid
            best_selected = sel
            hi = mid
        else:
            lo = mid + 1

    if best_selected is None:
        # No subset found in time / at all -> guaranteed fallback.
        return [840]

    # --- Post-process: enforce distinctness, bounds, and the 100-element cap. ---
    selected = []
    seen = set()
    for x in best_selected:
        if isinstance(x, int) and not isinstance(x, bool) and 210 < x <= MAX_N and x not in seen:
            seen.add(x)
            selected.append(x)
    if not selected:
        return [840]

    # If the exact subset exceeds 100 elements, trim to the 100 *smallest*
    # (keeps max minimal) and re-verify; if that no longer squares, fall back.
    if len(selected) > MAX_K:
        selected = sorted(selected)[:MAX_K]
        prod = N
        for x in selected:
            prod *= x
        if not _is_square(prod):
            return [840]

    # --- Exact verification of the square condition ---
    prod = N
    for x in selected:
        prod *= x
    if _is_square(prod):
        return [int(x) for x in selected]

    # Fallback guaranteed solution: 210 * 840 = 210 * 210 * 4 = (210*2)^2.
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
