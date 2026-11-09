"""Bitset-based sliding-window search for tightest packing of distinct multiples."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from functools import reduce

_DIVISORS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
_N = 10
_MAX_V = 1000000
_LCM = 270270  # lcm(6,10,14,15,21,22,26,33,35,39)


def _lcm(a, b):
    return a * b // gcd(a, b)


def _check_window(W, L):
    """Try to assign distinct multiples of each divisor within [L, L+W].
    Returns list of assigned values (in divisor order) or None."""
    n = _N
    divisors = _DIVISORS

    # Build bitset of candidate offsets for each divisor
    # Bit j set means L+j is a multiple of the divisor
    cand = []
    for d in divisors:
        p = L % d
        first = (d - p) % d
        if first > W:
            return None
        bits = 0
        off = first
        while off <= W:
            bits |= (1 << off)
            off += d
        cand.append(bits)

    # Build position -> divisor bitmask (which divisors can use position j)
    pos_mask = [0] * (W + 1)
    for i in range(n):
        b = cand[i]
        j = 0
        while b:
            if b & 1:
                pos_mask[j] |= (1 << i)
            b >>= 1
            j += 1

    # MRV ordering: sort divisors by number of available positions
    order = sorted(range(n), key=lambda i: bin(cand[i]).count('1'))

    assignment = [None] * n
    used = 0  # bitset of used positions

    def backtrack(idx):
        if idx == n:
            return True
        i = order[idx]
        avail = cand[i] & ~used
        if avail == 0:
            return False
        # Try positions, prefer those with fewer conflicts for remaining divisors
        bits = avail
        j = 0
        candidates = []
        while bits:
            if bits & 1:
                # Count how many remaining unassigned divisors also want this position
                conflicts = 0
                for k in range(idx + 1, n):
                    if cand[k] & (1 << j):
                        conflicts += 1
                candidates.append((conflicts, j))
            bits >>= 1
            j += 1
        candidates.sort()
        for _, j in candidates:
            bit = 1 << j
            if avail & bit:
                used |= bit
                assignment[i] = L + j
                if backtrack(idx + 1):
                    return True
                used &= ~bit
                assignment[i] = None
        return False

    if backtrack(0):
        return assignment
    return None


def construct():
    start = time.time()
    deadline = start + 85.0  # safety margin before 90s kill
    n = _N
    LCM = _LCM

    # Fallback: the divisors themselves (all distinct, all valid)
    fallback = [int(d) for d in _DIVISORS]
    best_assign = fallback[:]
    best_W = max(fallback) - min(fallback)  # 33

    # Theoretical minimum span for 10 distinct integers is 9
    # Search W from 9 upward
    for W in range(9, best_W + 1):
        if time.time() > deadline - 5:
            break

        # L only needs to be tried in [1, LCM] due to periodicity
        # But we can be smarter: L must be >= 1 and L+W <= MAX_V
        max_L = min(LCM, _MAX_V - W)

        found = False
        # Try all L from 1 to max_L
        # For small W, each check is very fast (tiny bitsets)
        for L in range(1, max_L + 1):
            if time.time() > deadline - 5:
                break
            result = _check_window(W, L)
            if result is not None:
                # Verify
                if (len(set(result)) == n
                        and all(result[i] % _DIVISORS[i] == 0 for i in range(n))
                        and all(1 <= result[i] <= _MAX_V for i in range(n))):
                    span = max(result) - min(result)
                    if span <= W:
                        best_W = span
                        best_assign = result[:]
                        found = True
                        break

        if found:
            # Found optimal for this W; since we search W in increasing order,
            # this is the minimum W. But the actual span might be < W.
            # We can stop since we want minimum span.
            break

    # Verify and return
    out = []
    for x in best_assign:
        out.append(int(x))
    return out
# EVOLVE-BLOCK-END
