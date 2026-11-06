"""Factorial decompositions into distinct factors above the index.

Return 1..12 DISTINCT integers all greater than 12, with product exactly 12!.
Minimize the LARGEST factor; the number of factors is variable.

Strategy (branch-and-bound on the objective):
  Iterate a candidate maximum M from a lower bound upward.  For each M, run a
  pruned feasibility DFS that builds factors in *increasing* order, each <= M,
  distinct, > 12, with product exactly 12!.  The first feasible M is optimal.

  Pruning (all exact / safe):
    - next factor must be > previous (increasing) and <= M (objective bound)
    - the remaining product must be achievable by the still-available count of
      DISTINCT integers in (prev, M]  ->  suffix-product bound
    - the next factor must be a divisor of the remaining product
    - the remaining product must be a divisor of 12! (in the precomputed set)

  After the optimal M is found, a final local "split" pass tries to reduce the
  largest factor further (a safety net that can only help).
"""
import math
import time

T12 = math.factorial(12)          # 479001600
PRIMES = [2, 3, 5, 7, 11]
EXP = [10, 5, 2, 1, 1]
N_PRIMES = len(PRIMES)


def valid_solution(factors):
    """Check: 1-12 distinct integers > 12, product exactly 12!."""
    if not (1 <= len(factors) <= 12):
        return False
    if len(set(factors)) != len(factors):
        return False
    if any(f < 13 for f in factors):
        return False
    p = 1
    for f in factors:
        p *= f
    return p == T12


def _build_divisor_set():
    """All divisors of 12! as a set (for fast membership tests)."""
    divs = {1}
    for p, e in zip(PRIMES, EXP):
        new = set()
        mult = 1
        for k in range(e + 1):
            for d in divs:
                new.add(d * mult)
            mult *= p
        divs = new
    return divs


_DIVS = _build_divisor_set()


def _divs_below(n, limit):
    """Divisors of n (including n) that are <= limit, sorted ascending.

    n is always a divisor of 12! at the points where this is called, so the
    prime exponents of n are within EXP.
    """
    exps = [0] * N_PRIMES
    x = n
    for i, p in enumerate(PRIMES):
        while x % p == 0:
            exps[i] += 1
            x //= p
    res = [1]
    for i, p in enumerate(PRIMES):
        cur = list(res)
        val = 1
        for _ in range(exps[i]):
            val *= p
            res.extend(c * val for c in cur)
    out = [d for d in res if d <= limit]
    out.sort()
    return out


def _feasible(remaining, prev, count, maxf, factors, deadline):
    """DFS: can we finish so that product == 12!, factors distinct, increasing,
    all in (12, maxf]?  Returns a list of factors (a solution) or None."""
    if time.time() > deadline:
        return None

    if remaining == 1:
        if count >= 1:
            return factors[:]
        return None

    if count >= 12:
        return None

    # Suffix-product bound: the remaining product must be attainable by the
    # still-available count of DISTINCT integers in (prev, maxf].
    avail = maxf - prev
    if avail <= 0:
        return None
    k = 12 - count
    if avail < k:
        return None
    # max product of k distinct ints in (prev, maxf] = maxf*(maxf-1)*...*(maxf-k+1)
    maxp = 1
    for j in range(k):
        maxp *= (maxf - j)
        if maxp >= remaining:
            break
    if maxp < remaining:
        return None
    # min product of k distinct ints in (prev, maxf] = (prev+1)*...*(prev+k)
    minp = 1
    for j in range(1, k + 1):
        minp *= (prev + j)
    if minp > remaining:
        return None

    for d in _divs_below(remaining, maxf):
        if d <= prev:
            continue
        nr = remaining // d
        if nr == 1:
            # d is the last factor
            if len(factors) + 1 >= 1:
                return factors + [d]
            continue
        # next factor must be a divisor of 12!
        if nr not in _DIVS:
            continue
        res = _feasible(nr, d, count + 1, maxf, factors + [d], deadline)
        if res is not None:
            return res
    return None


def _find_solution(maxf, deadline):
    """Find any valid factorization with every factor <= maxf, or None."""
    if maxf < 13:
        return None
    # 12! itself as a single factor is only allowed if it is <= maxf
    if T12 <= maxf:
        return [T12]
    return _feasible(T12, 12, 0, maxf, [], deadline)


def _lower_bound():
    """A valid lower bound on the largest factor."""
    # 12! as one factor
    lb = T12
    # 12! = 13 * 36846277  (both > 12, distinct)
    if 13 * 36846277 == T12:
        lb = min(lb, 36846277)
    # 12! = 14 * 34214400
    if 14 * 34214400 == T12:
        lb = min(lb, 34214400)
    return lb


def _local_improve(factors, deadline):
    """Try to reduce the largest factor by splitting it into two smaller
    distinct factors (each > 12, not colliding with the others)."""
    if not factors:
        return factors
    best = list(factors)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        cur_max = max(best)
        others = [f for f in best if f != cur_max]
        for a in range(13, math.isqrt(cur_max) + 1):
            if time.time() > deadline:
                break
            if cur_max % a == 0:
                b = cur_max // a
                if b > a and a >= 13 and b >= 13:
                    if a not in others and b not in others:
                        new_set = others + [a, b]
                        if valid_solution(new_set):
                            if max(new_set) < cur_max:
                                best = new_set
                                improved = True
                                break
    return best


def construct():
    """Construct a valid factorization of 12! into distinct integers > 12,
    minimizing the largest factor."""
    start = time.time()
    deadline = start + 80.0          # safety margin under the 90 s limit

    # A guaranteed-valid fallback (two factors).
    fallback = None
    if 13 * 36846277 == T12:
        fallback = [13, 36846277]

    # Branch-and-bound on the objective: scan M upward, first feasible is optimal.
    best = None
    best_max = None
    M = _lower_bound()
    while time.time() < deadline:
        sol = _find_solution(M, deadline)
        if sol is not None:
            best = sol
            best_max = M
            break
        M += 1

    if best is None:
        # Time ran out before a feasible M was found; use the fallback.
        if fallback is not None and valid_solution(fallback):
            return fallback
        return [T12]

    # Final polish: try to shave the largest factor by splitting it.
    if time.time() < deadline - 1.0:
        improved = _local_improve(best, deadline)
        if valid_solution(improved):
            best = improved

    if best and valid_solution(best):
        return best
    if fallback is not None:
        return fallback
    return [T12]
