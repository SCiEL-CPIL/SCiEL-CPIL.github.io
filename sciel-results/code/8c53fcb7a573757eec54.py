import time
from math import isqrt

_SOLUTIONS = {}
_DEADLINE = 85.0  # seconds, with safety margin

def _factorize(B):
    """Return dict of prime -> exponent for B."""
    factors = {}
    temp = B
    d = 2
    while d * d <= temp:
        while temp % d == 0:
            factors[d] = factors.get(d, 0) + 1
            temp //= d
        d += 1
    if temp > 1:
        factors[temp] = factors.get(temp, 0) + 1
    return factors

def _divisors_leq_B(B):
    """All divisors of B^2 that are <= B."""
    if B <= 0:
        return [1]
    factors = _factorize(B)
    primes = list(factors.keys())
    divs = [1]
    for p in primes:
        max_exp = 2 * factors[p]
        new_divs = []
        p_pows = [1]
        for e in range(1, max_exp + 1):
            p_pows.append(p_pows[-1] * p)
        for d in divs:
            for pp in p_pows:
                new_divs.append(d * pp)
        divs = new_divs
    return [d for d in divs if d <= B]

def _find_solution(n):
    """Find distinct positive integers x, y, z <= 10^12 such that 1/x+1/y+1/z = 4/n.
    Uses factorization: (Ay-B)(Az-B) = B^2 where A=4x-n, B=nx."""
    start_x = n // 4 + 1
    end_x = (3 * n) // 4 + 1
    for x in range(start_x, end_x + 1):
        A = 4 * x - n
        if A <= 0:
            continue
        B = n * x
        divs = _divisors_leq_B(B)
        B_sq = B * B
        for u in divs:
            v = B_sq // u
            if (u + B) % A != 0:
                continue
            if (v + B) % A != 0:
                continue
            y = (u + B) // A
            z = (v + B) // A
            if y <= 0 or z <= 0:
                continue
            if y > 10**12 or z > 10**12:
                continue
            if x == y or x == z or y == z:
                continue
            return (x, y, z)
    return None

def _find_solution_fast(n):
    """Fast fallback with limited x range."""
    start_x = n // 4 + 1
    end_x = min(start_x + 30, (3 * n) // 4 + 1)
    for x in range(start_x, end_x + 1):
        A = 4 * x - n
        if A <= 0:
            continue
        B = n * x
        divs = _divisors_leq_B(B)
        B_sq = B * B
        for u in divs:
            v = B_sq // u
            if (u + B) % A != 0:
                continue
            if (v + B) % A != 0:
                continue
            y = (u + B) // A
            z = (v + B) // A
            if y <= 0 or z <= 0 or y > 10**12 or z > 10**12:
                continue
            if x == y or x == z or y == z:
                continue
            return (x, y, z)
    return None

def _precompute():
    global _SOLUTIONS
    _SOLUTIONS = {}
    start_time = time.time()
    for n in range(3, 257):
        if time.time() - start_time > _DEADLINE:
            _SOLUTIONS[n] = _find_solution_fast(n)
            continue
        _SOLUTIONS[n] = _find_solution(n)

# Run precomputation at module load
_start_time = time.time()
_precompute()

def construct():
    answers = []
    for n in range(3, 257):
        sol = _SOLUTIONS.get(n)
        if sol:
            answers.append(list(sol))
        else:
            answers.append(None)
    return answers
