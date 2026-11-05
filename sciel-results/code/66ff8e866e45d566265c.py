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
        B_sq = B * B
        
        # Generate all divisors of B^2, sorted
        factors_B = _factorize(B)
        primes = list(factors_B.keys())
        divs = [1]
        for p in primes:
            max_exp = 2 * factors_B[p]
            new_divs = []
            p_pow = 1
            for e in range(max_exp + 1):
                for d in divs:
                    new_divs.append(d * p_pow)
                p_pow *= p
            divs = new_divs
        divs.sort()
        
        for u in divs:
            if B_sq % u != 0:
                continue
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

def _precompute():
    global _SOLUTIONS
    _SOLUTIONS = {}
    start_time = time.time()
    for n in range(3, 257):
        if time.time() - start_time > _DEADLINE:
            _SOLUTIONS[n] = None
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
