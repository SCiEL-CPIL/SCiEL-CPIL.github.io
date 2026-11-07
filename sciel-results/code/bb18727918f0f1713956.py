import time
import math
import sys
from collections import defaultdict

def construct():
    deadline = time.time() + 55.0  # safety margin before 60s
    n = 210
    max_val = 10210

    def is_square(m):
        if m < 0:
            return False
        s = math.isqrt(m)
        return s * s == m

    # Single element: a * 210 = s^2 => a = 210*k^2
    # k=2: a=840, value=630
    best_value = 630
    best_result = [840]

    # For pairs: a*b = 210*t^2, need 211 <= a < b <= 10210
    # M = 210*t^2. Need a | M, a in [lo, hi], b = M//a valid.
    # lo = max(211, ceil(M/10210)), hi = isqrt(M) - 1 (so a < sqrt(M), hence b > a)

    def factorize(x):
        factors = {}
        d = 2
        while d * d <= x:
            while x % d == 0:
                factors[d] = factors.get(d, 0) + 1
                x //= d
            d += 1
        if x > 1:
            factors[x] = factors.get(x, 0) + 1
        return factors

    def get_divisors(factors):
        divs = [1]
        for p, e in factors.items():
            new_divs = []
            for d in divs:
                val = d
                for _ in range(e + 1):
                    new_divs.append(val)
                    val *= p
            divs = new_divs
        return divs

    # Minimum t: 210*t^2 >= 211*212 = 44732 => t^2 >= 213.01 => t >= 15
    t = 15
    while time.time() < deadline:
        M = 210 * t * t
        lo = max(211, (M + 10209) // 10210)
        hi = int(math.isqrt(M)) - 1

        if lo > hi:
            t += 1
            continue

        # Factor M = 210 * t^2
        t_factors = factorize(t)
        M_factors = {2: 1, 3: 1, 5: 1, 7: 1}
        for p, e in t_factors.items():
            M_factors[p] = M_factors.get(p, 0) + 2 * e

        divs = get_divisors(M_factors)

        for a in divs:
            if a < lo or a > hi:
                continue
            if M % a != 0:
                continue
            b = M // a
            if b <= a or b > 10210 or b <= 210:
                continue
            val = b - 210
            if val < best_value:
                best_value = val
                best_result = [int(a), int(b)]

        if best_value <= 5:
            break

        t += 1
        if t > 50000:
            break

    # Verify
    prod = 210
    for x in best_result:
        prod *= x
    if not is_square(prod):
        return [840]

    return best_result
