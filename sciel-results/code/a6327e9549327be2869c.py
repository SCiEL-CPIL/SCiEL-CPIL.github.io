"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def _factorize(n):
    """Return list of (prime, exponent) pairs for n."""
    factors = []
    d = 2
    while d * d <= n:
        if n % d == 0:
            cnt = 0
            while n % d == 0:
                n //= d
                cnt += 1
            factors.append((d, cnt))
        d += 1
    if n > 1:
        factors.append((n, 1))
    return factors

def _gen_divisors(factors):
    """Generate all divisors from factorization list."""
    divs = [1]
    for p, e in factors:
        new_divs = []
        for d in divs:
            val = d
            for _ in range(e + 1):
                new_divs.append(val)
                val *= p
        divs = new_divs
    return divs

def construct():
    start_time = time.time()
    deadline = start_time + 82.0
    LIMIT = 10**12
    answers = [None] * 254

    for n in range(3, 257):
        if time.time() > deadline:
            break
        found = None
        x_min = n // 4 + 1
        x_max = 3 * n // 4

        # Primary: direct y-loop (O(1) per y, no factorization)
        for x in range(x_min, x_max + 1):
            if time.time() > deadline:
                break
            A = 4 * x - n
            if A <= 0:
                continue
            B = n * x
            y_min = max(x + 1, B // A + 1)
            if (2 * B) % A == 0:
                y_max = (2 * B) // A - 1
            else:
                y_max = (2 * B) // A
            if y_min > y_max:
                continue
            # Cap y iterations to avoid pathological cases
            y_span = y_max - y_min + 1
            if y_span > 200000:
                continue
            for y in range(y_min, y_max + 1):
                denom = A * y - B
                if denom <= 0:
                    continue
                if (B * y) % denom == 0:
                    z = (B * y) // denom
                    if z > y and z <= LIMIT:
                        if x != y and y != z and x != z:
                            found = [x, y, z]
                            break
            if found:
                break

        # Fallback: divisor-based search for x with large y-span
        if not found and time.time() < deadline - 2.0:
            for x in range(x_min, x_max + 1):
                if time.time() > deadline:
                    break
                A = 4 * x - n
                if A <= 0:
                    continue
                B = n * x
                B2 = B * B
                y_min = max(x + 1, B // A + 1)
                if (2 * B) % A == 0:
                    y_max = (2 * B) // A - 1
                else:
                    y_max = (2 * B) // A
                if y_min > y_max:
                    continue
                y_span = y_max - y_min + 1
                if y_span <= 200000:
                    continue  # Already tried in primary loop

                fac = _factorize(B2)
                divs = _gen_divisors(fac)
                for d in divs:
                    if (d + B) % A != 0:
                        continue
                    d2 = B2 // d
                    if (d2 + B) % A != 0:
                        continue
                    y = (d + B) // A
                    z = (d2 + B) // A
                    if y <= 0 or z <= 0:
                        continue
                    if y > x and z > y and z <= LIMIT:
                        if x != y and y != z and x != z:
                            found = [x, y, z]
                            break
                if found:
                    break

        answers[n - 3] = found

    return answers
# EVOLVE-BLOCK-END
