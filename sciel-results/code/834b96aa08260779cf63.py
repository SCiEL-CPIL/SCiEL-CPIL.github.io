import time

def _factorize(n):
    factors = {}
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors[d] = factors.get(d, 0) + 1
            n //= d
        d += 1
    if n > 1:
        factors[n] = factors.get(n, 0) + 1
    return factors

def _divisors_from_factors(factors):
    primes = list(factors.keys())
    exps = [factors[p] for p in primes]
    divisors = [1]
    for p, e in zip(primes, exps):
        new_divs = []
        for d in divisors:
            val = d
            for _ in range(e + 1):
                new_divs.append(val)
                val *= p
        divisors = new_divs
    return divisors

def construct():
    answers = []
    deadline = time.time() + 55.0
    LIMIT_Z = 10**12

    for n in range(3, 257):
        if time.time() > deadline:
            answers.append(None)
            continue

        found = None

        # Try divisor-based approach with several candidate x values
        # x must satisfy: 4x - n > 0, and for distinct y,z > 0
        # Also need 1/x < 4/n so x > n/4
        # And for y,z to exist: 4/n - 1/x > 0 and the remaining must be expressible
        # Try x = n+1, n+2, n+3, ..., up to a few values
        for x_try in range(n + 1, min(n + 11, 4 * n + 1)):
            if time.time() > deadline:
                break
            a = 4 * x_try - n
            b = n * x_try
            if a <= 0:
                continue

            bf = _factorize(b)
            bf2 = {p: e * 2 for p, e in bf.items()}
            divisors_b2 = _divisors_from_factors(bf2)

            for d in divisors_b2:
                if (d + b) % a != 0:
                    continue
                e_val = b * b // d
                if (e_val + b) % a != 0:
                    continue
                y = (d + b) // a
                z = (e_val + b) // a
                if y <= 0 or z <= 0:
                    continue
                if y > LIMIT_Z or z > LIMIT_Z:
                    continue
                if y == z or y == x_try or z == x_try:
                    continue
                if y > z:
                    y, z = z, y
                found = [x_try, y, z]
                break

            if found:
                break

        if found:
            answers.append(found)
            continue

        # Fallback: brute force over x with optimized y search
        x_min = n // 4 + 1
        x_max = (3 * n - 1) // 4

        for x in range(x_max, x_min - 1, -1):
            if time.time() > deadline:
                break
            A = 4 * x - n
            B = n * x

            y_min = B // A + 1
            y_max = (2 * B - 1) // A

            if y_max < y_min:
                continue
            if y_min <= x:
                y_min = x + 1
            if y_max < y_min:
                continue

            # Try y near the geometric mean first, then expand
            y_mid = (y_min + y_max) // 2
            # Search in order: mid, mid-1, mid+1, mid-2, mid+2, ...
            search_order = []
            for offset in range(0, (y_max - y_min) // 2 + 1):
                y_candidate = y_mid - offset
                if y_candidate >= y_min:
                    search_order.append(y_candidate)
                y_candidate2 = y_mid + offset
                if y_candidate2 <= y_max and y_candidate2 != y_candidate:
                    search_order.append(y_candidate2)

            for y in search_order:
                denom = A * y - B
                if denom <= 0:
                    continue
                if (B * y) % denom == 0:
                    z = (B * y) // denom
                    if z > y and z <= LIMIT_Z:
                        found = [x, y, z]
                        break

            if found:
                break

        answers.append(found)

    return answers
