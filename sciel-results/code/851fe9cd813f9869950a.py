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
        x_min = n // 4 + 1
        x_max = (3 * n - 1) // 4

        # Build x candidates: prioritize x=n+1, then small x values
        x_candidates = []
        if n + 1 >= x_min and n + 1 <= x_max:
            x_candidates.append(n + 1)
        for x in range(x_min, min(x_min + 80, x_max + 1)):
            if x != n + 1:
                x_candidates.append(x)
        for x in range(x_min + 80, min(x_max + 1, x_min + 200)):
            if x != n + 1 and x not in x_candidates:
                x_candidates.append(x)

        for x in x_candidates:
            if time.time() > deadline:
                break

            a = 4 * x - n
            if a <= 0:
                continue
            b = n * x

            bf = _factorize(b)
            bf2 = {p: e * 2 for p, e in bf.items()}

            num_divs = 1
            for e in bf2.values():
                num_divs *= (e + 1)
            if num_divs > 200000:
                continue

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
                if y == z or y == x or z == x:
                    continue
                if y > z:
                    y, z = z, y
                found = [x, y, z]
                break

            if found:
                break

        if found:
            answers.append(found)
            continue

        # Final fallback: brute force over x
        for x in range(x_min, x_max + 1):
            if time.time() > deadline:
                break
            A = 4 * x - n
            if A <= 0:
                continue
            B = n * x

            y_min = B // A + 1
            y_max = (2 * B - 1) // A
            if y_max < y_min:
                continue
            if y_min <= x:
                y_min = x + 1
            if y_max < y_min:
                continue

            y_search_max = min(y_max, y_min + 5000)
            for y in range(y_min, y_search_max + 1):
                denom = A * y - B
                if denom <= 0:
                    continue
                if (B * y) % denom == 0:
                    z = (B * y) // denom
                    if z > y and z <= LIMIT_Z and z != x and y != x:
                        found = [x, y, z]
                        break

            if found:
                break

        answers.append(found)

    return answers
