import time
import math

def construct():
    answers = []
    deadline = time.time() + 55
    BOUND = 10**12

    for n in range(3, 257):
        if time.time() > deadline:
            answers.append(None)
            continue

        found = None
        x_start = n // 4 + 1
        x_end = (3 * n) // 4

        for x in range(x_start, x_end + 1):
            if time.time() > deadline:
                break

            a = 4 * x - n
            b = n * x

            if a <= 0:
                continue

            # (a*y - b)(a*z - b) = b^2
            # d = a*y - b, d | b^2, d < b (for z > y), d > a*x - b (for y > x)
            d_min = max(1, a * x - b + 1)
            d_max = b - 1

            if d_min > d_max:
                continue

            # Factor b
            temp = b
            factors = {}
            p = 2
            while p * p <= temp:
                while temp % p == 0:
                    factors[p] = factors.get(p, 0) + 1
                    temp //= p
                p += 1
            if temp > 1:
                factors[temp] = factors.get(temp, 0) + 1

            # Generate divisors of b^2 in [d_min, d_max]
            primes = list(factors.keys())
            max_pows = [factors[p] * 2 for p in primes]
            divs = []

            def gen(idx, cur):
                if idx == len(primes):
                    if d_min <= cur <= d_max:
                        divs.append(cur)
                    return
                pw = 1
                for _ in range(max_pows[idx] + 1):
                    gen(idx + 1, cur * pw)
                    pw *= primes[idx]
                    if cur * pw > d_max:
                        break

            gen(0, 1)

            for d in divs:
                if time.time() > deadline:
                    break
                if (d + b) % a != 0:
                    continue
                y = (d + b) // a
                if y <= x:
                    continue
                b2d = (b * b) // d
                if (b2d + b) % a != 0:
                    continue
                z = (b2d + b) // a
                if z <= y:
                    continue
                if z > BOUND:
                    continue
                if x == y or y == z or x == z:
                    continue
                lhs = n * (y * z + x * z + x * y)
                rhs = 4 * x * y * z
                if lhs == rhs:
                    found = [x, y, z]
                    break

            if found:
                break

        answers.append(found)

    return answers
