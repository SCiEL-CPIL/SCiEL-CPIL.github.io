import time
from math import gcd

def construct():
    MAX_DENOM = 10**12
    deadline = time.time() + 80.0
    answers = [None] * 254

    for idx, n in enumerate(range(3, 257)):
        if time.time() > deadline:
            break
        found = None
        x_min = max(1, n // 4 + 1)
        x_cap = (3 * n) // 4
        if x_cap < x_min:
            x_cap = x_min
        for x in range(x_min, x_cap + 1):
            A = 4 * x - n
            if A <= 0:
                continue
            B = n * x
            B2 = B * B
            # y = (d + B) / A where d | B^2, need A | (d+B), y > x, z = (B^2/d + B)/A > y
            # Iterate d over divisors of B^2 that are < B^2 (so z > 0)
            # d = Ay - B, need d > 0 so y > B/A. Also y > x.
            # Find divisors of B^2
            # Factor B
            def divisors_of_square(B):
                # return list of divisors d of B^2
                # factor B
                b = B
                factors = {}
                d = 2
                while d * d <= b:
                    while b % d == 0:
                        factors[d] = factors.get(d, 0) + 1
                        b //= d
                    d += 1
                if b > 1:
                    factors[b] = factors.get(b, 1)
                # B^2 has exponents doubled
                # generate divisors
                divs = [1]
                for p, e in factors.items():
                    e2 = 2 * e
                    new = []
                    for dv in divs:
                        for i in range(e2 + 1):
                            new.append(dv * (p ** i))
                    divs = new
                return divs
            divs = divisors_of_square(B)
            for d in divs:
                if d >= B2:
                    continue
                # y = (d+B)/A must be integer
                if (d + B) % A != 0:
                    continue
                y = (d + B) // A
                if y <= x:
                    continue
                # z = (B2/d + B)/A
                num_z = B2 // d + B
                if num_z % A != 0:
                    continue
                z = num_z // A
                if z <= y:
                    continue
                if z > MAX_DENOM:
                    continue
                found = [x, y, z]
                break
            if found:
                break
        answers[idx] = found

    return answers
