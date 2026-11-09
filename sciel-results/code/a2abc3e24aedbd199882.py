import time
from math import isqrt

def construct():
    answers = [None] * 254
    t0 = time.time()
    DEADLINE = 85.0  # safety margin under 90s wall-clock

    # Precompute smallest prime factor (SPF) table up to a reasonable limit
    # B = n*x, max n=256, max x < 3*256/4 = 192, so max B = 256*192 = 49152
    # But to be safe, let's go up to 100000
    MAX_B = 100000
    spf = list(range(MAX_B + 1))
    for i in range(2, isqrt(MAX_B) + 1):
        if spf[i] == i:
            for j in range(i * i, MAX_B + 1, i):
                if spf[j] == j:
                    spf[j] = i

    def factorize(num):
        """Return dict of prime -> exponent for num using SPF table."""
        if num <= 1:
            return {}
        factors = {}
        while num > 1:
            p = spf[num]
            e = 0
            while num % p == 0:
                num //= p
                e += 1
            factors[p] = e
        return factors

    def generate_divisors_from_factors(factors):
        """Generate all divisors from prime factorization dict."""
        divs = [1]
        for p, e in factors.items():
            new_divs = []
            for d in divs:
                val = d
                for _ in range(e + 1):
                    new_divs.append(val)
                    val *= p
            divs = new_divs
        return sorted(divs)

    for idx, n in enumerate(range(3, 257)):
        if time.time() - t0 > DEADLINE:
            break
        found = None
        x_min = n // 4 + 1
        x_max = 3 * n // 4

        for x in range(x_min, x_max + 1):
            if time.time() - t0 > DEADLINE:
                break
            A = 4 * x - n
            if A <= 0:
                continue
            B = n * x

            # We need (A*y - B)(A*z - B) = B^2
            # Let u = A*y - B, then u divides B^2
            # y = (u + B) / A must be integer
            # z = (B^2/u + B) / A must be integer
            # For y < z, we need u < B (since u * v = B^2 and u < v)

            # Factorize B
            if B > MAX_B:
                # Fallback: use trial division
                factors_B = {}
                temp = B
                d = 2
                while d * d <= temp:
                    if temp % d:
                        d += 1
                    else:
                        nf = 0
                        while temp % d == 0:
                            temp //= d
                            nf += 1
                        factors_B[d] = nf
                if temp > 1:
                    factors_B[temp] = factors_B.get(temp, 0) + 1
            else:
                factors_B = factorize(B)

            # B^2 has exponents doubled
            factors_B2 = {p: 2 * e for p, e in factors_B.items()}

            divs = generate_divisors_from_factors(factors_B2)

            for u in divs:
                if u >= B:
                    continue
                # y = (u + B) / A
                num_y = u + B
                if num_y % A != 0:
                    continue
                y = num_y // A
                if y <= x:
                    continue
                # z = (B^2 / u + B) / A
                v = (B * B) // u
                num_z = v + B
                if num_z % A != 0:
                    continue
                z = num_z // A
                if z <= y:
                    continue
                # Check distinctness
                if x == y or x == z or y == z:
                    continue
                # Denominator bound
                if x > 10**12 or y > 10**12 or z > 10**12:
                    continue
                found = [x, y, z]
                break

            if found:
                break

        answers[idx] = found

    return answers
