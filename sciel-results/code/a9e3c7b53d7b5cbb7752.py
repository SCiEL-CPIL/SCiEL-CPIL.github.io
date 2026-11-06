import time
import math

def construct():
    deadline = time.time() + 78.0
    answers = [None] * 254
    BOUND = 10**12

    for n in range(3, 257):
        idx = n - 3
        if time.time() > deadline:
            break
        found = None

        x_start = n // 4 + 1
        # Try a generous range of x values
        x_end = min(4 * n + 100, 5000)

        for x in range(x_start, x_end + 1):
            if time.time() > deadline:
                break
            if found:
                break

            A = 4 * x - n
            if A <= 0:
                continue
            B = n * x

            # Method 1: k-based search
            # From algebra: y = k, z = B*k/(k*A - B)
            # Need k > B/A, and (k*A - B) | B*k
            # Upper bound on k: z > y => B*k/(k*A-B) > k => B/(k*A-B) > 1 => k*A - B < B => k*A < 2B => k < 2B/A
            # Also z != x, z != y, all distinct
            k_min = B // A + 1
            k_max = min(2 * B // A - 1, k_min + 500000)  # cap for safety

            # Optimize: iterate k and check divisibility
            # (k*A - B) | B*k
            # Let d = k*A - B, then k = (d+B)/A, need d | B*(d+B)/A
            # Equivalent: d | B*(d+B) and A | d+B
            # Since d = k*A - B, d ≡ -B (mod A), so d+B ≡ 0 (mod A) always
            # So we just need d | B*(d+B) where d = k*A - B > 0
            # B*(d+B) = B*d + B^2, so d | B^2
            # Wait: d | B*(d+B) iff d | B^2 (since d | B*d trivially)
            # So we need d | B^2 where d = k*A - B
            # This means for each k, check if (k*A - B) divides B^2

            found_in_k = False
            # For efficiency, only check k where (k*A - B) could divide B^2
            # (k*A - B) must be a divisor of B^2
            # Let's iterate k and check
            for k in range(k_min, k_max + 1):
                d = k * A - B
                if d <= 0:
                    continue
                # Check if d divides B^2
                # B^2 can be large, so use modular arithmetic
                # d | B^2 iff B^2 % d == 0
                # B^2 mod d = (B mod d)^2 mod d
                B_mod_d = B % d
                if (B_mod_d * B_mod_d) % d != 0:
                    continue
                # z = B*k / d
                z = B * k // d
                if z <= k or z == x or z > BOUND:
                    continue
                # Verify: 1/x + 1/k + 1/z == 4/n
                # Cross multiply: n*(k*z + x*z + x*k) == 4*x*k*z
                lhs = n * (k * z + x * z + x * k)
                rhs = 4 * x * k * z
                if lhs == rhs:
                    found = [x, k, z]
                    found_in_k = True
                    break

            if found_in_k:
                break

            # Method 2: Direct y-scan (faster for some cases)
            # y > B/A, y < 2B/A (for z > y)
            y_min = B // A + 1
            y_max = min(2 * B // A - 1, y_min + 100000)

            for y in range(y_min, y_max + 1):
                denom = A * y - B
                if denom <= 0:
                    continue
                if (B * y) % denom == 0:
                    z = (B * y) // denom
                    if z > y and z != x and z != y and z <= BOUND:
                        # Verify
                        lhs = n * (y * z + x * z + x * y)
                        rhs = 4 * x * y * z
                        if lhs == rhs:
                            found = [x, y, z]
                            break
            if found:
                break

        # Method 3: Divisor enumeration fallback for remaining n
        if not found:
            # Try divisor enumeration for a few x values
            for x in range(x_start, min(x_start + 15, x_end + 1)):
                if time.time() > deadline:
                    break
                if found:
                    break

                A = 4 * x - n
                if A <= 0:
                    continue
                B = n * x
                B_sq = B * B

                # Find divisors of B^2 that are ≡ -B (mod A)
                # Iterate over divisors of B, form u = d1*d2 where d1|B, d2|B, u*d1*d2 = B^2
                # Simpler: iterate u from 1 to B, check if u|B^2 and (u+B)%A==0
                # But B can be up to ~256*5000 = 1.28e6, so B^2 ~ 1.6e12
                # Too slow to iterate all. Instead use factorization.

                # Factorize B
                def _factorize(num):
                    factors = {}
                    d = 2
                    while d * d <= num:
                        while num % d == 0:
                            factors[d] = factors.get(d, 0) + 1
                            num //= d
                        d += 1
                    if num > 1:
                        factors[num] = factors.get(num, 0) + 1
                    return factors

                def _divisors_from_factorization(factors):
                    divisors = [1]
                    for prime, exp in factors.items():
                        new_divisors = []
                        power = 1
                        for e in range(exp + 1):
                            for d in divisors:
                                new_divisors.append(d * power)
                            power *= prime
                        divisors = new_divisors
                    return divisors

                factors_B = _factorize(B)
                factors_B_sq = {p: 2 * e for p, e in factors_B.items()}
                divs = _divisors_from_factorization(factors_B_sq)

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
                    if y > BOUND or z > BOUND or x > BOUND:
                        continue
                    if x == y or x == z or y == z:
                        continue

                    lhs = n * (y * z + x * z + x * y)
                    rhs = 4 * x * y * z
                    if lhs == rhs:
                        found = [x, y, z]
                        break
                if found:
                    break

        answers[idx] = found

    return answers
