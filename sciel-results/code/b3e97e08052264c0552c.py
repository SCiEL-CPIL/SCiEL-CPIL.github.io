"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from math import gcd

# EVOLVE-BLOCK-START
def construct():
    answers = []
    LIMIT = 10**12
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill

    for n in range(3, 257):
        if time.time() - start_time > DEADLINE:
            answers.append(None)
            continue

        found = None
        # x must satisfy: 1/x < 4/n => x > n/4, and 3/x > 4/n => x < 3n/4
        # Also x >= 1, and we want x < y < z for ordering
        start_x = n // 4 + 1
        end_x = (3 * n) // 4  # inclusive upper bound for x

        for x in range(start_x, end_x + 1):
            if time.time() - start_time > DEADLINE:
                break

            # 1/y + 1/z = 4/n - 1/x = (4x - n) / (n*x)
            A = 4 * x - n
            B = n * x
            if A <= 0:
                continue

            # Reduce A/B to lowest terms
            g = gcd(A, B)
            A_r = A // g
            B_r = B // g

            # We need (A_r * y - B_r)(A_r * z - B_r) = B_r^2
            # Let d be a divisor of B_r^2, then:
            #   y = (d + B_r) / A_r, z = (B_r^2 // d + B_r) / A_r
            # Both must be positive integers, y != z, y != x, z != x, and <= LIMIT

            B_sq = B_r * B_r

            # Iterate over divisors d of B_sq where d <= B_sq (and we'll get pairs)
            # To keep y < z, we need d < B_sq (since d = A_r*y - B_r and B_sq/d = A_r*z - B_r)
            # Actually let's just iterate all divisors and check

            # Find all divisors of B_sq
            # B_r can be large, so we need efficient factorization
            # But B_r = n*x/g, and n <= 256, x <= 3*256/4 = 192, so B_r <= 256*192 = 49152
            # B_sq <= 49152^2 ≈ 2.4e9, which is manageable for trial division

            # Actually let me just iterate d from 1 to sqrt(B_sq) and check
            # But B_sq can be up to ~2.4e9, sqrt is ~49000, that's fine

            found_pair = False
            # We need d such that (d + B_r) % A_r == 0 and (B_sq // d + B_r) % A_r == 0
            # and d * (B_sq // d) == B_sq (i.e., d divides B_sq)
            # and y = (d+B_r)//A_r > 0, z = (B_sq//d+B_r)//A_r > 0
            # and y != z, y != x, z != x, y <= LIMIT, z <= LIMIT

            # To avoid duplicates, assume d <= B_sq // d, so y <= z
            # Iterate d from 1 to int(sqrt(B_sq))
            import math
            sqrt_Bsq = int(math.isqrt(B_sq))

            for d in range(1, sqrt_Bsq + 1):
                if B_sq % d != 0:
                    continue

                d1 = d
                d2 = B_sq // d

                # y from d1: y = (d1 + B_r) / A_r
                if (d1 + B_r) % A_r != 0:
                    continue
                y = (d1 + B_r) // A_r

                # z from d2: z = (d2 + B_r) / A_r
                if (d2 + B_r) % A_r != 0:
                    continue
                z = (d2 + B_r) // A_r

                # Check validity
                if y <= 0 or z <= 0:
                    continue
                if y > LIMIT or z > LIMIT:
                    continue
                if y == z:
                    continue
                if y == x or z == x:
                    continue

                # Ensure x < y < z for consistent ordering (or just check distinctness)
                # The problem says DISTINCT, so as long as all three are different, it's fine
                # But let's verify the equation holds via cross multiplication
                # 1/x + 1/y + 1/z = 4/n
                # Check: y*z*n + x*z*n + x*y*n == 4*x*y*z
                lhs = y * z * n + x * z * n + x * y * n
                rhs = 4 * x * y * z
                if lhs != rhs:
                    continue

                found = [x, y, z]
                found_pair = True
                break

            if found_pair:
                break

        answers.append(found)

    return answers
# EVOLVE-BLOCK-END
