"""Direct construction of distinct three-term unit-fraction representations of 4/n."""
# EVOLVE-BLOCK-START
import time
import math

def _factorize(num):
    """Return all positive divisors of num."""
    if num <= 0:
        return []
    divs = []
    i = 1
    while i * i <= num:
        if num % i == 0:
            divs.append(i)
            if i != num // i:
                divs.append(num // i)
        i += 1
    return divs

def construct():
    deadline = time.time() + 85  # safety margin before the 90s kill
    answers = []
    BOUND = 10**12

    for n in range(3, 257):
        if time.time() > deadline:
            answers.append(None)
            continue

        found = None

        # x is the smallest denominator. Since 1/x is the largest term,
        # 1/x < 4/n => x > n/4. Also 3/x >= 4/n => x <= 3n/4.
        x_min = n // 4 + 1
        x_max = (3 * n) // 4

        for x in range(x_min, x_max + 1):
            if time.time() > deadline:
                break
            if found:
                break
            # 1/x + 1/y + 1/z = 4/n
            # 1/y + 1/z = 4/n - 1/x = (4x - n) / (nx)
            a = 4 * x - n          # positive since x > n/4
            b = n * x              # positive
            if a <= 0 or b <= 0:
                continue
            
            # (ay - b)(az - b) = b^2
            # Let u = ay - b, v = az - b, then u*v = b^2
            # y = (u + b) / a, z = (v + b) / a
            # Need u*v = b^2, u > 0, v > 0
            # y > x => (u+b)/a > x => u > ax - b
            # z > y => v > u (since z > y means (v+b)/a > (u+b)/a means v > u)
            # So we need u < v and u*v = b^2, u > ax - b, and a | (u+b), a | (v+b)
            
            b_sq = b * b
            # Find divisors u of b_sq such that u < sqrt(b_sq) and u > ax - b
            u_min = a * x - b + 1  # u > ax - b
            if u_min < 1:
                u_min = 1
            
            # We need u * v = b_sq with u < v, so u < sqrt(b_sq)
            sqrt_b_sq = math.isqrt(b_sq)
            if u_min > sqrt_b_sq:
                continue
            
            # Iterate over possible u values that divide b_sq
            # Instead of finding all divisors, iterate u from u_min to sqrt_b_sq
            # But that could be slow. Better: find all divisors of b_sq and filter.
            # For small b, this is fine. For large b, we need to be smarter.
            
            # Alternative: iterate u from u_min to min(sqrt_b_sq, some_limit)
            # and check if u divides b_sq
            # The range is sqrt_b_sq - u_min + 1
            
            # For efficiency, let's just iterate and check divisibility
            # sqrt_b_sq = b, so we iterate from u_min to b
            # This could be up to b iterations per x, which is too slow for large n.
            
            # Better approach: factorize b_sq and generate divisors
            # But b can be up to ~256*192 = ~49152, so b_sq ~ 2.4e9
            # Factorizing 2.4e9 is feasible but generating all divisors might be many.
            
            # Let's use the direct iteration but with a time check
            for u in range(u_min, sqrt_b_sq + 1):
                if time.time() > deadline:
                    break
                if b_sq % u != 0:
                    continue
                v = b_sq // u
                # Check a | (u + b) and a | (v + b)
                if (u + b) % a != 0:
                    continue
                if (v + b) % a != 0:
                    continue
                y = (u + b) // a
                z = (v + b) // a
                # Check distinctness and ordering
                if y <= x or z <= y:
                    continue
                if x == y or y == z or x == z:
                    continue
                if y > BOUND or z > BOUND:
                    continue
                # Verify via cross multiplication
                lhs_num = n * y * z + n * x * z + n * x * y
                rhs_num = 4 * x * y * z
                if lhs_num == rhs_num:
                    found = [x, y, z]
                    break
        answers.append(found)
    return answers
# EVOLVE-BLOCK-END
