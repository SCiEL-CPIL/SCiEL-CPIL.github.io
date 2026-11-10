import time
from math import gcd

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 82
    answers = []

    for n in range(3, 257):
        if time.time() > deadline:
            answers.append(None)
            continue

        found = None
        x_min = n // 4 + 1
        x_max = (3 * n) // 4

        for x in range(x_min, x_max + 1):
            if time.time() > deadline:
                break

            d = 4 * x - n
            if d <= 0:
                continue

            nx = n * x
            N = nx * nx
            target_mod = (-nx) % d

            # Enumerate divisors a of N with a < nx and a ≡ target_mod (mod d)
            # For each such a, b = N/a, and y = (a+nx)/d, z = (b+nx)/d
            # We need a < b (i.e., a < sqrt(N) = nx) to ensure y < z

            a = target_mod
            if a == 0:
                a = d
            # Make sure a is in [1, d-1] or handle a=0 case
            # Actually a must satisfy a % d == target_mod and a >= 1
            # a values: target_mod, target_mod + d, target_mod + 2d, ...
            # But if target_mod == 0, a = d, 2d, 3d, ...

            if target_mod == 0:
                a = d
            else:
                a = target_mod

            while a < nx:
                if N % a == 0:
                    b = N // a
                    if b < a:
                        break
                    y_num = a + nx
                    z_num = b + nx
                    if y_num % d == 0 and z_num % d == 0:
                        y = y_num // d
                        z = z_num // d
                        if y > x and z > y:
                            if x != y and y != z and x != z:
                                if z <= 10**12:
                                    # Verify: 1/x + 1/y + 1/z = 4/n
                                    # n*y*z + n*x*z + n*x*y = 4*x*y*z
                                    lhs = n * y * z + n * x * z + n * x * y
                                    rhs = 4 * x * y * z
                                    if lhs == rhs:
                                        found = [x, y, z]
                                        break
                a += d
                if time.time() > deadline:
                    break

            if found:
                break

        answers.append(found)

    return answers
# EVOLVE-BLOCK-END
