import time
from math import gcd

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85
    answers = []

    for n in range(3, 257):
        if time.time() > deadline:
            answers.append(None)
            continue

        found = None
        # x must satisfy: 1/x < 4/n => x > n/4, and 3/x >= 4/n => x <= 3n/4
        # Also x must be such that d = 4x - n > 0, i.e., x > n/4
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

            # We need a | N, a < nx, a ≡ target_mod (mod d)
            # Then b = N/a, y = (a+nx)/d, z = (b+nx)/d
            # Check y_num % d == 0 and z_num % d == 0

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
                                    # Verify by cross multiplication
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
