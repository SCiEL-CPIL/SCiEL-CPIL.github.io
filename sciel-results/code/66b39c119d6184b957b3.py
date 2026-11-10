"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 82.0
    LIMIT = 10**12
    answers = [None] * 254

    for n in range(3, 257):
        if time.time() > deadline:
            break
        found = None
        x_min = n // 4 + 1
        x_max = 3 * n // 4
        
        for x in range(x_min, x_max + 1):
            if time.time() > deadline:
                break
            A = 4 * x - n
            if A <= 0:
                continue
            B = n * x
            
            # 1/y + 1/z = A/B => (Ay-B)(Az-B) = A^2 * B
            # d = Ay - B must divide A^2 * B, and d ≡ -B (mod A)
            # y = (d+B)/A, z = (A^2*B/d + B)/A
            # Need y > x, z > y, all distinct, <= LIMIT
            
            A2B = A * A * B
            g = gcd(A, B)
            # d must be of form k*A - B for some k, and d | A^2*B
            # Equivalently, let d = A*k - B, then A*k - B | A^2*B
            # => A*k - B | A^2*B + B*(A*k - B) = A^2*B + A*B*k - B^2 = A*B*(A+k) - B^2
            # This is getting complex. Let's just loop over y but be smarter.
            
            y_min = max(x + 1, B // A + 1)
            if (2 * B) % A == 0:
                y_max = (2 * B) // A - 1
            else:
                y_max = (2 * B) // A
            
            if y_min > y_max:
                continue
            
            # Try y values, checking divisibility
            for y in range(y_min, y_max + 1):
                denom = A * y - B
                if denom <= 0:
                    continue
                if (B * y) % denom == 0:
                    z = (B * y) // denom
                    if z > y and z <= LIMIT:
                        if x != y and y != z and x != z:
                            found = [x, y, z]
                            break
            if found:
                break
        
        if not found and time.time() < deadline - 2.0:
            # Fallback: wider search with different x ordering
            for x in range(x_max, x_min - 1, -1):
                if time.time() > deadline:
                    break
                A = 4 * x - n
                if A <= 0:
                    continue
                B = n * x
                y_min = max(x + 1, B // A + 1)
                if (2 * B) % A == 0:
                    y_max = (2 * B) // A - 1
                else:
                    y_max = (2 * B) // A
                if y_min > y_max:
                    continue
                for y in range(y_min, y_max + 1):
                    if time.time() > deadline:
                        break
                    denom = A * y - B
                    if denom <= 0:
                        continue
                    if (B * y) % denom == 0:
                        z = (B * y) // denom
                        if z > y and z <= LIMIT:
                            if x != y and y != z and x != z:
                                found = [x, y, z]
                                break
                if found:
                    break
        
        answers[n - 3] = found

    return answers
# EVOLVE-BLOCK-END
