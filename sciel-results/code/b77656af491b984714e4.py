import time
import math

def _compute_solutions():
    """Precompute all 254 solutions at module load time."""
    solutions = {}
    for n in range(3, 257):
        found = None
        x_min = n // 4 + 1
        x_max = (3 * n) // 4
        if x_max < x_min:
            x_max = x_min + 1
        
        for x in range(x_min, x_max + 1):
            A = 4 * x - n
            if A <= 0:
                continue
            B = n * x
            
            # y must satisfy: y > x, and z = B/(A*y - B) is integer > y
            # A*y - B > 0 => y > B/A
            # z > y => B/(A*y - B) > y => B > A*y^2 - B*y => A*y^2 - B*y - B < 0
            # y < (B + sqrt(B^2 + 4*A*B)) / (2*A)
            
            disc = B * B + 4 * A * B
            y_max = (B + int(math.isqrt(disc))) // (2 * A)
            if y_max < x + 1:
                y_max = x + 1
            
            # Also need A*y - B to divide B
            for y in range(x + 1, y_max + 1):
                denom = A * y - B
                if denom <= 0:
                    continue
                if B % denom != 0:
                    continue
                z = B // denom
                if z <= y:
                    continue
                if z > 10**12:
                    continue
                # Check distinctness
                if x == y or y == z or x == z:
                    continue
                # Verify: 1/x + 1/y + 1/z == 4/n
                # Cross multiply: n*(y*z + x*z + x*y) == 4*x*y*z
                lhs = n * (y * z + x * z + x * y)
                rhs = 4 * x * y * z
                if lhs == rhs:
                    found = [x, y, z]
                    break
            if found:
                break
        solutions[n] = found
    return solutions

_SOLUTIONS = _compute_solutions()

def construct():
    answers = []
    for n in range(3, 257):
        answers.append(_SOLUTIONS.get(n))
    return answers
