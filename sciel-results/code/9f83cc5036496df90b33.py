# EVOLVE-BLOCK-START
import numpy as np


def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2 where the complement S is a single
    line, which has q = 49 points.
    For each point s in S, the line S itself intersects S at s and 48 other
    points, but we can use a different line through s that intersects S only at s.
    Actually, for a line S, for s in S, we need a line through s that intersects
    S only at s. Any line through s not equal to S works, since two distinct
    lines in a plane intersect in exactly one point.
    For x not in S, any line through x not equal to S works, since it intersects
    S in at most one point (not x, since x not in S), so L \\ {x} contains no points of S.
    So S = any line is a valid complement.
    |S| = 49, so |N| = 2401 - 49 = 2352.
    """
    p = 7
    k = 3  # alpha^2 = 3 in F_7

    # Field operations for F_49 = F_7[alpha], alpha^2 = k
    def add(x, y):
        return ((x[0] + y[0]) % p, (x[1] + y[1]) % p)

    def sub(x, y):
        return ((x[0] - y[0]) % p, (x[1] - y[1]) % p)

    def mul(x, y):
        a, b = x
        c, d = y
        # (a + b*alpha)(c + d*alpha) = (ac + bd*k) + (ad + bc)*alpha
        return ((a * c + b * d * k) % p, (a * d + b * c) % p)

    def neg(x):
        return ((-x[0]) % p, (-x[1]) % p)

    def inv(x):
        # x = (a, b), x^{-1} = (a, -b) / (a^2 + b^2*k)
        a, b = x
        denom = (a * a + b * b * k) % p
        if denom == 0:
            raise ZeroDivisionError("Cannot invert zero")
        inv_denom = pow(denom, p - 2, p)
        return ((a * inv_denom) % p, (-b * inv_denom) % p)

    # Enumerate all field elements
    field_elements = [(a, b) for a in range(p) for b in range(p)]

    # Choose S to be the x-axis: y = 0
    # S = {(x, 0) : x in F_49}
    S = set()
    zero = (0, 0)
    for x in field_elements:
        S.add((x, zero))

    # N = all points not in S
    points = []
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    x = (xa, xb)
                    y = (ya, yb)
                    if (x, y) not in S:
                        points.append([[xa, xb], [ya, yb]])

    return np.array(points, dtype=np.int64)


# EVOLVE-BLOCK-END
