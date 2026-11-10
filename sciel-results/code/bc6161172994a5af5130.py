import numpy as np

def nikodym_set() -> np.ndarray:
    p = 7
    q = p * p
    alpha_sq = 3  # alpha^2 = 3 in F_49

    def fmul(xa, xb, ya, yb):
        # (xa + xb*a) * (ya + yb*a) where a^2 = alpha_sq
        r0 = (xa * ya + xb * yb * alpha_sq) % p
        r1 = (xa * yb + xb * ya) % p
        return r0, r1

    def fadd(xa, xb, ya, yb):
        return (xa + ya) % p, (xb + yb) % p

    def finv(xa, xb):
        if xa == 0 and xb == 0:
            return 0, 0
        norm = (xa * xa - xb * xb * alpha_sq) % p
        norm_inv = pow(norm, p - 2, p)
        return (xa * norm_inv) % p, (-xb * norm_inv) % p

    # Precompute all field elements and inverses
    elems = [(a, b) for a in range(p) for b in range(p)]
    inv_table = {}
    for a, b in elems:
        inv_table[(a, b)] = finv(a, b)

    # Build N: all points EXCEPT those on the parabola {(x, x^2)}
    # Parabola has 49 points, so |N| = 2401 - 49 = 2352

    # Precompute x^2 for each field element
    squares = {}
    for a, b in elems:
        squares[(a, b)] = fmul(a, b, a, b)

    # Build the set of excluded points (the parabola)
    excluded = set()
    for a, b in elems:
        sa, sb = squares[(a, b)]
        excluded.add((a, b, sa, sb))

    # Build N: all points not in excluded
    N = []
    for xa, xb in elems:
        for ya, yb in elems:
            if (xa, xb, ya, yb) not in excluded:
                N.append([xa, xb, ya, yb])

    return np.array(N, dtype=np.int64)
