import sys

def construct():
    """
    Constructs a 4-coloring of K_49 using GF(7^2) algebraic structure.
    Uses the multiplicative group of order 48, partitioned into 4 cosets
    of the subgroup of order 12. Color(i,j) = log_g(j-i) mod 4.
    """
    
    # GF(49) = GF(7)[x] / (x^2 + x + 3)
    # Represent element as integer a0 + 7*a1 for (a0 + a1*x)
    
    def gf49_mul(a, b):
        a0, a1 = a % 7, a // 7
        b0, b1 = b % 7, b // 7
        # (a0 + a1*x)(b0 + b1*x) = a0*b0 + (a0*b1 + a1*b0)*x + a1*b1*x^2
        # x^2 = -x - 3 in GF(49)
        c0 = (a0 * b0 - 3 * a1 * b1) % 7
        c1 = (a0 * b1 + a1 * b0 - a1 * b1) % 7
        return c0 + 7 * c1

    def gf49_sub(a, b):
        a0, a1 = a % 7, a // 7
        b0, b1 = b % 7, b // 7
        c0 = (a0 - b0) % 7
        c1 = (a1 - b1) % 7
        return c0 + 7 * c1

    def gf49_pow(base, exp):
        res = 1  # multiplicative identity
        b = base
        e = exp
        while e > 0:
            if e & 1:
                res = gf49_mul(res, b)
            b = gf49_mul(b, b)
            e >>= 1
        return res

    # Find primitive element of GF(49)* (order 48)
    primitive = None
    for g in range(2, 49):
        if gf49_pow(g, 48) == 1:
            # Check proper divisors of 48: 24, 16, 12, 8, 6, 4, 3, 2
            if all(gf49_pow(g, d) != 1 for d in [24, 16, 12, 8, 6, 4, 3, 2]):
                primitive = g
                break

    if primitive is None:
        return _seed_construct()

    g = primitive

    # Build log table: log_map[value] = exponent
    log_map = {}
    x = 1
    for i in range(48):
        log_map[x] = i
        x = gf49_mul(x, g)

    # Verify all 48 nonzero elements are covered
    if len(log_map) != 48:
        return _seed_construct()

    n = 49
    colors = []

    # For each pair (i, j) with i < j, compute color = log(j - i) mod 4
    for i in range(n):
        for j in range(i + 1, n):
            d = gf49_sub(j, i)
            # d should be nonzero since i != j
            if d == 0:
                return _seed_construct()
            colors.append(log_map[d] % 4)

    # Verify no monochromatic triangles
    # Build adjacency matrix for verification
    C = [[-1] * n for _ in range(n)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            C[i][j] = colors[idx]
            C[j][i] = colors[idx]
            idx += 1

    valid = True
    for i in range(n):
        if not valid:
            break
        for j in range(i + 1, n):
            if not valid:
                break
            for k in range(j + 1, n):
                if C[i][j] == C[j][k] == C[i][k]:
                    valid = False
                    break

    if valid:
        return {'n': n, 'colors': colors}
    else:
        return _seed_construct()

def _seed_construct():
    def mul(a, b):
        r = 0
        for i in range(4):
            if b >> i & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if r >> i & 1:
                r ^= 0b10011 << (i - 4)
        return r
    log, x = {}, 1
    for e in range(15):
        log[x] = e
        x = mul(x, 2)
    colors = []
    for i in range(32):
        for j in range(i + 1, 32):
            colors.append(log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3)
    return {'n': 32, 'colors': colors}
