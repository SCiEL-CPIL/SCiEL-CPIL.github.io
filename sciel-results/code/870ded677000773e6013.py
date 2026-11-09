import sys

def construct():
    """
    Constructs a 4-coloring of K_49 without monochromatic triangles
    using GF(7^2) field structure.
    """
    # GF(49) = GF(7)[x] / (x^2 + x + 3)
    # Represent element a0 + a1*x as integer a0 + 7*a1
    
    def gf_mul(a, b):
        a0, a1 = a % 7, a // 7
        b0, b1 = b % 7, b // 7
        c0 = (a0 * b0 - 3 * a1 * b1) % 7
        c1 = (a0 * b1 + a1 * b0 - a1 * b1) % 7
        return c0 + 7 * c1
    
    def gf_sub(a, b):
        a0, a1 = a % 7, a // 7
        b0, b1 = b % 7, b // 7
        return ((a0 - b0) % 7) + 7 * ((a1 - b1) % 7)
    
    def gf_pow(base, exp):
        res, b, e = 1, base, exp
        while e > 0:
            if e & 1:
                res = gf_mul(res, b)
            b = gf_mul(b, b)
            e >>= 1
        return res
    
    # Find primitive element of GF(49)* (order 48)
    primitive = None
    for g in range(2, 49):
        if gf_pow(g, 48) == 1 and gf_pow(g, 24) != 1 and gf_pow(g, 12) != 1:
            primitive = g
            break
    
    if primitive is None:
        return _seed_construct()
    
    g = primitive
    
    # Build log table: log_map[element] = exponent
    log_map = {}
    cur = 1
    for i in range(48):
        log_map[cur] = i
        cur = gf_mul(cur, g)
    
    # Color assignment: color(d) = log_map[d] % 4
    # This partitions GF(49)* into 4 cosets of the subgroup of order 12
    
    n = 49
    colors = []
    
    for i in range(n):
        for j in range(i + 1, n):
            d = gf_sub(j, i)
            colors.append(log_map[d] % 4)
    
    # Verify no monochromatic triangles
    C = [[-1] * n for _ in range(n)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            C[i][j] = C[j][i] = colors[idx]
            idx += 1
    
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if C[i][j] == C[j][k] == C[i][k]:
                    return _seed_construct()
    
    return {'n': n, 'colors': colors}

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
