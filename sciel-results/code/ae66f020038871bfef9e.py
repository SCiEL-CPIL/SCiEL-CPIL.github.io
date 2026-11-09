import sys

def construct():
    """
    Constructs a 4-coloring of K_49 using GF(49) algebraic structure.
    """
    
    def gf49_mul(a, b):
        a0 = a % 7
        a1 = a // 7
        b0 = b % 7
        b1 = b // 7
        
        c0 = (a0 * b0 - 3 * a1 * b1) % 7
        c1 = (a0 * b1 + a1 * b0 - a1 * b1) % 7
        
        return c0 + 7 * c1

    def gf49_sub(a, b):
        a0 = a % 7
        a1 = a // 7
        b0 = b % 7
        b1 = b // 7
        c0 = (a0 - b0) % 7
        c1 = (a1 - b1) % 7
        return c0 + 7 * c1

    def pow_gf49(base, exp):
        res = 1
        b = base
        e = exp
        while e > 0:
            if e % 2 == 1:
                res = gf49_mul(res, b)
            b = gf49_mul(b, b)
            e //= 2
        return res

    # Find primitive element of GF(49)
    primitive = None
    for g in range(2, 49):
        if pow_gf49(g, 48) == 1:
            if pow_gf49(g, 24) != 1 and pow_gf49(g, 16) != 1:
                primitive = g
                break
    
    if primitive is None:
        return _seed_construct()

    g = primitive
    
    # Generate powers and log map
    powers = [1]
    for i in range(1, 48):
        powers.append(gf49_mul(powers[-1], g))
        
    log_map = {p: i for i, p in enumerate(powers)}
    
    n = 49
    colors = []
    
    # Precompute color for each non-zero difference
    diff_color = [0] * 49 # Index 0 unused
    for d in range(1, 49):
        diff_color[d] = log_map[d] % 4

    # Construct colors list
    for i in range(n):
        for j in range(i + 1, n):
            d = gf49_sub(j, i)
            colors.append(diff_color[d])
            
    # Verify
    C = [[-1]*n for _ in range(n)]
    k = 0
    for i in range(n):
        for j in range(i+1, n):
            C[i][j] = colors[k]
            C[j][i] = colors[k]
            k += 1
            
    valid = True
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                if C[i][j] == C[j][k] == C[i][k]:
                    valid = False
                    break
            if not valid: break
        if not valid: break
        
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
