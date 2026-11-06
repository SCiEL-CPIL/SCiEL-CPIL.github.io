def construct():
    # GF(25) implementation
    p = 5
    # Polynomial x^2 + x + 2 is irreducible over F5
    # alpha^2 = -alpha - 2 = 4*alpha + 3 (in F5: -1=4, -2=3)
    # Wait, previous code used x^2 + x + 2.
    # Let's use a standard irreducible polynomial. x^2 + 2 is irreducible?
    # Discriminant 0-4 = -4 = 1. Square. So reducible.
    # x^2 + 1? Disc -4 = 1. Reducible.
    # x^2 + 2x + 2? Disc 4-8 = -4 = 1. Reducible.
    # x^2 + x + 2? Disc 1-8 = -7 = 3. 3 is not a square in F5 (squares: 0,1,4). Irreducible.
    # So alpha^2 = -alpha - 2 = 4*alpha + 3.
    
    def gf_mul(a, b):
        x1, y1 = a
        x2, y2 = b
        # (x1 + y1*alpha)(x2 + y2*alpha) = x1x2 + (x1y2 + x2y1)alpha + y1y2*alpha^2
        # alpha^2 = 3 + 4*alpha
        # = x1x2 + (x1y2 + x2y1)alpha + y1y2(3 + 4*alpha)
        # = (x1x2 + 3*y1y2) + (x1y2 + x2y1 + 4*y1y2)alpha
        const_part = (x1 * x2 + 3 * y1 * y2) % p
        alpha_part = (x1 * y2 + x2 * y1 + 4 * y1 * y2) % p
        return (const_part, alpha_part)

    def gf_add(a, b):
        return ((a[0] + b[0]) % p, (a[1] + b[1]) % p)

    def gf_neg(a):
        return ((-a[0]) % p, (-a[1]) % p)

    elems = [(i, j) for i in range(p) for j in range(p)]
    
    # Find generator
    # Try (1,1)
    g = (1, 1)
    # Check if g is primitive (order 24)
    # g^12 != 1, g^8 != 1, g^6 != 1
    def gf_pow(base, exp):
        result = (1, 0)
        base = base
        while exp:
            if exp & 1:
                result = gf_mul(result, base)
            base = gf_mul(base, base)
            exp >>= 1
        return result

    if gf_pow(g, 12) == (1, 0) or gf_pow(g, 8) == (1, 0) or gf_pow(g, 6) == (1, 0):
        # Try other generators
        for e in elems:
            if e == (0,0) or e == (1,0): continue
            if gf_pow(e, 12) != (1, 0) and gf_pow(e, 8) != (1, 0) and gf_pow(e, 6) != (1, 0):
                g = e
                break

    # Generate cubic residues H
    H = set()
    for k in range(8):
        H.add(gf_pow(g, 3 * k))
        
    # Check if H is sum-free (triangle-free condition for Cayley graph)
    # No a, b in H such that -(a+b) in H
    # Since -1 is in H (check: g^8? g^24=1, g^8 != 1. -1 = g^12. 12 is multiple of 3? Yes. So -1 in H)
    # So we need a+b not in H for all a,b in H
    is_sum_free = True
    for a in H:
        for b in H:
            s = gf_add(a, b)
            if s in H:
                is_sum_free = False
                break
        if not is_sum_free:
            break
            
    if is_sum_free:
        # Construct 25-vertex coloring
        # Colors: 0 for H, 1 for g*H, 2 for g^2*H
        H1 = set(gf_mul(g, h) for h in H)
        H2 = set(gf_mul(gf_pow(g, 2), h) for h in H)
        
        # Verify partition
        if len(H | H1 | H2) != 24:
            is_sum_free = False # Fallback
        
    if is_sum_free:
        colors_25 = []
        # Precompute indices for speed
        # Pairs (i, j) with i < j
        # Index in flat list for (i, j) is sum_{k=0}^{i-1} (25-1-k) + (j-i-1)
        # sum_{k=0}^{i-1} (24-k) = 24*i - i*(i-1)//2
        for i in range(25):
            for j in range(i + 1, 25):
                diff = gf_add(elems[j], gf_neg(elems[i]))
                if diff in H:
                    col = 0
                elif diff in H1:
                    col = 1
                else:
                    col = 2
                colors_25.append(col)
                
        # Double to 50
        colors = []
        for i in range(50):
            for j in range(i + 1, 50):
                if i < 25 and j < 25:
                    # Index in colors_25
                    # i, j are 0..24
                    idx = 24 * i - i * (i - 1) // 2 + (j - i - 1)
                    colors.append(colors_25[idx])
                elif i >= 25 and j >= 25:
                    i2 = i - 25
                    j2 = j - 25
                    idx = 24 * i2 - i2 * (i2 - 1) // 2 + (j2 - i2 - 1)
                    colors.append(colors_25[idx])
                else:
                    colors.append(3)
                    
        return {'n': 50, 'colors': colors}
        
    # Fallback to 32-vertex construction (Paley-like or known)
    # The previous fallback was 32. Let's keep it.
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
