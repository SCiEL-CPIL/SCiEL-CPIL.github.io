import time
import random

def construct():
    t0 = time.time()
    deadline = t0 + 140.0

    # --- Helper: Check if a colouring is valid (no monochromatic triangles) ---
    def is_valid(n, colors):
        # Build adjacency matrix for each colour
        # adj[c][i][j] = 1 if edge (i,j) is colour c
        adj = [[0] * n for _ in range(n)] # Store colour directly
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[i][j] = c
                adj[j][i] = c
                idx += 1
        
        # Check for triangles
        for c in range(4):
            for k in range(n):
                for i in range(k):
                    if adj[i][k] != c:
                        continue
                    for j in range(i):
                        if adj[i][j] == c and adj[j][k] == c:
                            return False
        return True

    # --- Construction 1: n=32 (Greenwood-Gleason K16 doubled) ---
    def construct_32():
        # GF(16) with polynomial x^4 + x + 1
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

        base16 = {}
        for i in range(16):
            for j in range(i + 1, 16):
                base16[(i, j)] = log[i ^ j] % 3

        n32 = 32
        adj = [[0] * n32 for _ in range(n32)]
        for i in range(n32):
            for j in range(i + 1, n32):
                if i // 16 == j // 16:
                    c = base16[(i % 16, j % 16)]
                else:
                    c = 3
                adj[i][j] = c
                adj[j][i] = c
        
        colors = []
        for i in range(n32):
            for j in range(i + 1, n32):
                colors.append(adj[i][j])
        return 32, colors

    # --- Construction 2: n=50 (Paley-type q=25) ---
    def construct_50():
        q = 25
        n = 2 * q
        # Compute quadratic residues mod 25
        qr = set()
        for x in range(q):
            qr.add((x * x) % q)
        
        def chi(a):
            a = a % q
            if a == 0:
                return 0
            return 1 if a in qr else -1

        colors = []
        # Generate colours for all pairs
        for i in range(n):
            for j in range(i + 1, n):
                i0 = i % q
                i1 = i // q
                j0 = j % q
                j1 = j // q
                
                d = (i0 - j0) % q
                c_val = chi(d)
                
                if i1 == j1:
                    # Internal edges
                    if c_val == 1:
                        c = 0 if i1 == 0 else 2
                    else:
                        c = 1 if i1 == 0 else 3
                else:
                    # Cross edges
                    if c_val == 1:
                        c = 0
                    else:
                        c = 1
                    # Note: This specific mapping (cross 0,1) might fail if P(25) has triangles.
                    # Alternative mapping: Cross edges 2,3?
                    # Let's try the standard one first.
                
                colors.append(c)
        
        return n, colors

    # --- Try n=50 first ---
    n50, colors50 = construct_50()
    if is_valid(n50, colors50):
        return {'n': n50, 'colors': colors50}

    # --- Fallback to n=32 ---
    n32, colors32 = construct_32()
    if is_valid(n32, colors32):
        return {'n': n32, 'colors': colors32}

    # --- Fallback to n=16 ---
    def construct_16():
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
        for i in range(16):
            for j in range(i + 1, 16):
                colors.append(log[i ^ j] % 3)
        return 16, colors

    n16, colors16 = construct_16()
    if is_valid(n16, colors16):
        return {'n': n16, 'colors': colors16}

    # Last resort: n=1
    return {'n': 1, 'colors': []}
