<<<<<<< SEARCH
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    deadline = time.time() + 118.0
    rng = random.Random(12345)

    # ---------- GF(16) arithmetic (Greenwood-Gleason base colouring) ----------
    def mul(a, b):
        r = 0
        for i in range(4):
            if (b >> i) & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if (r >> i) & 1:
                r ^= 0b10011 << (i - 4)
        return r

    log, x = {}, 1
    for e in range(15):
        log[x] = e
        x = mul(x, 2)

    def base_color(i, j):
        if i // 16 == j // 16:
            return log[(i % 16) ^ (j % 16)] % 3
        return 3

    # ---------- initial 32-vertex colouring (two copies + bipartite colour 3) ----------
    n0 = 32
    mat0 = [[-1] * n0 for _ in range(n0)]
    for i in range(n0):
        for j in range(i + 1, n0):
            c = base_color(i, j)
            mat0[i][j] = c
            mat0[j][i] = c
=======
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    deadline = time.time() + 118.0
    rng = random.Random(12345)

    # ---------- GF(49) arithmetic (4-coloring via cubic residues) ----------
    # GF(49) = GF(7)[x]/(x^2 + 3) since x^2+3 is irreducible mod 7
    # (check: -3 = 4 is not a square mod 7; squares are 0,1,2,4)
    
    def gf49_add(a, b):
        # a, b are (a0, a1) representing a0 + a1*x
        return ((a[0] + b[0]) % 7, (a[1] + b[1]) % 7)

    def gf49_neg(a):
        return ((-a[0]) % 7, (-a[1]) % 7)

    def gf49_mul(a, b):
        # (a0 + a1*x)(b0 + b1*x) = a0*b0 + (a0*b1+a1*b0)*x + a1*b1*x^2
        # x^2 = -3, so x^2 -> (4, 0) since -3 mod 7 = 4
        a0, a1 = a
        b0, b1 = b
        c0 = (a0 * b0 + a1 * b1 * 4) % 7  # x^2 = -3 = 4 mod 7
        c1 = (a0 * b1 + a1 * b0) % 7
        return (c0, c1)

    def gf49_pow(a, e):
        result = (1, 0)
        base = a
        while e > 0:
            if e & 1:
                result = gf49_mul(result, base)
            base = gf49_mul(base, base)
            e >>= 1
        return result

    # Generate all 49 elements of GF(49)
    elements = []
    for i in range(7):
        for j in range(7):
            elements.append((i, j))

    # Find a primitive element (generator of multiplicative group of order 48)
    # Try (1,1): check if it generates the full group
    def find_primitive():
        for elem in elements:
            if elem == (0, 0):
                continue
            # Check if elem^48 = 1 and elem^(48/p) != 1 for primes p|48
            # 48 = 2^4 * 3, so check elem^24, elem^16
            if gf49_pow(elem, 48) != (1, 0):
                continue
            if gf49_pow(elem, 24) == (1, 0):
                continue
            if gf49_pow(elem, 16) == (1, 0):
                continue
            return elem
        return None

    g = find_primitive()
    
    # Build log table: log_g[elem] = e where g^e = elem
    log_g = {}
    x = (1, 0)
    for e in range(48):
        log_g[x] = e
        x = gf49_mul(x, g)
    log_g[(0, 0)] = -1  # special case

    # Cubic residues: elements whose discrete log is divisible by 3
    # Group of order 48, so cubic residues have log in {0,3,6,...,45} (16 elements)
    # Non-cubic-residues have log in {1,2,4,5,7,8,...} (32 elements)
    # We need 3 sets of 16 for colors 0,1,2, and handle 0 specially
    
    # Actually, 48/3 = 16, so there are 16 cubic residues (including 1)
    # The remaining 32 non-zero non-residues need to be split into 2 sets of 16
    # Color assignment based on discrete log mod 3:
    #   log % 3 == 0 -> color 0 (cubic residues, 16 elements)
    #   log % 3 == 1 -> color 1 (16 elements)  
    #   log % 3 == 2 -> color 2 (16 elements)
    #   zero difference -> color 3 (but i != j so this never happens)
    
    def gf49_color(diff):
        """Assign color based on the discrete log of diff in GF(49)*"""
        e = log_g.get(diff, -1)
        if e == -1:
            return 3  # should not happen for i != j
        return e % 3

    # ---------- initial 49-vertex colouring ----------
    n0 = 49
    mat0 = [[-1] * n0 for _ in range(n0)]
    for i in range(n0):
        for j in range(i + 1, n0):
            diff = gf49_add(elements[i], gf49_neg(elements[j]))
            c = gf49_color(diff)
            mat0[i][j] = c
            mat0[j][i] = c
>>>>>>> REPLACE

