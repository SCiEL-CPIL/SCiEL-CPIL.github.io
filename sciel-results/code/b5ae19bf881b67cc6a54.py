import hashlib
import struct

def construct():
    n = 32
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            if i < 16 and j < 16:
                diff = i ^ j
                if diff == 0:
                    c = 0
                else:
                    # Compute log2 in GF(16) with poly x^4+x+1 (0b10011)
                    # Actually let's use a precomputed table for 4-bit values
                    # GF(16) with primitive polynomial 0b10011 (x^4+x+1)
                    # Multiplication table for log
                    # We'll compute log[diff] % 3 for diff in 1..15
                    # Let's precompute
                    pass
            elif i >= 16 and j >= 16:
                vi = i - 16
                vj = j - 16
                diff = vi ^ vj
                if diff == 0:
                    c = 0
                else:
                    pass
            else:
                c = 3
            # We need a proper coloring. Let's use a simpler known construction.
            # Actually, the current program's approach was to use the 16-block.
            # Let's just use a simple valid construction: 
            # For n=32, use two copies of a 16-vertex 3-coloring (colors 0,1,2) 
            # with inter-block edges colored 3.
            # The 16-vertex coloring: use GF(16) as before.
            # But I need to make sure it's valid.
            # Let's just hardcode a known valid 32-vertex coloring.
            # Actually, let's use the simplest possible: 
            # Color edge (i,j) based on (i+j) % 4? No, that creates mono triangles.
            # 
            # Let me use the construction from the seed: 
            # Block 0: vertices 0-15, Block 1: vertices 16-31
            # Intra-block: use GF(16) coloring with 3 colors
            # Inter-block: color 3
            #
            # I need to implement the GF(16) coloring correctly.
            # Let's precompute the log table.
            pass
    # Let me rebuild this properly.
    n = 32
    colors = []
    
    # Precompute GF(16) log table
    # Primitive polynomial: x^4 + x + 1 = 0b10011
    # Multiplication in GF(16)
    def gf_mul(a, b):
        r = 0
        while b:
            if b & 1:
                r ^= a
            b >>= 1
            a <<= 1
            if a & 0b10000:
                a ^= 0b10011
        return r & 0b1111
    
    # Build log table: log[g^k] = k for g=2 (primitive element)
    log_table = {}
    x = 1
    for k in range(15):
        log_table[x] = k
        x = gf_mul(x, 2)
    
    def block_color(a, b):
        diff = a ^ b
        if diff == 0:
            return 0
        return log_table[diff] % 3
    
    for i in range(n):
        for j in range(i + 1, n):
            bi = i // 16
            bj = j // 16
            vi = i % 16
            vj = j % 16
            if bi == bj:
                c = block_color(vi, vj)
            else:
                c = 3
            colors.append(c)
    
    return {'n': n, 'colors': colors}
