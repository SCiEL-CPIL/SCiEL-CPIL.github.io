import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Fallback: Greenwood-Gleason K32
    def fallback_construct():
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

    # Attempt GF(49) construction
    try:
        p = 7
        # GF(49) represented as integers 0..48, where value = a + b*p, a,b in 0..6
        # Actually, let's use pairs (a,b) for clarity, then map to index
        elements = [(a, b) for b in range(p) for a in range(p)]
        # Index mapping
        idx = {el: i for i, el in enumerate(elements)}
        
        def add(e1, e2):
            return ((e1[0] + e2[0]) % p, (e1[1] + e2[1]) % p)
        
        def sub(e1, e2):
            return ((e1[0] - e2[0]) % p, (e1[1] - e2[1]) % p)
        
        def mul(e1, e2):
            a1, b1 = e1
            a2, b2 = e2
            # (a1 + b1*x)(a2 + b2*x) = a1a2 + (a1b2 + a2b1 + b1b2*4)x + b1b2*4
            # x^2 = 4x + 4 (from x^2 + x + 3 = 0 => x^2 = -x - 3 = 6x + 4 = 6x + 4? 
            # Wait: x^2 + x + 3 = 0 => x^2 = -x - 3. In mod 7, -1=6, -3=4. So x^2 = 6x + 4.
            # Let's re-verify irreducibility of x^2 + x + 3 over GF(7).
            # Discriminant D = 1 - 12 = -11 = 3. Is 3 a square in GF(7)? Squares: 1,4,2,2,4,1. No 3. Irreducible.
            # So x^2 = 6x + 4.
            # Product: (a1 + b1 x)(a2 + b2 x) = a1a2 + a1b2 x + a2b1 x + b1b2 x^2
            # = a1a2 + (a1b2 + a2b1) x + b1b2 (6x + 4)
            # = (a1a2 + 4*b1*b2) + (a1b2 + a2b1 + 6*b1*b2) x
            a_res = (a1 * a2 + 4 * b1 * b2) % p
            b_res = (a1 * b2 + a2 * b1 + 6 * b1 * b2) % p
            return (a_res, b_res)
        
        # Find primitive element
        # Order of multiplicative group is 48
        # Factors of 48: 2, 3, 4, 6, 8, 12, 16, 24
        def pow_el(e, k):
            res = (1, 0)
            base = e
            while k > 0:
                if k & 1:
                    res = mul(res, base)
                base = mul(base, base)
                k >>= 1
            return res
            
        def is_primitive(e):
            if e == (0, 0): return False
            # Check e^(48/q) != 1 for all prime factors q of 48
            # Prime factors of 48: 2, 3
            # So check e^(48/2)=e^24 != 1 and e^(48/3)=e^16 != 1
            if pow_el(e, 24) == (1, 0): return False
            if pow_el(e, 16) == (1, 0): return False
            return True
            
        primitive = None
        for e in elements:
            if e != (0, 0) and is_primitive(e):
                primitive = e
                break
                
        if primitive is None:
            return fallback_construct()
            
        # Generate cyclic group
        group = []
        cur = (1, 0)
        for _ in range(48):
            group.append(cur)
            cur = mul(cur, primitive)
            
        # Partition into 3 cosets of cubic residues
        # H = { g^(3k) }
        # C0 = H
        # C1 = { g^(3k+1) }
        # C2 = { g^(3k+2) }
        
        color_map = {}
        for i in range(48):
            g_i = group[i]
            if i % 3 == 0:
                color_map[g_i] = 0
            elif i % 3 == 1:
                color_map[g_i] = 1
            else:
                color_map[g_i] = 2
                
        # Build coloring for K49
        n = 49
        colors = []
        # Vertices are elements 0..48 mapped to field elements
        # We need to check for monochromatic triangles
        
        # Precompute edge colors
        edge_color = [[-1]*n for _ in range(n)]
        for i in range(n):
            vi = elements[i]
            for j in range(i+1, n):
                vj = elements[j]
                diff = sub(vi, vj)
                if diff == (0, 0):
                    c = 0 # Should not happen for i!=j
                else:
                    c = color_map[diff]
                edge_color[i][j] = c
                edge_color[j][i] = c
                colors.append(c)
                
        # Check for monochromatic triangles
        valid = True
        for c in range(3):
            # For each pair (i,j) with color c, check if any k forms a triangle
            # Optimization: Just iterate all triples? 49 choose 3 is ~18k. Fast.
            for i in range(n):
                for j in range(i+1, n):
                    if edge_color[i][j] != c:
                        continue
                    for k in range(j+1, n):
                        if edge_color[i][k] == c and edge_color[j][k] == c:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if not valid:
                break
                
        if valid:
            return {'n': 49, 'colors': colors}
        else:
            # If 3-coloring fails, we need to use color 3.
            # Simple local search: find a triangle, recolor one edge to 3, repeat.
            # This might not work if the structure is too rigid, but worth a try.
            # However, if the 3-coloring is the standard one, it might be valid.
            # If it's invalid, we might just return fallback to be safe, 
            # OR try a simple repair.
            
            # Let's try a quick repair
            colors_list = colors[:]
            # Rebuild edge_color from list for ease
            ec = [[-1]*n for _ in range(n)]
            idx = 0
            for i in range(n):
                for j in range(i+1, n):
                    ec[i][j] = colors_list[idx]
                    ec[j][i] = colors_list[idx]
                    idx += 1
                    
            # Find a triangle and break it
            # This is complex to implement robustly in short time.
            # Given the hint strongly suggests GF(49) works, I will bet on it.
            # If it fails, returning 32 is safe.
            return fallback_construct()

    except Exception as e:
        return fallback_construct()

# EVOLVE-BLOCK-END
