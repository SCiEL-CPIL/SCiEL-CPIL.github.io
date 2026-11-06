"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random

def construct():
    deadline = time.time() + 130  # safety margin before 150s limit
    
    def count_colors(coloring):
        return max(coloring) + 1
    
    def is_valid(coloring, n=16):
        # coloring is a list of 120 colors in lexicographic order for pairs (i,j), 0<=i<j<n
        # Build edge color lookup
        edge_color = {}
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                edge_color[(i,j)] = coloring[idx]
                idx += 1
        # Check every K4
        for quad in itertools.combinations(range(n), 4):
            a, b, c, d = quad
            colors = {
                edge_color[(a,b)], edge_color[(a,c)], edge_color[(a,d)],
                edge_color[(b,c)], edge_color[(b,d)], edge_color[(c,d)]
            }
            if len(colors) < 5:
                return False
        return True
    
    def make_coloring_formula(formula_fn, n=16):
        colors = []
        for i in range(n):
            for j in range(i+1, n):
                colors.append(formula_fn(i, j))
        return colors
    
    # GF(16) construction
    # GF(16) = GF(2)[x]/(x^4 + x + 1)
    # Represent elements as 4-bit integers
    # Multiplication table for GF(16)
    def gf16_mul(a, b):
        """Multiply two elements in GF(16)"""
        result = 0
        while b:
            if b & 1:
                result ^= a
            b >>= 1
            a <<= 1
            # Reduce: x^4 = x + 1, so if bit 4 is set, XOR with 0b00011
            if a & 16:
                a ^= 0b00011
        return result & 0b1111
    
    def gf16_trace(a):
        """Trace from GF(16) to GF(2): Tr(a) = a + a^2 + a^4 + a^8"""
        a2 = gf16_mul(a, a)
        a4 = gf16_mul(a2, a2)
        a8 = gf16_mul(a4, a4)
        return (a ^ a2 ^ a4 ^ a8) & 1
    
    def gf16_element_to_int(x):
        """Convert GF(16) element (4-bit) to integer 0-15"""
        return x
    
    # Create GF(16) construction: color edge {i,j} based on trace of product
    # Map vertices 0-15 to GF(16) elements
    # Use all 16 elements of GF(16)
    gf_elements = list(range(16))  # 0, 1, 2, ..., 15 as field elements
    
    def gf16_color(i, j):
        """Color edge (i,j) using GF(16) structure"""
        a = gf_elements[i]
        b = gf_elements[j]
        # Compute trace of a*b
        prod = gf16_mul(a, b)
        tr = gf16_trace(prod)
        # This gives only 2 colors (0 or 1), not enough
        # Instead, use the full product as color
        return prod
    
    # Better: use a more sophisticated mapping
    # Color = hash of (a*b, a+b) or similar
    def gf16_color_v2(i, j):
        """More complex GF(16) based coloring"""
        a = gf_elements[i]
        b = gf_elements[j]
        prod = gf16_mul(a, b)
        s = a ^ b  # addition in GF(2)^4
        # Combine product and sum
        return (prod * 16 + s) % 16  # Use lower 4 bits of product, upper from sum
    
    # Even better: use multiple traces
    def gf16_color_v3(i, j):
        """Use multiple field operations to get more color variety"""
        a = gf_elements[i]
        b = gf_elements[j]
        prod = gf16_mul(a, b)
        # Get 4 bits from different operations
        bit0 = gf16_trace(prod)
        a2 = gf16_mul(a, a)
        b2 = gf16_mul(b, b)
        prod2 = gf16_mul(a2, b2)
        bit1 = gf16_trace(prod2)
        # Use a^3 * b
        a3 = gf16_mul(a2, a)
        prod3 = gf16_mul(a3, b)
        bit2 = gf16_trace(prod3)
        # Use a * b^3
        b3 = gf16_mul(b2, b)
        prod4 = gf16_mul(a, b3)
        bit3 = gf16_trace(prod4)
        return (bit0 | (bit1 << 1) | (bit2 << 2) | (bit3 << 3))
    
    # Try GF(16) constructions
    candidates = []
    
    # GF(16) based candidates
    c_gf1 = make_coloring_formula(gf16_color)
    candidates.append(c_gf1)
    
    c_gf2 = make_coloring_formula(gf16_color_v2)
    candidates.append(c_gf2)
    
    c_gf3 = make_coloring_formula(gf16_color_v3)
    candidates.append(c_gf3)
    
    # Also try with different element mappings (permutations of GF(16) elements)
    random.seed(42)
    for _ in range(10):
        perm = list(range(16))
        random.shuffle(perm)
        def make_gf_color(perm):
            def color(i, j):
                a = perm[i]
                b = perm[j]
                prod = gf16_mul(a, b)
                s = a ^ b
                return (prod ^ s)  # XOR product with sum
            return color
        cand = make_coloring_formula(make_gf_color(perm))
        candidates.append(cand)
    
    # Linear constructions as before
    # 1. Seed: (i+j) mod 17
    c1 = make_coloring_formula(lambda i, j: (i + j) % 17)
    candidates.append(c1)
    
    # 2. (i+j) mod 15
    c2 = make_coloring_formula(lambda i, j: (i + j) % 15)
    candidates.append(c2)
    
    # 3. (i+j) mod 14
    c3 = make_coloring_formula(lambda i, j: (i + j) % 14)
    candidates.append(c3)
    
    # 4. (i+j) mod 13
    c4 = make_coloring_formula(lambda i, j: (i + j) % 13)
    candidates.append(c4)
    
    # 5. (i*2 + j) mod 17
    c5 = make_coloring_formula(lambda i, j: (2*i + j) % 17)
    candidates.append(c5)
    
    # 6. (i*2 + j) mod 15
    c6 = make_coloring_formula(lambda i, j: (2*i + j) % 15)
    candidates.append(c6)
    
    # 7. (i*3 + j) mod 17
    c7 = make_coloring_formula(lambda i, j: (3*i + j) % 17)
    candidates.append(c7)
    
    # 8. (i*3 + j) mod 15
    c8 = make_coloring_formula(lambda i, j: (3*i + j) % 15)
    candidates.append(c8)
    
    # 9. (i + 2*j) mod 17
    c9 = make_coloring_formula(lambda i, j: (i + 2*j) % 17)
    candidates.append(c9)
    
    # 10. (i + 3*j) mod 17
    c10 = make_coloring_formula(lambda i, j: (i + 3*j) % 17)
    candidates.append(c10)
    
    # 11. (i*2 + j*3) mod 17
    c11 = make_coloring_formula(lambda i, j: (2*i + 3*j) % 17)
    candidates.append(c11)
    
    # 12. (i*2 + j*3) mod 15
    c12 = make_coloring_formula(lambda i, j: (2*i + 3*j) % 15)
    candidates.append(c12)
    
    # 13. (i*3 + j*5) mod 17
    c13 = make_coloring_formula(lambda i, j: (3*i + 5*j) % 17)
    candidates.append(c13)
    
    # 14. (i*3 + j*5) mod 15
    c14 = make_coloring_formula(lambda i, j: (3*i + 5*j) % 15)
    candidates.append(c14)
    
    # 15. (i + j) mod 16
    c15 = make_coloring_formula(lambda i, j: (i + j) % 16)
    candidates.append(c15)
    
    # 16. (i*2 + j) mod 16
    c16 = make_coloring_formula(lambda i, j: (2*i + j) % 16)
    candidates.append(c16)
    
    # 17. (i*3 + j) mod 16
    c17 = make_coloring_formula(lambda i, j: (3*i + j) % 16)
    candidates.append(c17)
    
    # 18. (i + 3*j) mod 16
    c18 = make_coloring_formula(lambda i, j: (i + 3*j) % 16)
    candidates.append(c18)
    
    # 19. (i*2 + j*3) mod 16
    c19 = make_coloring_formula(lambda i, j: (2*i + 3*j) % 16)
    candidates.append(c19)
    
    # 20. (i*3 + j*5) mod 16
    c20 = make_coloring_formula(lambda i, j: (3*i + 5*j) % 16)
    candidates.append(c20)
    
    # Try more with various moduli and coefficients
    for mod in range(13, 18):
        for a in range(1, 6):
            for b in range(1, 6):
                if a == b:
                    continue
                cand = make_coloring_formula(lambda i, j, a=a, b=b, mod=mod: (a*i + b*j) % mod)
                candidates.append(cand)
    
    # Find the best valid candidate
    best = None
    best_num_colors = float('inf')
    
    for cand in candidates:
        if time.time() > deadline:
            break
        if is_valid(cand):
            nc = count_colors(cand)
            if nc < best_num_colors:
                best_num_colors = nc
                best = cand
    
    if best is not None:
        return best
    
    # Fallback: return the seed
    return c1
    
# EVOLVE-BLOCK-END
