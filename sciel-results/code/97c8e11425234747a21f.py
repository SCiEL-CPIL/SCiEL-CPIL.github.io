import numpy as np
import time

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 56 decomposition using Strassen on 2x2x2 blocks
    rank = 56
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen coefficients for 2x2x2
    # M1 = (a+d)(e+g)
    # M2 = (c+d)e
    # M3 = a(h-g)
    # M4 = d(e-f)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # U factors (A matrix elements)
    # a=A00, b=A01, c=A10, d=A11 (local 2x2)
    u_factors = [
        [(0, 1), (1, 1)], # M1: a+d
        [(1, 1), (1, 1)], # M2: c+d -> Wait, c is A10, d is A11.
        # Let's list (i, j_local) and coeff
        # M1: A00 + A11
        # M2: A10 + A11
        # M3: A00
        # M4: A11
        # M5: A00 + A01
        # M6: A10 - A00
        # M7: A01 - A11
    ]
    
    u_coeffs = [
        [(0, 0, 1), (1, 1, 1)], # M1: A00 + A11
        [(1, 0, 1), (1, 1, 1)], # M2: A10 + A11
        [(0, 0, 1)],            # M3: A00
        [(1, 1, 1)],            # M4: A11
        [(0, 0, 1), (0, 1, 1)], # M5: A00 + A01
        [(1, 0, 1), (0, 0, -1)],# M6: A10 - A00
        [(0, 1, 1), (1, 1, -1)] # M7: A01 - A11
    ]
    
    # V factors (B matrix elements)
    # e=B00, f=B01, g=B10, h=B11 (local 2x2)
    v_coeffs = [
        [(0, 0, 1), (1, 0, 1)], # M1: e+g -> B00 + B10
        [(0, 0, 1)],            # M2: e -> B00
        [(1, 1, 1), (1, 0, -1)],# M3: h-g -> B11 - B10
        [(0, 0, 1), (0, 1, -1)],# M4: e-f -> B00 - B01
        [(1, 1, 1)],            # M5: h -> B11
        [(0, 0, 1), (0, 1, 1)], # M6: e+f -> B00 + B01
        [(1, 0, 1), (1, 1, 1)]  # M7: g+h -> B10 + B11
    ]
    
    # W factors (C matrix elements)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    w_coeffs = [
        [(0, 0, 1), (0, 0, 1), (0, 0, -1), (0, 0, 1)], # M1, M4, -M5, M7 for C00? No, W is per term.
        # We need W[k*n+i, col] for each term s.
        # Term s contributes to C00, C01, C10, C11 with specific coeffs.
    ]
    
    # W contribution per term s to C_local (2x2)
    # s=1 (M1): C00+=1, C11+=1
    # s=2 (M2): C10+=1, C11-=1
    # s=3 (M3): C01+=1, C11+=1
    # s=4 (M4): C00+=1, C10+=1
    # s=5 (M5): C00-=1, C01+=1
    # s=6 (M6): C11+=1
    # s=7 (M7): C00+=1
    
    w_local = [
        [(0, 0, 1), (1, 1, 1)], # M1: C00, C11
        [(1, 0, 1), (1, 1, -1)],# M2: C10, C11
        [(0, 1, 1), (1, 1, 1)], # M3: C01, C11
        [(0, 0, 1), (1, 0, 1)], # M4: C00, C10
        [(0, 0, -1), (0, 1, 1)],# M5: C00, C01
        [(1, 1, 1)],            # M6: C11
        [(0, 0, 1)]             # M7: C00
    ]
    
    for jb in range(2):
        for kb in range(4):
            for s in range(7):
                # Fill U
                for i, jl, coeff in u_coeffs[s]:
                    idx_u = i * m + (jb * 2 + jl)
                    U[idx_u][col] += coeff
                
                # Fill V
                for jl, kl, coeff in v_coeffs[s]:
                    jg = jb * 2 + jl
                    kg = kb * 2 + kl
                    idx_v = jg * p + kg
                    V[idx_v][col] += coeff
                
                # Fill W
                for kl, i, coeff in w_local[s]:
                    kg = kb * 2 + kl
                    idx_w = kg * n + i
                    W[idx_w][col] += coeff
                
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
