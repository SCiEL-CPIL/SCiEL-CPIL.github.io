import numpy as np

def construct():
    n, m, p = 2, 4, 5
    rank = 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add Strassen block
    def add_strassen(i_off, j_off, k_off):
        nonlocal col
        # Strassen terms
        # M1: a00(b00+b11)
        # M2: (a10+a11)b00
        # M3: a01(b10-b00)
        # M4: a11(b11-b10)
        # M5: (a00+a11)(b00+b10)
        # M6: (a01-a10)(b10+b11)
        # M7: (a00-a01)(b01+b11)
        
        # We define coefficients for A, B, C
        # A indices: (i, j) local
        # B indices: (j, k) local
        # C indices: (i, k) local
        
        terms = [
            # (A_coeffs, B_coeffs, C_coeffs)
            # M1
            ({(0,0): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (0,1): 1, (1,0): 0, (1,1): 0}), # Wait, C coeffs are for the result of this term added to C.
            # Actually, it's easier to define the contribution to C.
            # C = sum M_t * V_t
            # So for each M_t, we have a vector in A space, vector in B space, and vector in C space.
        ]
        
        # Let's just hardcode the 7 terms for Strassen
        # Term 1
        U[i_off*m + 0][col] += 1
        V[j_off*p + 0][col] += 1
        V[j_off*p + 1][col] += 1
        W[(k_off)*n + 0][col] += 1 # C00
        W[(k_off+1)*n + 0][col] += 1 # C01
        W[(k_off)*n + 1][col] += 0
        W[(k_off+1)*n + 1][col] += 0
        # This is getting messy to do manually in text. I will write a small internal loop.

    # Better approach: Define the Strassen decomposition as a list of tuples (a_idx, b_idx, c_idx, coeff)
    # Or just use the standard formulas.
    
    pass
