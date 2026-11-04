import numpy as np
import time

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # Rank 46
    rank = 46
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a Strassen block
    # A_rows: [i0, i1] (fixed 0,1)
    # A_cols: [j0, j1]
    # B_rows: [j0, j1]
    # B_cols: [k0, k1]
    # C_rows: [k0, k1]
    # C_cols: [i0, i1] (fixed 0,1)
    # Note: W index is k*n + i
    
    def add_strassen_block(j0, j1, k0, k1):
        nonlocal col
        # Local indices
        # A: i in {0,1}, j in {j0, j1}
        # B: j in {j0, j1}, k in {k0, k1}
        # C: k in {k0, k1}, i in {0,1}
        
        # M1 = (A00 + A11)(B00 + B11)
        # A00: i=0, j=j0
        # A11: i=1, j=j1
        # B00: j=j0, k=k0
        # B11: j=j1, k=k1
        
        terms = []
        
        # M1
        # (A[0,j0] + A[1,j1]) * (B[j0,k0] + B[j1,k1])
        # Expansion:
        # A[0,j0]B[j0,k0] + A[0,j0]B[j1,k1] + A[1,j1]B[j0,k0] + A[1,j1]B[j1,k1]
        # This is not a simple tensor product.
        # Strassen computes M1 as a single product of sums.
        # In tensor decomposition, we represent the bilinear form.
        # The term is (u1 + u2) * (v1 + v2).
        # This contributes to U, V, W.
        # U part: u1, u2
        # V part: v1, v2
        # W part: coefficient for C.
        
        # Actually, the standard way to add Strassen to a tensor decomposition:
        # Each Mx is a scalar product of two linear forms.
        # M1 = (A00 + A11) * (B00 + B11)
        # U entries: A00 coeff 1, A11 coeff 1
        # V entries: B00 coeff 1, B11 coeff 1
        # W entries: C00 coeff 1, C11 coeff 1
        
        # Let's define the contributions for each M
        
        # M1
        U[0*m + j0][col] += 1
        U[1*m + j1][col] += 1
        V[j0*p + k0][col] += 1
        V[j1*p + k1][col] += 1
        W[k0*n + 0][col] += 1 # C00
        W[k1*n + 1][col] += 1 # C11
        
        # M2 = (A10 + A11) * B00
        U[1*m + j0][col] += 1
        U[1*m + j1][col] += 1
        V[j0*p + k0][col] += 1
        W[k0*n + 1][col] += 1 # C10
        
        # M3 = A00 * (B01 - B11)
        U[0*m + j0][col] += 1
        V[j0*p + k1][col] += 1
        V[j1*p + k1][col] += -1
        W[k1*n + 0][col] += 1 # C01
        
        # M4 = A11 * (B10 - B00)
        U[1*m + j1][col] += 1
        V[j1*p + k0][col] += 1
        V[j0*p + k0][col] += -1
        W[k0*n + 0][col] += 1 # C00
        W[k0*n + 1][col] += 1 # C10
        
        # M5 = (A00 + A10) * B11
        U[0*m + j0][col] += 1
        U[1*m + j0][col] += 1
        V[j1*p + k1][col] += 1
        W[k1*n + 0][col] += -1 # C00
        W[k1*n + 1][col] += 1 # C01
        
        # M6 = (A11 - A00) * (B00 + B01)
        U[1*m + j1][col] += 1
        U[0*m + j0][col] += -1
        V[j0*p + k0][col] += 1
        V[j0*p + k1][col] += 1
        W[k0*n + 0][col] += 1 # C00
        
        # M7 = (A01 - A11) * (B10 + B11)
        U[0*m + j1][col] += 1
        U[1*m + j1][col] += -1
        V[j1*p + k0][col] += 1
        V[j1*p + k1][col] += 1
        W[k1*n + 1][col] += 1 # C11
        
        col += 1
        return col

    # 4 Strassen blocks for 2x4x4
    add_strassen_block(0, 1, 0, 1) # A0, B00 -> C00
    add_strassen_block(2, 3, 0, 1) # A1, B10 -> C00
    add_strassen_block(0, 1, 2, 3) # A0, B01 -> C01
    add_strassen_block(2, 3, 2, 3) # A1, B11 -> C01
    
    # 2x4x1: j in 0..3, k=4
    for i in range(2):
        for j in range(4):
            U[i*m + j][col] = 1
            V[j*p + 4][col] = 1
            W[4*n + i][col] = 1
            col += 1
            
    # 2x1x4: j=4, k in 0..3
    for i in range(2):
        for k in range(4):
            U[i*m + 4][col] = 1
            V[4*p + k][col] = 1
            W[k*n + i][col] = 1
            col += 1
            
    # 2x1x1: j=4, k=4
    for i in range(2):
        U[i*m + 4][col] = 1
        V[4*p + 4][col] = 1
        W[4*n + i][col] = 1
        col += 1
        
    return {"U": U, "V": V, "W": W}
