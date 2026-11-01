import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 5, 6
    rank = 54
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a Strassen term
    def add_strassen(i1, i2, j1, j2, k1, k2, start_col):
        # i1, i2: row indices in A
        # j1, j2: col indices in A / row indices in B
        # k1, k2: col indices in B / col indices in C
        
        # Strassen formulas for 2x2x2
        # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
        # M1 = (a+d)(e+h)
        # M2 = (c+d)e
        # M3 = a(f-h)
        # M4 = d(g-e)
        # M5 = (a+b)h
        # M6 = (c-a)(e+g)
        # M7 = (b-d)(f+h)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # Term 1: M1
        c = start_col
        # U: a=1, d=1 => (i1, j1)=1, (i2, j2)=1
        U[i1*m + j1][c] += 1.0
        U[i2*m + j2][c] += 1.0
        # V: e=1, h=1 => (j1, k1)=1, (j2, k2)=1
        V[j1*p + k1][c] += 1.0
        V[j2*p + k2][c] += 1.0
        # W: C00=1, C11=1 => (k1, i1)=1, (k2, i2)=1
        W[k1*n + i1][c] += 1.0
        W[k2*n + i2][c] += 1.0
        col_val = c
        
        # Term 2: M2
        c = start_col + 1
        # U: c=1, d=1 => (i1, j2)=1, (i2, j2)=1
        U[i1*m + j2][c] += 1.0
        U[i2*m + j2][c] += 1.0
        # V: e=1 => (j1, k1)=1
        V[j1*p + k1][c] += 1.0
        # W: C10=1, C11=-1 => (k1, i2)=1, (k2, i2)=-1
        W[k1*n + i2][c] += 1.0
        W[k2*n + i2][c] += -1.0
        
        # Term 3: M3
        c = start_col + 2
        # U: a=1 => (i1, j1)=1
        U[i1*m + j1][c] += 1.0
        # V: f=1, h=-1 => (j1, k2)=1, (j2, k2)=-1
        V[j1*p + k2][c] += 1.0
        V[j2*p + k2][c] += -1.0
        # W: C01=1 => (k2, i1)=1
        W[k2*n + i1][c] += 1.0
        
        # Term 4: M4
        c = start_col + 3
        # U: d=1 => (i2, j2)=1
        U[i2*m + j2][c] += 1.0
        # V: g=1, e=-1 => (j2, k1)=1, (j1, k1)=-1
        V[j2*p + k1][c] += 1.0
        V[j1*p + k1][c] += -1.0
        # W: C00=1, C10=1 => (k1, i1)=1, (k1, i2)=1
        W[k1*n + i1][c] += 1.0
        W[k1*n + i2][c] += 1.0
        
        # Term 5: M5
        c = start_col + 4
        # U: a=1, b=1 => (i1, j1)=1, (i1, j2)=1
        U[i1*m + j1][c] += 1.0
        U[i1*m + j2][c] += 1.0
        # V: h=1 => (j2, k2)=1
        V[j2*p + k2][c] += 1.0
        # W: C00=-1, C01=1 => (k1, i1)=-1, (k2, i1)=1
        W[k1*n + i1][c] += -1.0
        W[k2*n + i1][c] += 1.0
        
        # Term 6: M6
        c = start_col + 5
        # U: c=1, a=-1 => (i2, j2)=1, (i1, j1)=-1
        U[i2*m + j2][c] += 1.0
        U[i1*m + j1][c] += -1.0
        # V: e=1, g=1 => (j1, k1)=1, (j2, k1)=1
        V[j1*p + k1][c] += 1.0
        V[j2*p + k1][c] += 1.0
        # W: C11=1 => (k2, i2)=1
        W[k2*n + i2][c] += 1.0
        
        # Term 7: M7
        c = start_col + 6
        # U: b=1, d=-1 => (i1, j2)=1, (i2, j2)=-1
        U[i1*m + j2][c] += 1.0
        U[i2*m + j2][c] += -1.0
        # V: f=1, h=1 => (j1, k2)=1, (j2, k2)=1
        V[j1*p + k2][c] += 1.0
        V[j2*p + k2][c] += 1.0
        # W: C00=1 => (k1, i1)=1
        W[k1*n + i1][c] += 1.0
        
        return start_col + 7

    # Part 1: A1 (2x2) * B1 (2x6)
    # A1 rows 0,1. Cols 0,1.
    # B1 rows 0,1. Cols 0..5.
    # Split B1 into 3 blocks of 2x2.
    # Block 0: B cols 0,1.
    col = add_strassen(0, 1, 0, 1, 0, 1, col)
    # Block 1: B cols 2,3.
    col = add_strassen(0, 1, 0, 1, 2, 3, col)
    # Block 2: B cols 4,5.
    col = add_strassen(0, 1, 0, 1, 4, 5, col)
    
    # Part 2: A2 (2x3) * B2 (3x6)
    # A2 rows 0,1. Cols 2,3,4.
    # B2 rows 2,3,4. Cols 0..5.
    # Split B2 into 3 blocks of 3x2.
    # Block b: B2 cols 2b, 2b+1.
    for b in range(3):
        k1 = 2 * b
        k2 = 2 * b + 1
        
        # Sub-part 2.1: A21 (2x2) * B2b_top (2x2)
        # A21 uses A cols 2,3. B2b_top uses B rows 2,3.
        col = add_strassen(0, 1, 2, 3, k1, k2, col)
        
        # Sub-part 2.2: A22 (2x1) * B2b_bottom (1x2)
        # A22 uses A col 4. B2b_bottom uses B row 4.
        # Naive 4 terms.
        # Term 1: A[0,4] * B[4, k1] -> C[0, k1]
        c = col
        U[0*m + 4][c] += 1.0
        V[4*p + k1][c] += 1.0
        W[k1*n + 0][c] += 1.0
        col += 1
        
        # Term 2: A[0,4] * B[4, k2] -> C[0, k2]
        c = col
        U[0*m + 4][c] += 1.0
        V[4*p + k2][c] += 1.0
        W[k2*n + 0][c] += 1.0
        col += 1
        
        # Term 3: A[1,4] * B[4, k1] -> C[1, k1]
        c = col
        U[1*m + 4][c] += 1.0
        V[4*p + k1][c] += 1.0
        W[k1*n + 1][c] += 1.0
        col += 1
        
        # Term 4: A[1,4] * B[4, k2] -> C[1, k2]
        c = col
        U[1*m + 4][c] += 1.0
        V[4*p + k2][c] += 1.0
        W[k2*n + 1][c] += 1.0
        col += 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
