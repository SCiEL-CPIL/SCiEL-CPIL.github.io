import numpy as np

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    # Rank 44: 4 Strassen blocks (rank 7 each) for top 2 rows, 16 schoolbook for bottom 1 row.
    rank = 44
    
    U = [[0] * rank for _ in range(n * m)] # 12 x 44
    V = [[0] * rank for _ in range(m * p)] # 16 x 44
    W = [[0] * rank for _ in range(p * n)] # 12 x 44
    
    col = 0
    
    # Helper to add a Strassen 2x2x2 block
    # a_row_start, a_col_start: start of A block (2x2)
    # b_row_start, b_col_start: start of B block (2x2)
    # c_row_start, c_col_start: start of C block (2x2)
    def add_strassen(a_rs, a_cs, b_rs, b_cs, c_rs, c_cs):
        nonlocal col
        # Strassen 7 terms
        # M1 = (a00+a11)(b00+b11)
        # U: a00, a11
        U[a_rs * m + a_cs][col] = 1
        U[(a_rs+1) * m + (a_cs+1)][col] = 1
        # V: b00, b11
        V[b_rs * p + b_cs][col] = 1
        V[(b_rs+1) * p + (b_cs+1)][col] = 1
        # W: c00, c11
        W[c_cs * n + c_rs][col] = 1
        W[(c_cs+1) * n + (c_rs+1)][col] = 1
        col += 1
        
        # M2 = (a10+a11)b00
        U[a_rs * m + (a_cs+1)][col] = 1 # a10? No. a_rs+1, a_cs
        U[(a_rs+1) * m + a_cs][col] = 1
        U[(a_rs+1) * m + (a_cs+1)][col] = 1
        V[b_rs * p + b_cs][col] = 1
        # W: c10
        W[c_cs * n + (c_rs+1)][col] = 1
        col += 1
        
        # M3 = a00(b01-b11)
        U[a_rs * m + a_cs][col] = 1
        V[b_rs * p + (b_cs+1)][col] = 1
        V[(b_rs+1) * p + (b_cs+1)][col] = -1
        # W: c01
        W[(c_cs+1) * n + c_rs][col] = 1
        col += 1
        
        # M4 = a11(b10-b00)
        U[(a_rs+1) * m + (a_cs+1)][col] = 1
        V[(b_rs+1) * p + b_cs][col] = 1
        V[b_rs * p + b_cs][col] = -1
        # W: c00, c10
        W[c_cs * n + c_rs][col] = 1
        W[c_cs * n + (c_rs+1)][col] = 1
        col += 1
        
        # M5 = (a00+a01)b11
        U[a_rs * m + a_cs][col] = 1
        U[a_rs * m + (a_cs+1)][col] = 1
        V[(b_rs+1) * p + (b_cs+1)][col] = 1
        # W: c00, c01
        W[c_cs * n + c_rs][col] = -1
        W[(c_cs+1) * n + c_rs][col] = 1
        col += 1
        
        # M6 = (a10-a00)(b00+b01)
        U[(a_rs+1) * m + a_cs][col] = 1
        U[a_rs * m + a_cs][col] = -1
        V[b_rs * p + b_cs][col] = 1
        V[b_rs * p + (b_cs+1)][col] = 1
        # W: c11
        W[(c_cs+1) * n + (c_rs+1)][col] = 1
        col += 1
        
        # M7 = (a01-a11)(b10+b11)
        U[a_rs * m + (a_cs+1)][col] = 1
        U[(a_rs+1) * m + (a_cs+1)][col] = -1
        V[(b_rs+1) * p + b_cs][col] = 1
        V[(b_rs+1) * p + (b_cs+1)][col] = 1
        # W: c00, c11
        W[c_cs * n + c_rs][col] = 1
        W[(c_cs+1) * n + (c_rs+1)][col] = -1
        col += 1

    # Top 2 rows of C (i=0,1)
    # C[0:2, 0:2] = A[0:2, 0:2] B[0:2, 0:2] + A[0:2, 2:4] B[2:4, 0:2]
    add_strassen(0, 0, 0, 0, 0, 0)
    add_strassen(0, 2, 2, 0, 0, 0)
    
    # C[0:2, 2:4] = A[0:2, 0:2] B[0:2, 2:4] + A[0:2, 2:4] B[2:4, 2:4]
    add_strassen(0, 0, 0, 2, 0, 2)
    add_strassen(0, 2, 2, 2, 0, 2)
    
    # Bottom 1 row of C (i=2)
    # C[2, 0:4] = A[2, 0:4] B[0:4, 0:4]
    for j in range(m):
        for k in range(p):
            U[2 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 2][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
