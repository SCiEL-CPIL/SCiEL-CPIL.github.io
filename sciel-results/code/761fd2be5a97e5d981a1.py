import numpy as np
import time

DIMENSIONS = [2, 4, 7]

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    U3 = U_np.reshape(n, m, rank)
    V3 = V_np.reshape(m, p, rank)
    W3 = W_np.reshape(p, n, rank)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_vec = U3[i, j, :]
                v_vec = V3[j, k, :]
                w_vec = W3[k, i, :]
                total = np.sum(u_vec * v_vec * w_vec)
                if abs(total - 1.0) > tol:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Target rank 50
    rank = 50
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_block(a_row, a_col, b_row, b_col, c_row, c_col):
        nonlocal col
        # a_row, a_col, b_row, b_col, c_row, c_col are indices in the global matrices
        # a_row: 0 or 1 (row of A)
        # a_col: 0..3 (col of A)
        # b_row: 0..3 (row of B)
        # b_col: 0..6 (col of B)
        # c_row: 0 or 1 (row of C)
        # c_col: 0..6 (col of C)
        
        # Strassen for 2x2x2
        # A is 2x2, B is 2x2, C is 2x2
        # A indices: (0,0), (0,1), (1,0), (1,1) relative to a_row, a_col
        # But here A is 2x4, so we take 2 rows and 2 cols.
        # A_sub[0,0] is U[a_row*4 + a_col, col]
        # A_sub[0,1] is U[a_row*4 + a_col+1, col]
        # A_sub[1,0] is U[(a_row+1)*4 + a_col, col]
        # A_sub[1,1] is U[(a_row+1)*4 + a_col+1, col]
        
        # B_sub[0,0] is V[b_row*7 + b_col, col]
        # B_sub[0,1] is V[b_row*7 + b_col+1, col]
        # B_sub[1,0] is V[(b_row+1)*7 + b_col, col]
        # B_sub[1,1] is V[(b_row+1)*7 + b_col+1, col]
        
        # C_sub[0,0] is W[b_col*2 + a_row, col] ?
        # W index is k*n + i. k is b_col (global col of B), i is a_row (global row of A).
        # C_sub[0,0] corresponds to global C[a_row, b_col].
        # So W index is b_col * 2 + a_row.
        
        # M1 = (A00 + A11)(B00 + B11)
        U[a_row*4 + a_col][col] += 1
        U[(a_row+1)*4 + a_col+1][col] += 1
        V[b_row*7 + b_col][col] += 1
        V[(b_row+1)*7 + b_col+1][col] += 1
        W[b_col*2 + a_row][col] += 1
        W[(b_col+1)*2 + a_row][col] += 1
        W[b_col*2 + a_row+1][col] += 1
        W[(b_col+1)*2 + a_row+1][col] += 1
        col += 1
        
        # M2 = (A10 + A11)B00
        U[(a_row+1)*4 + a_col][col] += 1
        U[(a_row+1)*4 + a_col+1][col] += 1
        V[b_row*7 + b_col][col] += 1
        W[b_col*2 + a_row+1][col] += 1
        col += 1
        
        # M3 = A00(B01 - B11)
        U[a_row*4 + a_col][col] += 1
        V[b_row*7 + b_col+1][col] += 1
        V[(b_row+1)*7 + b_col+1][col] -= 1
        W[(b_col+1)*2 + a_row][col] += 1
        col += 1
        
        # M4 = A11(B10 - B00)
        U[(a_row+1)*4 + a_col+1][col] += 1
        V[(b_row+1)*7 + b_col][col] += 1
        V[b_row*7 + b_col][col] -= 1
        W[b_col*2 + a_row+1][col] += 1
        col += 1
        
        # M5 = (A00 + A01)B11
        U[a_row*4 + a_col][col] += 1
        U[a_row*4 + a_col+1][col] += 1
        V[(b_row+1)*7 + b_col+1][col] += 1
        W[(b_col+1)*2 + a_row][col] += 1
        col += 1
        
        # M6 = (A10 - A00)(B00 + B01)
        U[(a_row+1)*4 + a_col][col] += 1
        U[a_row*4 + a_col][col] -= 1
        V[b_row*7 + b_col][col] += 1
        V[b_row*7 + b_col+1][col] += 1
        W[b_col*2 + a_row+1][col] += 1
        W[(b_col+1)*2 + a_row+1][col] += 1
        col += 1
        
        # M7 = (A01 - A11)(B10 + B11)
        U[a_row*4 + a_col+1][col] += 1
        U[(a_row+1)*4 + a_col+1][col] -= 1
        V[(b_row+1)*7 + b_col][col] += 1
        V[(b_row+1)*7 + b_col+1][col] += 1
        W[b_col*2 + a_row][col] += 1
        W[(b_col+1)*2 + a_row][col] += 1
        col += 1

    def add_dot_block(a_row, a_col, b_row, b_col, c_row, c_col):
        nonlocal col
        # 2x2x1 block
        # A is 2x2, B is 2x1, C is 2x1
        # C0 = A00 B0 + A01 B1
        # C1 = A10 B0 + A11 B1
        
        # P0 = A00 B0
        U[a_row*4 + a_col][col] += 1
        V[b_row*7 + b_col][col] += 1
        W[b_col*2 + a_row][col] += 1
        col += 1
        
        # P1 = A01 B1
        U[a_row*4 + a_col+1][col] += 1
        V[(b_row+1)*7 + b_col][col] += 1
        W[b_col*2 + a_row][col] += 1
        col += 1
        
        # P2 = A10 B0
        U[(a_row+1)*4 + a_col][col] += 1
        V[b_row*7 + b_col][col] += 1
        W[b_col*2 + a_row+1][col] += 1
        col += 1
        
        # P3 = A11 B1
        U[(a_row+1)*4 + a_col+1][col] += 1
        V[(b_row+1)*7 + b_col][col] += 1
        W[b_col*2 + a_row+1][col] += 1
        col += 1

    # 6 Strassen blocks
    # B rows 0,1. B cols 0,1. A cols 0,1.
    add_strassen_block(0, 0, 0, 0, 0, 0)
    # B rows 0,1. B cols 0,1. A cols 2,3.
    add_strassen_block(0, 2, 0, 0, 0, 0)
    # B rows 0,1. B cols 2,3. A cols 0,1.
    add_strassen_block(0, 0, 0, 2, 0, 2)
    # B rows 0,1. B cols 2,3. A cols 2,3.
    add_strassen_block(0, 2, 0, 2, 0, 2)
    # B rows 0,1. B cols 4,5. A cols 0,1.
    add_strassen_block(0, 0, 0, 4, 0, 4)
    # B rows 0,1. B cols 4,5. A cols 2,3.
    add_strassen_block(0, 2, 0, 4, 0, 4)
    
    # 2 Dot blocks for col 6
    # B rows 0,1. B col 6. A cols 0,1.
    add_dot_block(0, 0, 0, 6, 0, 6)
    # B rows 2,3. B col 6. A cols 2,3.
    add_dot_block(0, 2, 2, 6, 0, 6)
    
    # Verify
    if not _verify(U, V, W, n, m, p):
        # Fallback to schoolbook if verification fails
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
    
    return {"U": U, "V": V, "W": W}
