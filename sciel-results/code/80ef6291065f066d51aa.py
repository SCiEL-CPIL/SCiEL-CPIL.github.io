import numpy as np
import time

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS  # n=3, m=3, p=4
    # A is n x m = 3x3, B is m x p = 3x4, C is n x p = 3x4
    # Index convention: A[i,j], B[j,k], C[i,k]
    # U shape: (n*m, rank) = (9, rank), indexed by i*m+j
    # V shape: (m*p, rank) = (12, rank), indexed by j*p+k
    # W shape: (p*n, rank) = (12, rank), indexed by k*n+i
    
    start_time = time.time()
    deadline = start_time + 100.0
    
    # Block decomposition:
    # A (3x3) split into:
    #   A22 = A[0:2, 0:2] (2x2)
    #   A21 = A[0:2, 2]   (2x1)
    #   A12 = A[2, 0:2]   (1x2)
    #   A11 = A[2, 2]     (1x1)
    # B (3x4) split into:
    #   B22 = B[0:2, 0:2] (2x2)
    #   B2p = B[0:2, 2:4] (2x2)
    #   B12 = B[2, 0:2]   (1x2)
    #   B1p = B[2, 2:4]   (1x2)
    # C (3x4) = 
    #   C[0:2, 0:2] = A22*B22 + A21*B12
    #   C[0:2, 2:4] = A22*B2p + A21*B1p
    #   C[2, 0:2]   = A12*B22 + A11*B12
    #   C[2, 2:4]   = A12*B2p + A11*B1p
    
    # Strassen 7-term for 2x2x2:
    # For matrices X (2x2), Y (2x2):
    # M1 = (X00+X11)(Y00+Y11)
    # M2 = (X10+X11)Y00
    # M3 = X00(Y10-Y00)
    # M4 = X11(Y01-Y11)
    # M5 = (X00+X01)Y11
    # M6 = (X10-X00)(Y00+Y01)
    # M7 = (X01-X11)(Y10+Y11)
    # Z00 = M1+M4-M5+M7
    # Z01 = M3+M5
    # Z10 = M2+M4
    # Z11 = M1-M2+M3+M6
    
    # Build the rank-34 decomposition
    rank = 34
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    col = 0
    
    # Helper to add a Strassen 2x2x2 decomposition
    # X = X22 (2x2), Y = Y22 (2x2), Z = Z22 (2x2)
    # X[i,j] contributes to U at row i*m+j (for i,j in 0,1)
    # Y[i,j] contributes to V at row i*p+j (for i in 0,1, j in 0,1)
    # Z[i,j] contributes to W at row j*n+i (for i,j in 0,1)
    def add_strassen_2x2(x_row_offset, x_col_offset, y_row_offset, y_col_offset, z_row_offset, z_col_offset):
        """
        Add Strassen 7-term for multiplying X (2x2) by Y (2x2) to get Z (2x2).
        X is at A[x_row_offset:x_row_offset+2, x_col_offset:x_col_offset+2]
        Y is at B[y_row_offset:y_row_offset+2, y_col_offset:y_col_offset+2]
        Z is at C[z_row_offset:z_row_offset+2, z_col_offset:z_col_offset+2]
        """
        nonlocal col
        # Strassen terms for X*Y = Z
        # M1 = (X00+X11)(Y00+Y11)
        # M2 = (X10+X11)Y00
        # M3 = X00(Y10-Y00)
        # M4 = X11(Y01-Y11)
        # M5 = (X00+X01)Y11
        # M6 = (X10-X00)(Y00+Y01)
        # M7 = (X01-X11)(Y10+Y11)
        
        # Term 1: M1 = (X00+X11)(Y00+Y11)
        # X00 at A[x_row_offset, x_col_offset], X11 at A[x_row_offset+1, x_col_offset+1]
        # Y00 at B[y_row_offset, y_col_offset], Y11 at B[y_row_offset+1, y_col_offset+1]
        # Z00 += M1, Z01 -= M1? No: Z00 = M1+M4-M5+M7, Z11 = M1-M2+M3+M6
        # So M1 contributes +1 to Z00 and +1 to Z11
        
        # U for X00: row = (x_row_offset)*m + (x_col_offset)
        # U for X11: row = (x_row_offset+1)*m + (x_col_offset+1)
        # V for Y00: row = (y_row_offset)*p + (y_col_offset)
        # V for Y11: row = (y_row_offset+1)*p + (y_col_offset+1)
        # W for Z00: row = (z_col_offset)*n + (z_row_offset)
        # W for Z11: row = (z_col_offset+1)*n + (z_row_offset+1)
        
        U[x_row_offset*m + x_col_offset, col] = 1
        U[(x_row_offset+1)*m + (x_col_offset+1), col] = 1
        V[y_row_offset*p + y_col_offset, col] = 1
        V[(y_row_offset+1)*p + (y_col_offset+1), col] = 1
        W[z_col_offset*n + z_row_offset, col] = 1
        W[(z_col_offset+1)*n + (z_row_offset+1), col] = 1
        col += 1
        
        # Term 2: M2 = (X10+X11)Y00
        # X10 at A[x_row_offset+1, x_col_offset], X11 at A[x_row_offset+1, x_col_offset+1]
        # Y00 at B[y_row_offset, y_col_offset]
        # Z10 += M2, Z11 -= M2
        
        U[(x_row_offset+1)*m + x_col_offset, col] = 1
        U[(x_row_offset+1)*m + (x_col_offset+1), col] = 1
        V[y_row_offset*p + y_col_offset, col] = 1
        W[(z_col_offset)*n + (z_row_offset+1), col] = 1  # Z10
        W[(z_col_offset+1)*n + (z_row_offset+1), col] = -1  # Z11
        col += 1
        
        # Term 3: M3 = X00(Y10-Y00)
        # X00 at A[x_row_offset, x_col_offset]
        # Y10 at B[y_row_offset+1, y_col_offset], Y00 at B[y_row_offset, y_col_offset]
        # Z01 += M3, Z11 += M3
        
        U[x_row_offset*m + x_col_offset, col] = 1
        V[(y_row_offset+1)*p + y_col_offset, col] = 1
        V[y_row_offset*p + y_col_offset, col] = -1
        W[(z_col_offset+1)*n + z_row_offset, col] = 1  # Z01
        W[(z_col_offset+1)*n + (z_row_offset+1), col] = 1  # Z11
        col += 1
        
        # Term 4: M4 = X11(Y01-Y11)
        # X11 at A[x_row_offset+1, x_col_offset+1]
        # Y01 at B[y_row_offset, y_col_offset+1], Y11 at B[y_row_offset+1, y_col_offset+1]
        # Z00 += M4, Z10 += M4
        
        U[(x_row_offset+1)*m + (x_col_offset+1), col] = 1
        V[y_row_offset*p + (y_col_offset+1), col] = 1
        V[(y_row_offset+1)*p + (y_col_offset+1), col] = -1
        W[z_col_offset*n + z_row_offset, col] = 1  # Z00
        W[z_col_offset*n + (z_row_offset+1), col] = 1  # Z10
        col += 1
        
        # Term 5: M5 = (X00+X01)Y11
        # X00 at A[x_row_offset, x_col_offset], X01 at A[x_row_offset, x_col_offset+1]
        # Y11 at B[y_row_offset+1, y_col_offset+1]
        # Z00 -= M5, Z01 += M5
        
        U[x_row_offset*m + x_col_offset, col] = 1
        U[x_row_offset*m + (x_col_offset+1), col] = 1
        V[(y_row_offset+1)*p + (y_col_offset+1), col] = 1
        W[z_col_offset*n + z_row_offset, col] = -1  # Z00
        W[(z_col_offset+1)*n + z_row_offset, col] = 1  # Z01
        col += 1
        
        # Term 6: M6 = (X10-X00)(Y00+Y01)
        # X10 at A[x_row_offset+1, x_col_offset], X00 at A[x_row_offset, x_col_offset]
        # Y00 at B[y_row_offset, y_col_offset], Y01 at B[y_row_offset, y_col_offset+1]
        # Z11 += M6
        
        U[(x_row_offset+1)*m + x_col_offset, col] = 1
        U[x_row_offset*m + x_col_offset, col] = -1
        V[y_row_offset*p + y_col_offset, col] = 1
        V[y_row_offset*p + (y_col_offset+1), col] = 1
        W[(z_col_offset+1)*n + (z_row_offset+1), col] = 1  # Z11
        col += 1
        
        # Term 7: M7 = (X01-X11)(Y10+Y11)
        # X01 at A[x_row_offset, x_col_offset+1], X11 at A[x_row_offset+1, x_col_offset+1]
        # Y10 at B[y_row_offset+1, y_col_offset], Y11 at B[y_row_offset+1, y_col_offset+1]
        # Z00 += M7
        
        U[x_row_offset*m + (x_col_offset+1), col] = 1
        U[(x_row_offset+1)*m + (x_col_offset+1), col] = -1
        V[(y_row_offset+1)*p + y_col_offset, col] = 1
        V[(y_row_offset+1)*p + (y_col_offset+1), col] = 1
        W[z_col_offset*n + z_row_offset, col] = 1  # Z00
        col += 1
    
    # Helper to add a 2x1x2 decomposition (rank 4)
    # X (2x1) * Y (1x2) = Z (2x2)
    # Z[i,j] = sum_k X[i,k] * Y[k,j] = X[i,0] * Y[0,j]
    # So each Z[i,j] needs one multiplication: X[i,0] * Y[0,j]
    def add_2x1x2(x_row_offset, y_col_offset, z_row_offset, z_col_offset):
        """
        Add rank-4 decomposition for X (2x1) * Y (1x2) = Z (2x2).
        X is at A[x_row_offset:x_row_offset+2, 2] (column 2 of A, rows 0,1)
        Y is at B[2, y_col_offset:y_col_offset+2] (row 2 of B, cols y_col_offset, y_col_offset+1)
        Z is at C[z_row_offset:z_row_offset+2, z_col_offset:z_col_offset+2]
        """
        nonlocal col
        # Z[0,0] = X[0,0] * Y[0,0]
        U[(x_row_offset)*m + 2, col] = 1
        V[2*p + y_col_offset, col] = 1
        W[(z_col_offset)*n + z_row_offset, col] = 1
        col += 1
        
        # Z[0,1] = X[0,0] * Y[0,1]
        U[(x_row_offset)*m + 2, col] = 1
        V[2*p + (y_col_offset+1), col] = 1
        W[(z_col_offset+1)*n + z_row_offset, col] = 1
        col += 1
        
        # Z[1,0] = X[1,0] * Y[0,0]
        U[(x_row_offset+1)*m + 2, col] = 1
        V[2*p + y_col_offset, col] = 1
        W[(z_col_offset)*n + (z_row_offset+1), col] = 1
        col += 1
        
        # Z[1,1] = X[1,0] * Y[0,1]
        U[(x_row_offset+1)*m + 2, col] = 1
        V[2*p + (y_col_offset+1), col] = 1
        W[(z_col_offset+1)*n + (z_row_offset+1), col] = 1
        col += 1
    
    # Helper to add a 1x2x2 decomposition (rank 4)
    # X (1x2) * Y (2x2) = Z (1x2)
    # Z[0,j] = sum_k X[0,k] * Y[k,j]
    # Z[0,0] = X[0,0]*Y[0,0] + X[0,1]*Y[1,0]
    # Z[0,1] = X[0,0]*Y[0,1] + X[0,1]*Y[1,1]
    def add_1x2x2(x_row_offset, y_row_offset, z_col_offset):
        """
        Add rank-4 decomposition for X (1x2) * Y (2x2) = Z (1x2).
        X is at A[2, x_row_offset:x_row_offset+2] (row 2 of A, cols 0,1)
        Y is at B[y_row_offset:y_row_offset+2, 0:2] (rows 0,1 of B, cols 0,1)
        Z is at C[2, z_col_offset:z_col_offset+2] (row 2 of C, cols z_col_offset, z_col_offset+1)
        """
        nonlocal col
        # Z[0,0] = X[0,0]*Y[0,0] + X[0,1]*Y[1,0]
        # Term 1: X[0,0]*Y[0,0]
        U[2*m + x_row_offset, col] = 1
        V[(y_row_offset)*p + 0, col] = 1
        W[(z_col_offset)*n + 2, col] = 1
        col += 1
        
        # Term 2: X[0,1]*Y[1,0]
        U[2*m + (x_row_offset+1), col] = 1
        V[(y_row_offset+1)*p + 0, col] = 1
        W[(z_col_offset)*n + 2, col] = 1
        col += 1
        
        # Z[0,1] = X[0,0]*Y[0,1] + X[0,1]*Y[1,1]
        # Term 3: X[0,0]*Y[0,1]
        U[2*m + x_row_offset, col] = 1
        V[(y_row_offset)*p + 1, col] = 1
        W[(z_col_offset+1)*n + 2, col] = 1
        col += 1
        
        # Term 4: X[0,1]*Y[1,1]
        U[2*m + (x_row_offset+1), col] = 1
        V[(y_row_offset+1)*p + 1, col] = 1
        W[(z_col_offset+1)*n + 2, col] = 1
        col += 1
    
    # Helper to add a 1x1x2 decomposition (rank 2)
    # X (1x1) * Y (1x2) = Z (1x2)
    # Z[0,j] = X[0,0] * Y[0,j]
    def add_1x1x2(y_col_offset, z_col_offset):
        """
        Add rank-2 decomposition for X (1x1) * Y (1x2) = Z (1x2).
        X is at A[2,2]
        Y is at B[2, y_col_offset:y_col_offset+2]
        Z is at C[2, z_col_offset:z_col_offset+2]
        """
        nonlocal col
        # Z[0,0] = X[0,0] * Y[0,0]
        U[2*m + 2, col] = 1
        V[2*p + y_col_offset, col] = 1
        W[(z_col_offset)*n + 2, col] = 1
        col += 1
        
        # Z[0,1] = X[0,0] * Y[0,1]
        U[2*m + 2, col] = 1
        V[2*p + (y_col_offset+1), col] = 1
        W[(z_col_offset+1)*n + 2, col] = 1
        col += 1
    
    # Now add all the terms:
    
    # 1. A22 * B22 -> C[0:2, 0:2] (2x2x2, rank 7)
    # A22 at A[0:2, 0:2], B22 at B[0:2, 0:2], C[0:2, 0:2]
    add_strassen_2x2(0, 0, 0, 0, 0, 0)
    
    # 2. A22 * B2p -> C[0:2, 2:4] (2x2x2, rank 7)
    # A22 at A[0:2, 0:2], B2p at B[0:2, 2:4], C[0:2, 2:4]
    add_strassen_2x2(0, 0, 0, 2, 0, 2)
    
    # 3. A21 * B12 -> C[0:2, 0:2] (2x1x2, rank 4)
    # A21 at A[0:2, 2], B12 at B[2, 0:2], C[0:2, 0:2]
    add_2x1x2(0, 0, 0, 0)
    
    # 4. A21 * B1p -> C[0:2, 2:4] (2x1x2, rank 4)
    # A21 at A[0:2, 2], B1p at B[2, 2:4], C[0:2, 2:4]
    add_2x1x2(0, 2, 0, 2)
    
    # 5. A12 * B22 -> C[2, 0:2] (1x2x2, rank 4)
    # A12 at A[2, 0:2], B22 at B[0:2, 0:2], C[2, 0:2]
    add_1x2x2(0, 0, 0)
    
    # 6. A12 * B2p -> C[2, 2:4] (1x2x2, rank 4)
    # A12 at A[2, 0:2], B2p at B[0:2, 2:4], C[2, 2:4]
    add_1x2x2(0, 0, 2)
    
    # 7. A11 * B12 -> C[2, 0:2] (1x1x2, rank 2)
    # A11 at A[2,2], B12 at B[2, 0:2], C[2, 0:2]
    add_1x1x2(0, 0)
    
    # 8. A11 * B1p -> C[2, 2:4] (1x1x2, rank 2)
    # A11 at A[2,2], B1p at B[2, 2:4], C[2, 2:4]
    add_1x1x2(2, 2)
    
    # Verify col == rank
    assert col == rank, f"Expected {rank} columns, got {col}"
    
    # Verify the decomposition is correct
    # T[i*m+j, j*p+k, k*n+i] should equal sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    TARGET_TENSOR = np.zeros((n*m, m*p, p*n), dtype=int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                TARGET_TENSOR[i*m+j, j*p+k, k*n+i] = 1
    
    T_approx = np.einsum('ar,br,cr->abc', U, V, W)
    if not np.allclose(T_approx, TARGET_TENSOR, atol=1e-9):
        # Fallback to schoolbook
        schoolbook_rank = n * m * p
        U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
        V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
        W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
        col_sb = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j][col_sb] = 1
                    V_sb[j * p + k][col_sb] = 1
                    W_sb[k * n + i][col_sb] = 1
                    col_sb += 1
        return {"U": U_sb, "V": V_sb, "W": W_sb}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
