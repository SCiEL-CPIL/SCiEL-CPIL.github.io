import numpy as np
import time

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # 10s safety margin
    
    # Schoolbook solution as fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_solution = {'U': U_school, 'V': V_school, 'W': W_school}
    best_rank = rank_school
    
    # Block decomposition using Strassen for 2x3x4 -> rank 22
    # Partition: A (2x3) = [A_L (2x2) | A_R (2x1)]
    #             B (3x4) = [B_T (2x4); B_B (1x4)]
    # C = A_L * B_T + A_R * B_B
    # A_L * B_T: two 2x2x2 multiplications (left and right halves of B_T), each Strassen rank 7
    # A_R * B_B: outer product 2x1 * 1x4, rank 8
    # Total rank: 7 + 7 + 8 = 22
    
    R_block = 22
    U_block = [[0] * R_block for _ in range(n * m)]  # 6 x 22
    V_block = [[0] * R_block for _ in range(m * p)]  # 12 x 22
    W_block = [[0] * R_block for _ in range(p * n)]  # 8 x 22
    
    # Strassen terms for 2x2x2 multiplication
    # For matrices X (2x2), Y (2x2), Z = X*Y
    # M1 = (X00+X11)*(Y00+Y11)
    # M2 = (X10+X11)*Y00
    # M3 = X00*Y10
    # M4 = X11*Y11
    # M5 = X00*(Y01-Y11)
    # M6 = (X10-X00)*(Y00+Y01)
    # M7 = (X01-X11)*(Y10+Y11)
    # Z00 = M1 - M4 + M7
    # Z01 = M5 + M4
    # Z10 = M2 + M4
    # Z11 = M1 - M2 + M3
    
    # Helper to add a Strassen block for 2x2x2
    # X is 2x2 submatrix of A, Y is 2x2 submatrix of B
    # X indices in A: rows r0,r1; cols c0,c1
    # Y indices in B: rows r0,r1; cols c0,c1
    # Z indices in C: rows r0,r1; cols c0,c1
    def add_strassen(U, V, W, base_col, x_rows, x_cols, y_rows, y_cols, z_rows, z_cols):
        """
        Add Strassen 7-term decomposition for 2x2x2.
        x_rows: [r0, r1] rows of X in A
        x_cols: [c0, c1] cols of X in A
        y_rows: [r0, r1] rows of Y in B
        y_cols: [c0, c1] cols of Y in B
        z_rows: [r0, r1] rows of Z in C
        z_cols: [c0, c1] cols of Z in C
        """
        col = base_col
        # M1 = (X00+X11)*(Y00+Y11)
        # Z00 += M1, Z11 += M1
        for i in x_rows:
            for j in x_cols:
                pass  # We'll handle per-element
        
        # Let's be more explicit
        # X00 is at A[x_rows[0], x_cols[0]], X11 at A[x_rows[1], x_cols[1]]
        # Y00 at B[y_rows[0], y_cols[0]], Y11 at B[y_rows[1], y_cols[1]]
        
        # M1 = (X00+X11)*(Y00+Y11)
        U[x_rows[0]*m + x_cols[0]][col] += 1
        U[x_rows[1]*m + x_cols[1]][col] += 1
        V[y_rows[0]*p + y_cols[0]][col] += 1
        V[y_rows[1]*p + y_cols[1]][col] += 1
        W[z_cols[0]*n + z_rows[0]][col] += 1  # Z00
        W[z_cols[1]*n + z_rows[1]][col] += 1  # Z11
        col += 1
        
        # M2 = (X10+X11)*Y00
        # Z10 += M2
        U[x_rows[1]*m + x_cols[0]][col] += 1
        U[x_rows[1]*m + x_cols[1]][col] += 1
        V[y_rows[0]*p + y_cols[0]][col] += 1
        W[z_cols[0]*n + z_rows[1]][col] += 1  # Z10
        col += 1
        
        # M3 = X00*Y10
        # Z11 += M3
        U[x_rows[0]*m + x_cols[0]][col] += 1
        V[y_rows[1]*p + y_cols[0]][col] += 1
        W[z_cols[1]*n + z_rows[1]][col] += 1  # Z11
        col += 1
        
        # M4 = X11*Y11
        # Z00 += M4, Z01 += M4, Z10 += M4
        U[x_rows[1]*m + x_cols[1]][col] += 1
        V[y_rows[1]*p + y_cols[1]][col] += 1
        W[z_cols[0]*n + z_rows[0]][col] += 1  # Z00
        W[z_cols[1]*n + z_rows[0]][col] += 1  # Z01
        W[z_cols[0]*n + z_rows[1]][col] += 1  # Z10
        col += 1
        
        # M5 = X00*(Y01-Y11)
        # Z01 += M5
        U[x_rows[0]*m + x_cols[0]][col] += 1
        V[y_rows[0]*p + y_cols[1]][col] += 1
        V[y_rows[1]*p + y_cols[1]][col] -= 1
        W[z_cols[1]*n + z_rows[0]][col] += 1  # Z01
        col += 1
        
        # M6 = (X10-X00)*(Y00+Y01)
        # Z10 += M6
        U[x_rows[1]*m + x_cols[0]][col] += 1
        U[x_rows[0]*m + x_cols[0]][col] -= 1
        V[y_rows[0]*p + y_cols[0]][col] += 1
        V[y_rows[0]*p + y_cols[1]][col] += 1
        W[z_cols[0]*n + z_rows[1]][col] += 1  # Z10
        col += 1
        
        # M7 = (X01-X11)*(Y10+Y11)
        # Z00 += M7
        U[x_rows[0]*m + x_cols[1]][col] += 1
        U[x_rows[1]*m + x_cols[1]][col] -= 1
        V[y_rows[1]*p + y_cols[0]][col] += 1
        V[y_rows[1]*p + y_cols[1]][col] += 1
        W[z_cols[0]*n + z_rows[0]][col] += 1  # Z00
        col += 1
        
        return col
    
    # A_L * B_T1: A_L is rows 0,1 cols 0,1 of A; B_T1 is rows 0,1 cols 0,1 of B
    # Z is rows 0,1 cols 0,1 of C
    col = add_strassen(U_block, V_block, W_block, 0, [0,1], [0,1], [0,1], [0,1], [0,1], [0,1])
    
    # A_L * B_T2: A_L is rows 0,1 cols 0,1 of A; B_T2 is rows 0,1 cols 2,3 of B
    # Z is rows 0,1 cols 2,3 of C
    col = add_strassen(U_block, V_block, W_block, col, [0,1], [0,1], [0,1], [2,3], [0,1], [2,3])
    
    # A_R * B_B: A_R is rows 0,1 col 2 of A (2x1); B_B is row 2 cols 0,1,2,3 of B (1x4)
    # This is outer product: C[i,k] += A_R[i] * B_B[k]
    # 8 terms: for each i in {0,1}, k in {0,1,2,3}
    for i in range(2):
        for k in range(4):
            U_block[i*m + 2][col] = 1
            V_block[2*p + k][col] = 1
            W_block[k*n + i][col] = 1
            col += 1
    
    # Verify col == 22
    assert col == R_block, f"Expected 22 cols, got {col}"
    
    # Verify the decomposition
    def compute_error(U, V, W, R):
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    for r in range(R):
                        s += U[idx_u][r] * V[idx_v][r] * W[idx_w][r]
                    diff = s - 1.0
                    err += diff * diff
        return err
    
    err_block = compute_error(U_block, V_block, W_block, R_block)
    if err_block < 1e-10:
        best_solution = {'U': U_block, 'V': V_block, 'W': W_block}
        best_rank = R_block
    else:
        # Fallback to schoolbook if block decomposition is incorrect
        pass
    
    return best_solution
