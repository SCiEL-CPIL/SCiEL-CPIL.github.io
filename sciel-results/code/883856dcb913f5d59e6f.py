import numpy as np

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = 2, 4, 4
    
    # We will construct a rank-28 decomposition using 4 Strassen blocks.
    # The 2x4 matrix A is split into A_L (cols 0,1) and A_R (cols 2,3).
    # The 4x4 matrix B is split into 4 2x2 blocks:
    # B_LT (rows 0,1; cols 0,1), B_LB (rows 2,3; cols 0,1)
    # B_RT (rows 0,1; cols 2,3), B_RB (rows 2,3; cols 2,3)
    
    # C_L = A_L * B_LT + A_R * B_LB  (2x2 result)
    # C_R = A_L * B_RT + A_R * B_RB  (2x2 result)
    
    # Each product (e.g., A_L * B_LT) is computed via Strassen (rank 7).
    # Total rank = 4 * 7 = 28.
    
    rank = 28
    U = [[0.0]*rank for _ in range(n*m)]  # 8 x 28
    V = [[0.0]*rank for _ in range(m*p)]  # 16 x 28
    W = [[0.0]*rank for _ in range(p*n)]  # 8 x 28
    
    # Strassen factors for 2x2 * 2x2 -> 2x2
    # M1 = (A11) * (B11 - B22)
    # M2 = (A11 + A21) * B22
    # M3 = (A12 - A22) * (B11 + B21)
    # M4 = A22 * (B22 - B12)
    # M5 = (A11 + A12) * B12
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 + A22) * (B21 - B11)
    
    # Result:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M3 + M2 + M6
    
    # We define a helper to add a Strassen block.
    # a_rows: list of 2 row indices for A (in U)
    # b_rows: list of 2 row indices for B (in V)
    # b_cols: list of 2 col indices for B (in V)
    # c_rows: list of 2 row indices for C (in W)
    # c_cols: list of 2 col indices for C (in W)
    # offset: starting column index in U, V, W
    
    def add_strassen_block(a_rows, b_rows, b_cols, c_rows, c_cols, offset):
        # a_rows: [i1, i2] -> U indices are a_rows[0]*m + j, a_rows[1]*m + j
        # b_rows: [j1, j2] -> V indices are b_rows[0]*p + k, b_rows[1]*p + k
        # b_cols: [k1, k2]
        # c_rows: [i1, i2] -> W indices are c_cols[0]*n + i, c_cols[1]*n + i
        # c_cols: [k1, k2]
        
        # Strassen multipliers for A (coefficients for U)
        # M1: A11
        # M2: A11 + A21
        # M3: A12 - A22
        # M4: A22
        # M5: A11 + A12
        # M6: A21 - A11
        # M7: A12 + A22
        
        # Let A11 = A[i1, j1], A12 = A[i1, j2], A21 = A[i2, j1], A22 = A[i2, j2]
        # U indices:
        # u11 = a_rows[0]*m + b_rows[0]
        # u12 = a_rows[0]*m + b_rows[1]
        # u21 = a_rows[1]*m + b_rows[0]
        # u22 = a_rows[1]*m + b_rows[1]
        
        u11 = a_rows[0]*m + b_rows[0]
        u12 = a_rows[0]*m + b_rows[1]
        u21 = a_rows[1]*m + b_rows[0]
        u22 = a_rows[1]*m + b_rows[1]
        
        # V indices:
        # v11 = b_rows[0]*p + b_cols[0]
        # v12 = b_rows[0]*p + b_cols[1]
        # v21 = b_rows[1]*p + b_cols[0]
        # v22 = b_rows[1]*p + b_cols[1]
        
        v11 = b_rows[0]*p + b_cols[0]
        v12 = b_rows[0]*p + b_cols[1]
        v21 = b_rows[1]*p + b_cols[0]
        v22 = b_rows[1]*p + b_cols[1]
        
        # W indices:
        # w11 = c_cols[0]*n + c_rows[0]
        # w12 = c_cols[1]*n + c_rows[0]
        # w21 = c_cols[0]*n + c_rows[1]
        # w22 = c_cols[1]*n + c_rows[1]
        
        w11 = c_cols[0]*n + c_rows[0]
        w12 = c_cols[1]*n + c_rows[0]
        w21 = c_cols[0]*n + c_rows[1]
        w22 = c_cols[1]*n + c_rows[1]
        
        # For each of the 7 factors, set U, V, W entries.
        # Factor 1: M1 = A11 * (B11 - B22)
        col = offset + 0
        U[u11][col] += 1
        V[v11][col] += 1
        V[v22][col] -= 1
        W[w11][col] += 1
        W[w22][col] += 1
        
        # Factor 2: M2 = (A11 + A21) * B22
        col = offset + 1
        U[u11][col] += 1
        U[u21][col] += 1
        V[v22][col] += 1
        W[w21][col] += 1
        W[w22][col] += 1
        
        # Factor 3: M3 = (A12 - A22) * (B11 + B21)
        col = offset + 2
        U[u12][col] += 1
        U[u22][col] -= 1
        V[v11][col] += 1
        V[v21][col] += 1
        W[w12][col] += 1
        W[w22][col] -= 1
        
        # Factor 4: M4 = A22 * (B22 - B12)
        col = offset + 3
        U[u22][col] += 1
        V[v22][col] += 1
        V[v12][col] -= 1
        W[w11][col] += 1
        W[w21][col] += 1
        
        # Factor 5: M5 = (A11 + A12) * B12
        col = offset + 4
        U[u11][col] += 1
        U[u12][col] += 1
        V[v12][col] += 1
        W[w11][col] -= 1
        W[w12][col] += 1
        
        # Factor 6: M6 = (A21 - A11) * (B11 + B12)
        col = offset + 5
        U[u21][col] += 1
        U[u11][col] -= 1
        V[v11][col] += 1
        V[v12][col] += 1
        W[w22][col] += 1
        
        # Factor 7: M7 = (A12 + A22) * (B21 - B11)
        col = offset + 6
        U[u12][col] += 1
        U[u22][col] += 1
        V[v21][col] += 1
        V[v11][col] -= 1
        W[w11][col] += 1
        W[w12][col] += 1
        
    # Block 1: A_L * B_LT -> C_L (top-left 2x2 of C)
    # A_L: rows 0,1; cols 0,1 of A
    # B_LT: rows 0,1; cols 0,1 of B
    # C_L: rows 0,1; cols 0,1 of C
    add_strassen_block([0, 1], [0, 1], [0, 1], [0, 1], [0, 1], 0)
    
    # Block 2: A_R * B_LB -> C_L (top-left 2x2 of C)
    # A_R: rows 0,1; cols 2,3 of A
    # B_LB: rows 2,3; cols 0,1 of B
    # C_L: rows 0,1; cols 0,1 of C
    add_strassen_block([0, 1], [2, 3], [0, 1], [0, 1], [0, 1], 7)
    
    # Block 3: A_L * B_RT -> C_R (top-right 2x2 of C)
    # A_L: rows 0,1; cols 0,1 of A
    # B_RT: rows 0,1; cols 2,3 of B
    # C_R: rows 0,1; cols 2,3 of C
    add_strassen_block([0, 1], [0, 1], [2, 3], [0, 1], [2, 3], 14)
    
    # Block 4: A_R * B_RB -> C_R (top-right 2x2 of C)
    # A_R: rows 0,1; cols 2,3 of A
    # B_RB: rows 2,3; cols 2,3 of B
    # C_R: rows 0,1; cols 2,3 of C
    add_strassen_block([0, 1], [2, 3], [2, 3], [0, 1], [2, 3], 21)
    
    # Convert to list of lists of integers (or floats if needed, but they are integers here)
    U = [[int(x) for x in row] for row in U]
    V = [[int(x) for x in row] for row in V]
    W = [[int(x) for x in row] for row in W]
    
    return {"U": U, "V": V, "W": W}
