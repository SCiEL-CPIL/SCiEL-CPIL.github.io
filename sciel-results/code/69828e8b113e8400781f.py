"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill
    
    # Schoolbook rank is n*m*p = 96
    best_rank = n * m * p
    best_result = None
    
    def build_schoolbook():
        """Standard schoolbook decomposition with rank n*m*p."""
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
    
    def validate_einsum(U, V, W, rank):
        """
        Validation using einsum for speed.
        T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        """
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        
        U_r = U_arr.reshape(n, m, rank)
        V_r = V_arr.reshape(m, p, rank)
        W_r = W_arr.reshape(p, n, rank)
        
        T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
        
        return np.allclose(T, 1.0, atol=1e-8)

    def build_strassen_222():
        """Returns U, V, W for 2x2x2 matrix multiplication with rank 7."""
        U = [
            [-1, 1, 0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, -1, 1],
            [0, 1, 1, 0, 0, 0, 0],
            [1, 0, 1, -1, 1, 0, 1]
        ]
        V = [
            [0, 0, 1, 1, 0, 1, 0],
            [0, 0, 0, 0, 1, 0, 1],
            [1, 0, 0, 0, 0, 1, 0],
            [-1, 1, 0, 1, 0, 0, -1]
        ]
        W = [
            [1, 0, 0, 1, -1, 0, 1],
            [0, 0, 1, 0, 1, 0, 0],
            [0, 1, 0, 1, 0, 0, 0],
            [1, -1, 1, 0, 0, 1, 0]
        ]
        return U, V, W

    def build_221():
        """Returns U, V, W for 2x2x1 matrix multiplication with rank 4."""
        U = [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ]
        # V is 2x3 (m=2, p=3) but we only use col 2 and 5 (b02, b12)
        # Indices: 0:b00, 1:b01, 2:b02, 3:b10, 4:b11, 5:b12
        V = [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [1, 0, 1, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 1, 0, 1]
        ]
        # W is 3x2 (p=3, n=2) but we only use row 2 (c02) and 5 (c12) is not right
        # W indices: k*n + i.
        # c00: 0, c01: 1, c02: 2
        # c10: 3, c11: 4, c12: 5
        # We want c02 and c12.
        W = [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 1, 1]
        ]
        return U, V, W

    def build_223():
        """Returns U, V, W for 2x2x3 matrix multiplication with rank 11."""
        U_222, V_222, W_222 = build_strassen_222()
        U_221, V_221, W_221 = build_221()
        
        U = [row + [0, 0, 0, 0] for row in U_222]
        for i in range(4):
            U[i] = U_222[i] + U_221[i]
            
        V = []
        for i in range(6):
            if i < 4:
                V.append(V_222[i] + [0, 0, 0, 0])
            else:
                V.append(V_221[i] + [0, 0, 0, 0])
                
        W = []
        for i in range(6):
            if i < 4:
                W.append(W_222[i] + [0, 0, 0, 0])
            else:
                W.append(W_221[i] + [0, 0, 0, 0])
                
        return U, V, W

    def build_246():
        """Returns U, V, W for 2x4x6 matrix multiplication with rank 44."""
        U_223, V_223, W_223 = build_223()
        
        # X is 2x4, Y is 4x6.
        # X = [X1, X2], Y = [[Y11, Y12], [Y21, Y22]]
        # M = X1 Y1 + X2 Y2
        # Y1 = [[Y11], [Y21]], Y2 = [[Y12], [Y22]]
        # X1 Y1 is 2x3. X1 is 2x2, Y1 is 4x3.
        # Y1 = [Y11; Y21].
        # X1 Y1 = X1 [Y11; Y21] = X1 Y11 + X1 Y21.
        # This is sum of two 2x2x3 multiplications.
        
        U = []
        V = []
        W = []
        
        # Part 1: X1 Y11 (2x2x3)
        # X1 indices in X (2x4): 0,1,2,3 -> 0,1,2,3
        # Y11 indices in Y (4x6): 0,1,2, 4,5,6 -> 0,1,2, 4,5,6
        # Wait, Y is 4x6.
        # Y11 is top-left 2x3. Indices: rows 0,1; cols 0,1,2.
        # Y12 is top-right 2x3. Indices: rows 0,1; cols 3,4,5.
        # Y21 is bottom-left 2x3. Indices: rows 2,3; cols 0,1,2.
        # Y22 is bottom-right 2x3. Indices: rows 2,3; cols 3,4,5.
        
        # We need to map the 2x2x3 decomposition to the larger indices.
        
        # Helper to place a 2x2x3 decomposition into a larger context
        def place_223(U_base, V_base, W_base, x_offset, y_row_offset, y_col_offset):
            # x_offset: 0 or 2 (for X1 or X2)
            # y_row_offset: 0 or 2 (for Y1 or Y2)
            # y_col_offset: 0 or 3 (for Y11/Y21 or Y12/Y22)
            
            U_new = [[0]*44 for _ in range(8)] # 2x4
            V_new = [[0]*44 for _ in range(24)] # 4x6
            W_new = [[0]*44 for _ in range(12)] # 6x2
            
            # U: 2x2 -> 2x4
            for i in range(2):
                for c in range(11):
                    U_new[x_offset + i][c] = U_base[i][c]
                    
            # V: 2x3 -> 4x6
            for j in range(2):
                for k in range(3):
                    for c in range(11):
                        V_new[(y_row_offset + j) * 6 + (y_col_offset + k)][c] = V_base[j * 3 + k][c]
                        
            # W: 3x2 -> 6x2
            for k in range(3):
                for i in range(2):
                    for c in range(11):
                        W_new[(y_col_offset + k) * 2 + i][c] = W_base[k * 2 + i][c]
                        
            return U_new, V_new, W_new

        # Sum the 4 placements
        U_sum = [[0]*44 for _ in range(8)]
        V_sum = [[0]*44 for _ in range(24)]
        W_sum = [[0]*44 for _ in range(12)]
        
        placements = [
            (0, 0, 0), # X1, Y11
            (0, 2, 0), # X1, Y21
            (2, 0, 3), # X2, Y12
            (2, 2, 3)  # X2, Y22
        ]
        
        for x_off, y_row_off, y_col_off in placements:
            U_p, V_p, W_p = place_223(U_223, V_223, W_223, x_off, y_row_off, y_col_off)
            for i in range(8):
                for c in range(44):
                    U_sum[i][c] += U_p[i][c]
            for i in range(24):
                for c in range(44):
                    V_sum[i][c] += V_p[i][c]
            for i in range(12):
                for c in range(44):
                    W_sum[i][c] += W_p[i][c]
                    
        return U_sum, V_sum, W_sum

    def build_446():
        """Returns U, V, W for 4x4x6 matrix multiplication with rank 88."""
        U_246, V_246, W_246 = build_246()
        
        # A is 4x4, B is 4x6.
        # A = [[A00, A01], [A10, A11]]
        # B is 4x6.
        # C = [[C00, C01, C02], [C10, C11, C12]]
        # C0 = A0 B, C1 = A1 B
        # A0 is 2x4, A1 is 2x4.
        # So C0 = A0 B (2x4 x 4x6 -> 2x6)
        # C1 = A1 B (2x4 x 4x6 -> 2x6)
        
        U = [[0]*88 for _ in range(16)] # 4x4
        V = [[0]*88 for _ in range(24)] # 4x6
        W = [[0]*88 for _ in range(24)] # 6x4
        
        # Place C0 (rows 0,1 of C)
        for i in range(2):
            for c in range(44):
                U[i][c] = U_246[i][c]
                
        # Place C1 (rows 2,3 of C)
        for i in range(2):
            for c in range(44):
                U[4 + i][c + 44] = U_246[i][c]
                
        # V is same for both (it's B)
        for i in range(24):
            for c in range(44):
                V[i][c] = V_246[i][c]
                V[i][c + 44] = V_246[i][c]
                
        # W: 6x4
        # C0 is 2x6. W indices: k*4 + i.
        # C1 is 2x6. W indices: k*4 + 2 + i.
        for k in range(6):
            for i in range(2):
                for c in range(44):
                    W[k * 4 + i][c] = W_246[k * 2 + i][c]
                    W[k * 4 + 2 + i][c + 44] = W_246[k * 2 + i][c]
                    
        return U, V, W

    # Build the rank 88 decomposition
    U, V, W = build_446()
    result = {"U": U, "V": V, "W": W}
    
    if validate_einsum(U, V, W, 88):
        best_result = result
        best_rank = 88
    else:
        # Fallback to schoolbook
        school = build_schoolbook()
        if validate_einsum(school["U"], school["V"], school["W"], best_rank):
            best_result = school
    
    if best_result is None:
        best_result = build_schoolbook()
        
    return best_result
# EVOLVE-BLOCK-END

