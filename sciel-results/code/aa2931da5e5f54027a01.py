"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s kill
    
    # Rank 72 decomposition based on 4 blocks of 2x4x3 (rank 18 each)
    # Each 2x4x3 is split into two 2x2x3 (rank 9 each)
    # Each 2x2x3 is split into 2x2x2 (rank 7, Strassen) + 2x2x1 (rank 2)
    
    rank = 72
    U = [[0] * rank for _ in range(n * m)]  # 16 x 72
    V = [[0] * rank for _ in range(m * p)]  # 24 x 72
    W = [[0] * rank for _ in range(p * n)]  # 24 x 72
    
    # Strassen coefficients for 2x2 * 2x2
    # M1 = (a+d)(e+h) -> C11 += 1, C22 += 1
    # M2 = (c+d)e -> C21 += 1
    # M3 = a(f-h) -> C12 += 1
    # M4 = d(g-e) -> C11 += 1, C21 += 1
    # M5 = (a+b)h -> C11 -= 1, C12 += 1
    # M6 = (c-a)(e+f) -> C22 += 1
    # M7 = (b-d)(g+h) -> C11 += 1, C22 -= 1
    
    # Local indices for 2x2 A: (0,0), (0,1), (1,0), (1,1)
    # Local indices for 2x2 B: (0,0), (0,1), (1,0), (1,1)
    # Local indices for 2x2 C: (0,0), (0,1), (1,0), (1,1)
    
    strassen = [
        # (A_coeffs, B_coeffs, C_coeffs)
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1, (1,1): -1}),
    ]
    
    col = 0
    for I in range(2):  # Row block of A (0, 1)
        for J in range(2):  # Col block of B (0, 1) -> Cols 0-2, 3-5
        
            # Part 1: A_L (cols 0,1) * B_T (rows 0,1)
            # A_L is 2x2, B_T is 2x3
            # Split B_T into B_T_left (cols 0,1) and B_T_right (col 2)
            
            # B_T_left * A_L (Rank 7)
            for r in range(7):
                A_coeffs, B_coeffs, C_coeffs = strassen[r]
                
                # A_L: local (i, j) where i in 0,1, j in 0,1
                # Global A: (2I + i, j)
                for (i, j), coeff in A_coeffs.items():
                    u_idx = (2*I + i) * m + j
                    U[u_idx][col] += coeff
                
                # B_T: local (j, k) where j in 0,1, k in 0,1 (for left part)
                # Global B: (j, 3*J + k)
                for (j, k), coeff in B_coeffs.items():
                    v_idx = j * p + (3*J + k)
                    V[v_idx][col] += coeff
                
                # C: local (i, k) where i in 0,1, k in 0,1
                # Global C: (2I + i, 3*J + k)
                # W index: k*n + i
                for (i, k), coeff in C_coeffs.items():
                    w_idx = (3*J + k) * n + (2*I + i)
                    W[w_idx][col] += coeff
                
                col += 1
            
            # B_T_right * A_L (Rank 2)
            # B_T_right is col 2 of B_T. Global col 3*J + 2.
            # C col is 3*J + 2.
            
            # Term 1: C[0, 2] = A[0,0]*B[0,2] + A[0,1]*B[1,2]
            # A coeffs: (0,0):1, (0,1):1
            # B coeffs: (0,2):1, (1,2):1
            # C coeffs: (0,2):1
            
            U[(2*I + 0) * m + 0][col] += 1
            U[(2*I + 0) * m + 1][col] += 1
            V[0 * p + (3*J + 2)][col] += 1
            V[1 * p + (3*J + 2)][col] += 1
            W[(3*J + 2) * n + (2*I + 0)][col] += 1
            col += 1
            
            # Term 2: C[1, 2] = A[1,0]*B[0,2] + A[1,1]*B[1,2]
            # A coeffs: (1,0):1, (1,1):1
            # B coeffs: (0,2):1, (1,2):1
            # C coeffs: (1,2):1
            
            U[(2*I + 1) * m + 0][col] += 1
            U[(2*I + 1) * m + 1][col] += 1
            V[0 * p + (3*J + 2)][col] += 1
            V[1 * p + (3*J + 2)][col] += 1
            W[(3*J + 2) * n + (2*I + 1)][col] += 1
            col += 1
            
            # Part 2: A_R (cols 2,3) * B_B (rows 2,3)
            # A_R is 2x2 (local cols 0,1 map to global 2,3)
            # B_B is 2x3 (local rows 0,1 map to global 2,3)
            
            # B_B_left * A_R (Rank 7)
            for r in range(7):
                A_coeffs, B_coeffs, C_coeffs = strassen[r]
                
                # A_R: local (i, j) where i in 0,1, j in 0,1
                # Global A: (2I + i, 2 + j)
                for (i, j), coeff in A_coeffs.items():
                    u_idx = (2*I + i) * m + (2 + j)
                    U[u_idx][col] += coeff
                
                # B_B: local (j, k) where j in 0,1, k in 0,1
                # Global B: (2 + j, 3*J + k)
                for (j, k), coeff in B_coeffs.items():
                    v_idx = (2 + j) * p + (3*J + k)
                    V[v_idx][col] += coeff
                
                # C: local (i, k) where i in 0,1, k in 0,1
                # Global C: (2I + i, 3*J + k)
                for (i, k), coeff in C_coeffs.items():
                    w_idx = (3*J + k) * n + (2*I + i)
                    W[w_idx][col] += coeff
                
                col += 1
            
            # B_B_right * A_R (Rank 2)
            # B_B_right is col 2 of B_B. Global col 3*J + 2.
            # C col is 3*J + 2.
            
            # Term 1: C[0, 2] += A_R[0,0]*B_B[0,2] + A_R[0,1]*B_B[1,2]
            # Global A: (2I, 2), (2I, 3)
            # Global B: (2, 3J+2), (3, 3J+2)
            
            U[(2*I + 0) * m + 2][col] += 1
            U[(2*I + 0) * m + 3][col] += 1
            V[2 * p + (3*J + 2)][col] += 1
            V[3 * p + (3*J + 2)][col] += 1
            W[(3*J + 2) * n + (2*I + 0)][col] += 1
            col += 1
            
            # Term 2: C[1, 2] += A_R[1,0]*B_B[0,2] + A_R[1,1]*B_B[1,2]
            # Global A: (2I+1, 2), (2I+1, 3)
            # Global B: (2, 3J+2), (3, 3J+2)
            
            U[(2*I + 1) * m + 2][col] += 1
            U[(2*I + 1) * m + 3][col] += 1
            V[2 * p + (3*J + 2)][col] += 1
            V[3 * p + (3*J + 2)][col] += 1
            W[(3*J + 2) * n + (2*I + 1)][col] += 1
            col += 1

    # Verify the decomposition
    def verify_decomposition(U_mat, V_mat, W_mat, tol=1e-9):
        U_arr = np.array(U_mat, dtype=float)
        V_arr = np.array(V_mat, dtype=float)
        W_arr = np.array(W_mat, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_arr[i * m + j], V_arr[j * p + k] * W_arr[k * n + i])
                    if abs(val - 1.0) > tol:
                        return False
        return True

    if not verify_decomposition(U, V, W):
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
