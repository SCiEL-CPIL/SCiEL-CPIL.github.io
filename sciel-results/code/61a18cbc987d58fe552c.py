import numpy as np
import time

DIMENSIONS = [2, 3, 9]

def construct():
    """
    Returns a rank-50 exact tensor decomposition for 2x3 * 3x9 matrix multiplication.
    
    The decomposition uses:
    - Strassen's algorithm on four 2x2 blocks of the left part (rank 7 each = 28)
    - Schoolbook for the last column of the left part (rank 4)
    - Schoolbook for the right 2x1 * 1x9 part (rank 18)
    
    Total rank: 28 + 4 + 18 = 50
    """
    n, m, p = DIMENSIONS  # 2, 3, 9
    rank = 50
    
    U = np.zeros((n * m, rank), dtype=np.int64)  # 6 x 50
    V = np.zeros((m * p, rank), dtype=np.int64)  # 27 x 50
    W = np.zeros((p * n, rank), dtype=np.int64)  # 18 x 50
    
    col = 0
    
    # Part 1: A_L (2x2) * B_T (2x9) using Strassen
    # B_T has rows 0 and 1 (first two rows of B, each of length 9)
    for b in range(4):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # V indices for B_T block
        v_e = 0 * p + k0  # B[0, k0]
        v_f = 0 * p + k1  # B[0, k1]
        v_g = 1 * p + k0  # B[1, k0]
        v_h = 1 * p + k1  # B[1, k1]
        
        # U indices for A_L block
        u_a = 0 * m + 0  # A[0, 0]
        u_b = 0 * m + 1  # A[0, 1]
        u_c = 1 * m + 0  # A[1, 0]
        u_d = 1 * m + 1  # A[1, 1]
        
        # Strassen's 7 products for 2x2 * 2x2
        # M1 = (A00+A11)*(B00+B11)
        U[u_a][col] += 1; U[u_d][col] += 1
        V[v_e][col] += 1; V[v_h][col] += 1
        # M2 = (A10+A11)*B00
        U[u_c][col+1] += 1; U[u_d][col+1] += 1
        V[v_e][col+1] += 1
        # M3 = A00*(B01-B11)
        U[u_a][col+2] += 1
        V[v_f][col+2] += 1; V[v_h][col+2] -= 1
        # M4 = A11*(B10-B00)
        U[u_d][col+3] += 1
        V[v_g][col+3] += 1; V[v_e][col+3] -= 1
        # M5 = (A00+A01)*B11
        U[u_a][col+4] += 1; U[u_b][col+4] += 1
        V[v_h][col+4] += 1
        # M6 = (A10-A00)*(B00+B01)
        U[u_c][col+5] += 1; U[u_a][col+5] -= 1
        V[v_e][col+5] += 1; V[v_f][col+5] += 1
        # M7 = (A01-A10)*(B10+B11)
        U[u_b][col+6] += 1; U[u_d][col+6] -= 1
        V[v_g][col+6] += 1; V[v_h][col+6] += 1
        
        # W indices for output block (2x2)
        # C[0,0] = M1 + M4 - M5 - M7
        w_00 = k0 * n + 0
        W[w_00][col] += 1; W[w_00][col+3] += 1; W[w_00][col+4] -= 1; W[w_00][col+6] += 1
        # C[0,1] = M3 + M5
        w_01 = k1 * n + 0
        W[w_01][col+2] += 1; W[w_01][col+4] += 1
        # C[1,0] = M2 + M4
        w_10 = k0 * n + 1
        W[w_10][col+1] += 1; W[w_10][col+3] += 1
        # C[1,1] = M1 - M2 + M3 + M6
        w_11 = k1 * n + 1
        W[w_11][col] += 1; W[w_11][col+1] -= 1; W[w_11][col+2] += 1; W[w_11][col+5] += 1
        
        col += 7
    
    # Last column of B_T (k=8), schoolbook for 2x1 * 1x1
    k8 = 8
    v_e = 0 * p + k8  # B[0, 8]
    v_g = 1 * p + k8  # B[1, 8]
    u_a = 0 * m + 0  # A[0, 0]
    u_b = 0 * m + 1  # A[0, 1]
    u_c = 1 * m + 0  # A[1, 0]
    u_d = 1 * m + 1  # A[1, 1]
    
    # C[0,8] = A[0,0]*B[0,8] + A[0,1]*B[1,8]
    U[u_a][col] += 1; V[v_e][col] += 1
    U[u_b][col+1] += 1; V[v_g][col+1] += 1
    W[k8 * n + 0][col] += 1; W[k8 * n + 0][col+1] += 1
    
    # C[1,8] = A[1,0]*B[0,8] + A[1,1]*B[1,8]
    U[u_c][col+2] += 1; V[v_e][col+2] += 1
    U[u_d][col+3] += 1; V[v_g][col+3] += 1
    W[k8 * n + 1][col+2] += 1; W[k8 * n + 1][col+3] += 1
    
    col += 4
    
    # Part 2: A_R (2x1) * B_B (1x9) using schoolbook
    # A_R is column 2 of A (indices A[0,2] and A[1,2])
    # B_B is row 2 of B (indices B[2,0] through B[2,8])
    u_ar0 = 0 * m + 2  # A[0, 2]
    u_ar1 = 1 * m + 2  # A[1, 2]
    
    for k in range(9):
        v_b = 2 * p + k  # B[2, k]
        # C[0, k] += A[0,2] * B[2,k]
        U[u_ar0][col] += 1; V[v_b][col] += 1; W[k * n + 0][col] += 1
        # C[1, k] += A[1,2] * B[2,k]
        U[u_ar1][col+1] += 1; V[v_b][col+1] += 1; W[k * n + 1][col+1] += 1
        col += 2
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
