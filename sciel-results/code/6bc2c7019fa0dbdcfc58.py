"""Matrix multiplication 2x3x9 - rank-50 decomposition using Strassen on 2x2 blocks."""
import time
import numpy as np

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    """
    Return a rank-50 decomposition for 2x3x9 matrix multiplication.
    
    Strategy: Split the inner dimension (size 3) into (2+1).
    - The 2x2 part uses Strassen's algorithm on 2x2x2 blocks.
    - The 1x1 part uses schoolbook multiplication.
    
    The trace convention is W[k*n+i, r].
    """
    n, m, p = DIMENSIONS  # n=2, m=3, p=9
    
    rank = 50
    U = [[0] * rank for _ in range(n * m)]  # shape (6, 50)
    V = [[0] * rank for _ in range(m * p)]  # shape (27, 50)
    W = [[0] * rank for _ in range(p * n)]  # shape (18, 50)
    
    col = 0
    
    # Part 1: A[:,0:2] (2x2) * B[0:2,:] (2x9)
    # A indices: 0=A[0,0], 1=A[0,1], 3=A[1,0], 4=A[1,1]
    # B indices: 0-8=B[0,0:9], 9-17=B[1,0:9]
    
    u_a = 0  # A[0,0]
    u_b = 1  # A[0,1]
    u_c = 3  # A[1,0]
    u_d = 4  # A[1,1]
    
    # 4 blocks of 2x2 (columns 0-7)
    for b in range(4):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        v_e = 0 * p + k0  # B[0, k0]
        v_f = 0 * p + k1  # B[0, k1]
        v_g = 1 * p + k0  # B[1, k0]
        v_h = 1 * p + k1  # B[1, k1]
        
        # Strassen's 7 products for 2x2x2
        # M1 = (a+d)(e+h)
        U[u_a][col] += 1
        U[u_d][col] += 1
        V[v_e][col] += 1
        V[v_h][col] += 1
        
        # M2 = (c+d)e
        U[u_c][col+1] += 1
        U[u_d][col+1] += 1
        V[v_e][col+1] += 1
        
        # M3 = a(f-h)
        U[u_a][col+2] += 1
        V[v_f][col+2] += 1
        V[v_h][col+2] -= 1
        
        # M4 = d(g-e)
        U[u_d][col+3] += 1
        V[v_g][col+3] += 1
        V[v_e][col+3] -= 1
        
        # M5 = (a+b)h
        U[u_a][col+4] += 1
        U[u_b][col+4] += 1
        V[v_h][col+4] += 1
        
        # M6 = (c-a)(e+f)
        U[u_c][col+5] += 1
        U[u_a][col+5] -= 1
        V[v_e][col+5] += 1
        V[v_f][col+5] += 1
        
        # M7 = (b-d)(g+h)
        U[u_b][col+6] += 1
        U[u_d][col+6] -= 1
        V[v_g][col+6] += 1
        V[v_h][col+6] += 1
        
        # Output indices: C[0,k0], C[0,k1], C[1,k0], C[1,k1]
        w_00 = k0 * n + 0
        w_01 = k1 * n + 0
        w_10 = k0 * n + 1
        w_11 = k1 * n + 1
        
        # C[0,k0] = M1 + M4 - M5 + M7
        W[w_00][col] += 1
        W[w_00][col+3] += 1
        W[w_00][col+4] -= 1
        W[w_00][col+6] += 1
        
        # C[0,k1] = M3 + M5
        W[w_01][col+2] += 1
        W[w_01][col+4] += 1
        
        # C[1,k0] = M2 + M4
        W[w_10][col+1] += 1
        W[w_10][col+3] += 1
        
        # C[1,k1] = M1 - M2 + M3 + M6
        W[w_11][col] += 1
        W[w_11][col+1] -= 1
        W[w_11][col+2] += 1
        W[w_11][col+5] += 1
        
        col += 7
    
    # 1 block of 2x1 (column 8)
    k8 = 8
    v_e = 0 * p + k8  # B[0, 8]
    v_g = 1 * p + k8  # B[1, 8]
    
    # P1 = a*e
    U[u_a][col] += 1
    V[v_e][col] += 1
    # P2 = b*g
    U[u_b][col+1] += 1
    V[v_g][col+1] += 1
    # P3 = c*e
    U[u_c][col+2] += 1
    V[v_e][col+2] += 1
    # P4 = d*g
    U[u_d][col+3] += 1
    V[v_g][col+3] += 1
    
    w_08 = k8 * n + 0
    w_18 = k8 * n + 1
    
    # C[0,8] = P1 + P2
    W[w_08][col] += 1
    W[w_08][col+1] += 1
    
    # C[1,8] = P3 + P4
    W[w_18][col+2] += 1
    W[w_18][col+3] += 1
    
    col += 4
    
    # Part 2: A[:,2:3] (2x1) * B[2:3,:] (1x9)
    # A indices: 2=A[0,2], 5=A[1,2]
    # B indices: 18-26=B[2,0:9]
    
    u_ar0 = 2  # A[0,2]
    u_ar1 = 5  # A[1,2]
    
    for k in range(9):
        v_b = 2 * p + k  # B[2, k]
        
        # Q_k = A[0,2] * B[2,k]
        U[u_ar0][col] += 1
        V[v_b][col] += 1
        W[k * n + 0][col] += 1
        
        # R_k = A[1,2] * B[2,k]
        U[u_ar1][col+1] += 1
        V[v_b][col+1] += 1
        W[k * n + 1][col+1] += 1
        
        col += 2
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
