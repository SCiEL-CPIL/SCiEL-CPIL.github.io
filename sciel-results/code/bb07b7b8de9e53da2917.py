import numpy as np
import time

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    # A: n x m (2x3), B: m x p (3x9), C: n x p (2x9)
    # C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # Tensor indices:
    # U index: i*m + j  (0..n*m-1 = 0..5)
    # V index: j*p + k  (0..m*p-1 = 0..26)
    # W index: k*n + i  (0..p*n-1 = 0..17)
    
    # Rank-50 decomposition using Strassen for 2x2 blocks + schoolbook for the rest
    rank = 50
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: A[0:2, 0:2] * B[0:2, :]
    # A_L is 2x2, B_T is 2x9
    # Split B_T into 4 blocks of 2 columns + 1 block of 1 column
    
    # Blocks of 2x2 * 2x2 using Strassen
    for b in range(4):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # B indices: B[0,k0], B[0,k1], B[1,k0], B[1,k1]
        v_e = 0 * p + k0
        v_f = 0 * p + k1
        v_g = 1 * p + k0
        v_h = 1 * p + k1
        
        # A indices: A[0,0], A[0,1], A[1,0], A[1,1]
        u_a = 0 * m + 0
        u_b = 0 * m + 1
        u_c = 1 * m + 0
        u_d = 1 * m + 1
        
        # Strassen products (7 multiplications)
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
        
        # Output: C[0,k0], C[0,k1], C[1,k0], C[1,k1]
        # W indices: k*n+i
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
    
    # Last block: 2x2 * 2x1 (column k=8)
    k8 = 8
    v_e = 0 * p + k8
    v_g = 1 * p + k8
    
    u_a = 0 * m + 0
    u_b = 0 * m + 1
    u_c = 1 * m + 0
    u_d = 1 * m + 1
    
    # Schoolbook: 4 products
    # P1 = a*e -> C[0,k8]
    U[u_a][col] += 1
    V[v_e][col] += 1
    W[k8 * n + 0][col] += 1
    
    # P2 = b*g -> C[0,k8]
    U[u_b][col+1] += 1
    V[v_g][col+1] += 1
    W[k8 * n + 0][col+1] += 1
    
    # P3 = c*e -> C[1,k8]
    U[u_c][col+2] += 1
    V[v_e][col+2] += 1
    W[k8 * n + 1][col+2] += 1
    
    # P4 = d*g -> C[1,k8]
    U[u_d][col+3] += 1
    V[v_g][col+3] += 1
    W[k8 * n + 1][col+3] += 1
    
    col += 4
    
    # Part 2: A[0:2, 2:3] * B[2:3, :]
    # A_R is 2x1, B_B is 1x9
    # Schoolbook: 18 products
    
    u_ar0 = 0 * m + 2  # A[0,2]
    u_ar1 = 1 * m + 2  # A[1,2]
    
    for k in range(9):
        v_b = 2 * p + k  # B[2,k]
        
        # Q_k = A[0,2] * B[2,k] -> C[0,k]
        U[u_ar0][col] += 1
        V[v_b][col] += 1
        W[k * n + 0][col] += 1
        
        # R_k = A[1,2] * B[2,k] -> C[1,k]
        U[u_ar1][col+1] += 1
        V[v_b][col+1] += 1
        W[k * n + 1][col+1] += 1
        
        col += 2
    
    # Verify: col should be 50
    assert col == rank, f"Expected col={rank}, got col={col}"
    
    return {"U": U, "V": V, "W": W}
