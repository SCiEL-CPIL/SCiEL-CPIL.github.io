import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    # Total rank: 22 (from 4x5x4) + 18 (from 2x4x9) = 40
    rank = 40
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: A_L (2x2) * B_T (2x4) using 4x5x4 Strassen scheme
    # A indices: i*m+j for i in {0,1}, j in {0,1}
    # B indices: j*p+k for j in {0,1}, k in {0,1,2,3}
    # C indices: k*n+i for k in {0,1,2,3}, i in {0,1}
    
    for k in range(4):
        v_e = 0 * p + k  # B_{0,k}
        v_g = 1 * p + k  # B_{1,k}
        
        u_a = 0 * m + 0  # A_{00}
        u_b = 0 * m + 1  # A_{01}
        u_c = 1 * m + 0  # A_{10}
        u_d = 1 * m + 1  # A_{11}
        
        w_0k = k * n + 0  # C_{0,k}
        w_1k = k * n + 1  # C_{1,k}
        
        # P1 = a*e
        U[u_a][col] += 1
        V[v_e][col] += 1
        W[w_0k][col] += 1
        
        # P2 = b*g
        U[u_b][col+1] += 1
        V[v_g][col+1] += 1
        W[w_0k][col+1] += 1
        
        # P3 = c*e
        U[u_c][col+2] += 1
        V[v_e][col+2] += 1
        W[w_1k][col+2] += 1
        
        # P4 = d*g
        U[u_d][col+3] += 1
        V[v_g][col+3] += 1
        W[w_1k][col+3] += 1
        
        # P5 = (a+d)*(e+g)
        U[u_a][col+4] += 1
        U[u_d][col+4] += 1
        V[v_e][col+4] += 1
        V[v_g][col+4] += 1
        
        # C0k = P1 + P2 + P5 - P4 - P3 + ... 
        # Actually, let's use a cleaner 4x5x4 scheme.
        # Standard Strassen for 2x2x2 uses 7 mults.
        # For 2x2x4, we can use two 2x2x2 blocks + cross terms? No.
        # Let's stick to the simple 4 mults per column if we allow rank 18 for this part.
        # 4 columns * 4 mults = 16. Plus 18 for Part 2 = 34.
        # Let's try to get Part 1 rank to 16.
        
    # Reset and use simple 4 mults per column for Part 1 (rank 16)
    rank = 16 + 18
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for k in range(4):
        v_e = 0 * p + k
        v_g = 1 * p + k
        
        u_a = 0
        u_b = 1
        u_c = 3
        u_d = 4
        
        w_0k = k * n + 0
        w_1k = k * n + 1
        
        U[u_a][col] += 1
        V[v_e][col] += 1
        W[w_0k][col] += 1
        
        U[u_b][col+1] += 1
        V[v_g][col+1] += 1
        W[w_0k][col+1] += 1
        
        U[u_c][col+2] += 1
        V[v_e][col+2] += 1
        W[w_1k][col+2] += 1
        
        U[u_d][col+3] += 1
        V[v_g][col+3] += 1
        W[w_1k][col+3] += 1
        
        col += 4
        
    # Part 2: A_R (2x1) * B_B (1x9)
    # A indices: 2 (A_{02}), 5 (A_{12})
    # B indices: 18..26 (B_{2,k})
    # C indices: k*n+i
    
    for k in range(9):
        v_b = 2 * p + k
        
        U[2][col] += 1
        V[v_b][col] += 1
        W[k * n + 0][col] += 1
        
        U[5][col+1] += 1
        V[v_b][col+1] += 1
        W[k * n + 1][col+1] += 1
        
        col += 2
        
    return {"U": U, "V": V, "W": W}
