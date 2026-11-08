import numpy as np
import time

DIMENSIONS = [2, 3, 9]

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    # 3 blocks of 2x2 * 2x3 (9 mults each) + 2x1 * 1x9 (18 mults) = 27 + 18 = 45
    rank = 45
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # A indices: i*m+j (A is 2x3, so i=0,1; j=0,1,2)
    # B indices: j*p+k (B is 3x9, so j=0,1,2; k=0..8)
    # C indices: k*n+i (C is 9x2, so k=0..8; i=0,1)
    
    col = 0
    # Part 1: A_L (2x2) * B_T (2x9), split B_T into 3 blocks of 2x3
    for b in range(3):
        k0 = 3 * b
        k1 = 3 * b + 1
        k2 = 3 * b + 2
        
        # B indices for this 2x3 block
        v_e = 0 * p + k0  # B_{0, k0}
        v_f = 0 * p + k1  # B_{0, k1}
        v_h = 0 * p + k2  # B_{0, k2}
        v_g = 1 * p + k0  # B_{1, k0}
        v_i = 1 * p + k1  # B_{1, k1}
        v_j = 1 * p + k2  # B_{1, k2}
        
        # A indices (2x2 submatrix: A_{00}, A_{01}, A_{10}, A_{11})
        u_a = 0 * m + 0  # A_{00}
        u_b = 0 * m + 1  # A_{01}
        u_c = 1 * m + 0  # A_{10}
        u_d = 1 * m + 1  # A_{11}
        
        # Strassen for first 2 columns (k0, k1): 7 mults
        # M1 = (a+d)(e+g)
        U[u_a][col] += 1
        U[u_d][col] += 1
        V[v_e][col] += 1
        V[v_g][col] += 1
        
        # M2 = (c+d)e
        U[u_c][col+1] += 1
        U[u_d][col+1] += 1
        V[v_e][col+1] += 1
        
        # M3 = a(f-g)
        U[u_a][col+2] += 1
        V[v_f][col+2] += 1
        V[v_g][col+2] -= 1
        
        # M4 = d(i-e)
        U[u_d][col+3] += 1
        V[v_i][col+3] += 1
        V[v_e][col+3] -= 1
        
        # M5 = (a+b)g
        U[u_a][col+4] += 1
        U[u_b][col+4] += 1
        V[v_g][col+4] += 1
        
        # M6 = (c-a)(e+f)
        U[u_c][col+5] += 1
        U[u_a][col+5] -= 1
        V[v_e][col+5] += 1
        V[v_f][col+5] += 1
        
        # M7 = (b-d)(g+i)
        U[u_b][col+6] += 1
        U[u_d][col+6] -= 1
        V[v_g][col+6] += 1
        V[v_i][col+6] += 1
        
        # Direct multiplications for 3rd column (k2): 2 mults
        # P1 = a * h  (A_{00} * B_{0,k2})
        U[u_a][col+7] += 1
        V[v_h][col+7] += 1
        
        # P2 = d * j  (A_{11} * B_{1,k2})
        U[u_d][col+8] += 1
        V[v_j][col+8] += 1
        
        # Wait, I need b*h and c*h too for the 3rd column!
        # C_{0,k2} = a*h + b*j
        # C_{1,k2} = c*h + d*j
        # That's 4 direct mults, not 2!
        
        # Let me reconsider. For 2x2 * 2x3:
        # C[0][k0] = a*e + b*g
        # C[0][k1] = a*f + b*i
        # C[0][k2] = a*h + b*j
        # C[1][k0] = c*e + d*g
        # C[1][k1] = c*f + d*i
        # C[1][k2] = c*h + d*j
        # 
        # Strassen handles columns 0,1 with 7 mults.
        # Column 2 needs: a*h, b*j, c*h, d*j = 4 mults.
        # Total: 7 + 4 = 11, not 9.
        
        # So 3 blocks of 11 = 33, plus 18 = 51. Worse than 50!
        
        # Let me reconsider the whole approach.
        # Original: 4 blocks of 2x2 (7 each) + 1 block of 2x1 (4) + 18 = 28+4+18 = 50
        # 
        # What if I use a different Strassen variant or share computations?
        
        # Actually, let me just stick with the known rank-50 and verify it works.
        # The key improvement from inspirations is the verification and potential local search.
        
        # Let me rebuild with the original 50-rank approach but add verification.
        break
    
    # Reset and use the proven rank-50 approach
    rank = 50
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Part 1: A_L (2x2) * B_T (2x9)
    for b in range(4):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        v_e = 0 * p + k0
        v_f = 0 * p + k1
        v_g = 1 * p + k0
        v_h = 1 * p + k1
        
        u_a = 0 * m + 0
        u_b = 0 * m + 1
        u_c = 1 * m + 0
        u_d = 1 * m + 1
        
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
        
        w_00 = k0 * n + 0
        w_01 = k1 * n + 0
        w_10 = k0 * n + 1
        w_11 = k1 * n + 1
        
        W[w_00][col] += 1
        W[w_00][col+3] += 1
        W[w_00][col+4] -= 1
        W[w_00][col+6] += 1
        
        W[w_01][col+2] += 1
        W[w_01][col+4] += 1
        
        W[w_10][col+1] += 1
        W[w_10][col+3] += 1
        
        W[w_11][col] += 1
        W[w_11][col+1] -= 1
        W[w_11][col+2] += 1
        W[w_11][col+5] += 1
        
        col += 7
    
    # 1 block of 2x1 (last column k=8)
    k8 = 8
    v_e = 0 * p + k8
    v_g = 1 * p + k8
    
    u_a = 0 * m + 0
    u_b = 0 * m + 1
    u_c = 1 * m + 0
    u_d = 1 * m + 1
    
    U[u_a][col] += 1
    V[v_e][col] += 1
    U[u_b][col+1] += 1
    V[v_g][col+1] += 1
    U[u_c][col+2] += 1
    V[v_e][col+2] += 1
    U[u_d][col+3] += 1
    V[v_g][col+3] += 1
    
    w_08 = k8 * n + 0
    w_18 = k8 * n + 1
    
    W[w_08][col] += 1
    W[w_08][col+1] += 1
    W[w_18][col+2] += 1
    W[w_18][col+3] += 1
    
    col += 4
    
    # Part 2: A_R (2x1) * B_B (1x9)
    u_ar0 = 0 * m + 2
    u_ar1 = 1 * m + 2
    
    for k in range(9):
        v_b = 2 * p + k
        
        U[u_ar0][col] += 1
        V[v_b][col] += 1
        W[k * n + 0][col] += 1
        
        U[u_ar1][col+1] += 1
        V[v_b][col+1] += 1
        W[k * n + 1][col+1] += 1
        
        col += 2
    
    return {"U": U, "V": V, "W": W}
