import time
import numpy as np

DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = 3, 4, 8
    # Rank calculation
    # 2 halves of m
    # Each half:
    #   2x2x8 part (n=2, m=2, p=8) -> 4 blocks of 2x2x2 (rank 7) -> 28
    #   1x2x8 part (n=1, m=2, p=8) -> schoolbook (rank 16) -> 16
    # Total per half: 44
    # Total rank: 88
    
    rank = 88
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a scalar to U, V, W at column c
    def add_factor(u_idx, v_idx, w_idx, val):
        U[u_idx][col] += val
        V[v_idx][col] += val
        W[w_idx][col] += val

    # Strassen coefficients for 2x2x2
    # M1: (A11+A22)(B11+B22)
    # M2: (A21+A22)B11
    # M3: A11(B21-B22)
    # M4: A22(B12-B11)
    # M5: (A11+A12)B22
    # M6: (A21-A11)(B11+B12)
    # M7: (A12-A22)(B21+B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We will iterate through the 2 halves of m (j=0,1 and j=2,3)
    for half in range(2):
        j_base = half * 2
        
        # Part 1: n=2 (i=0,1), m=2 (j=j_base, j_base+1), p=8
        # Split p=8 into 4 blocks of 2 (k=0,1; 2,3; 4,5; 6,7)
        for k_block in range(4):
            k_base = k_block * 2
            
            # Indices for A (U rows)
            # i=0,1; j=j_base, j_base+1
            # A00 = U[0*m + j_base] = U[j_base]
            # A01 = U[0*m + j_base+1] = U[j_base+1]
            # A10 = U[1*m + j_base] = U[m + j_base]
            # A11 = U[1*m + j_base+1] = U[m + j_base+1]
            
            # Indices for B (V rows)
            # j=j_base, j_base+1; k=k_base, k_base+1
            # B00 = V[j_base*p + k_base]
            # B01 = V[j_base*p + k_base+1]
            # B10 = V[(j_base+1)*p + k_base]
            # B11 = V[(j_base+1)*p + k_base+1]
            
            # Indices for C (W rows)
            # k=k_base, k_base+1; i=0,1
            # C00 = W[k_base*n + 0]
            # C01 = W[(k_base+1)*n + 0]
            # C10 = W[k_base*n + 1]
            # C11 = W[(k_base+1)*n + 1]
            
            u_00 = 0 * m + j_base
            u_01 = 0 * m + (j_base + 1)
            u_10 = 1 * m + j_base
            u_11 = 1 * m + (j_base + 1)
            
            v_00 = j_base * p + k_base
            v_01 = j_base * p + (k_base + 1)
            v_10 = (j_base + 1) * p + k_base
            v_11 = (j_base + 1) * p + (k_base + 1)
            
            w_00 = k_base * n + 0
            w_01 = (k_base + 1) * n + 0
            w_10 = k_base * n + 1
            w_11 = (k_base + 1) * n + 1
            
            # M1: (A00+A11)(B00+B11)
            U[u_00][col] += 1
            U[u_11][col] += 1
            V[v_00][col] += 1
            V[v_11][col] += 1
            W[w_00][col] += 1
            W[w_11][col] += 1
            col += 1
            
            # M2: (A10+A11)B00
            U[u_10][col] += 1
            U[u_11][col] += 1
            V[v_00][col] += 1
            W[w_10][col] += 1
            col += 1
            
            # M3: A00(B10-B11)
            U[u_00][col] += 1
            V[v_10][col] += 1
            V[v_11][col] -= 1
            W[w_01][col] += 1
            col += 1
            
            # M4: A11(B01-B00)
            U[u_11][col] += 1
            V[v_01][col] += 1
            V[v_00][col] -= 1
            W[w_00][col] += 1
            W[w_11][col] += 1
            col += 1
            
            # M5: (A00+A01)B11
            U[u_00][col] += 1
            U[u_01][col] += 1
            V[v_11][col] += 1
            W[w_00][col] -= 1
            W[w_01][col] += 1
            col += 1
            
            # M6: (A10-A00)(B00+B01)
            U[u_10][col] += 1
            U[u_00][col] -= 1
            V[v_00][col] += 1
            V[v_01][col] += 1
            W[w_11][col] += 1
            col += 1
            
            # M7: (A01-A11)(B10+B11)
            U[u_01][col] += 1
            U[u_11][col] -= 1
            V[v_10][col] += 1
            V[v_11][col] += 1
            W[w_00][col] += 1
            W[w_11][col] += 1
            col += 1
            
        # Part 2: n=1 (i=2), m=2 (j=j_base, j_base+1), p=8
        # Schoolbook for 1x2x8
        # C2k = A2j0 B0k + A2j1 B1k
        for k in range(8):
            # Term 1: A[2, j_base] * B[j_base, k]
            u_idx = 2 * m + j_base
            v_idx = j_base * p + k
            w_idx = k * n + 2
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1
            
            # Term 2: A[2, j_base+1] * B[j_base+1, k]
            u_idx = 2 * m + (j_base + 1)
            v_idx = (j_base + 1) * p + k
            w_idx = k * n + 2
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1

    return {"U": U, "V": V, "W": W}
