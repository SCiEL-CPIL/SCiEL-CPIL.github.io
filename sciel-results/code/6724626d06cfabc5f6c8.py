"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=3, m=4, p=8
    
    U_rows = n * m  # 12
    V_rows = m * p  # 32
    W_rows = p * n  # 24
    
    rank = 88
    
    U = [[0] * rank for _ in range(U_rows)]
    V = [[0] * rank for _ in range(V_rows)]
    W = [[0] * rank for _ in range(W_rows)]
    
    col = 0
    
    # Part 1: Schoolbook for i=0 (1x4 by 4x8)
    # A[0, j] * B[j, k] -> C[0, k]
    for j in range(4):
        for k in range(8):
            # U index: i*m + j = 0*4 + j = j
            U[j][col] = 1
            # V index: j*p + k = j*8 + k
            V[j * 8 + k][col] = 1
            # W index: k*n + i = k*3 + 0 = 3k
            W[k * 3][col] = 1
            col += 1
            
    # Part 2: Strassen for i=1,2 (2x4 by 4x8)
    # Split B into 4 blocks of 4x2
    for q in range(4):
        k0 = 2 * q
        k1 = 2 * q + 1
        
        # Left Part: A_L (rows 1,2; cols 0,1) * B_T (rows 0,1; cols k0,k1)
        # A_L indices in U:
        # i=1, j=0 -> 1*4+0=4
        # i=1, j=1 -> 1*4+1=5
        # i=2, j=0 -> 2*4+0=8
        # i=2, j=1 -> 2*4+1=9
        a00_u = 4
        a01_u = 5
        a10_u = 8
        a11_u = 9
        
        # B_T indices in V:
        # j=0, k=k0 -> 0*8+k0 = k0
        # j=0, k=k1 -> 0*8+k1 = k1
        # j=1, k=k0 -> 1*8+k0 = 8+k0
        # j=1, k=k1 -> 1*8+k1 = 8+k1
        b00_v = k0
        b01_v = k1
        b10_v = 8 + k0
        b11_v = 8 + k1
        
        # Result C indices in W:
        # k=k0, i=1 -> k0*3+1
        # k=k0, i=2 -> k0*3+2
        # k=k1, i=1 -> k1*3+1
        # k=k1, i=2 -> k1*3+2
        c00_w = k0 * 3 + 1
        c01_w = k1 * 3 + 1
        c10_w = k0 * 3 + 2
        c11_w = k1 * 3 + 2
        
        # Add 7 Strassen terms for A_L * B_T
        # M1 = (A00+A11)(B00+B11)
        U[a00_u][col] += 1; U[a11_u][col] += 1
        V[b00_v][col] += 1; V[b11_v][col] += 1
        W[c00_w][col] += 1; W[c11_w][col] += 1
        col += 1
        
        # M2 = (A10+A11)B00
        U[a10_u][col] += 1; U[a11_u][col] += 1
        V[b00_v][col] += 1
        W[c10_w][col] += 1; W[c11_w][col] -= 1
        col += 1
        
        # M3 = A00(B10-B11)
        U[a00_u][col] += 1
        V[b10_v][col] += 1; V[b11_v][col] -= 1
        W[c10_w][col] += 1; W[c11_w][col] -= 1
        col += 1
        
        # M4 = A11(B01-B00)
        U[a11_u][col] += 1
        V[b01_v][col] += 1; V[b00_v][col] -= 1
        W[c00_w][col] -= 1; W[c01_w][col] += 1
        col += 1
        
        # M5 = (A00+A01)B11
        U[a00_u][col] += 1; U[a01_u][col] += 1
        V[b11_v][col] += 1
        W[c00_w][col] += 1; W[c01_w][col] -= 1
        col += 1
        
        # M6 = (A10-A00)(B00+B01)
        U[a10_u][col] += 1; U[a00_u][col] -= 1
        V[b00_v][col] += 1; V[b01_v][col] += 1
        W[c11_w][col] += 1
        col += 1
        
        # M7 = (A01-A11)(B10+B11)
        U[a01_u][col] += 1; U[a11_u][col] -= 1
        V[b10_v][col] += 1; V[b11_v][col] += 1
        W[c00_w][col] -= 1; W[c01_w][col] += 1
        col += 1
        
        # Right Part: A_R (rows 1,2; cols 2,3) * B_B (rows 2,3; cols k0,k1)
        # A_R indices in U:
        # i=1, j=2 -> 1*4+2=6
        # i=1, j=3 -> 1*4+3=7
        # i=2, j=2 -> 2*4+2=10
        # i=2, j=3 -> 2*4+3=11
        a00_u = 6
        a01_u = 7
        a10_u = 10
        a11_u = 11
        
        # B_B indices in V:
        # j=2, k=k0 -> 2*8+k0 = 16+k0
        # j=2, k=k1 -> 2*8+k1 = 16+k1
        # j=3, k=k0 -> 3*8+k0 = 24+k0
        # j=3, k=k1 -> 3*8+k1 = 24+k1
        b00_v = 16 + k0
        b01_v = 16 + k1
        b10_v = 24 + k0
        b11_v = 24 + k1
        
        # W indices are the same as above (c00_w, c01_w, c10_w, c11_w)
        
        # Add 7 Strassen terms for A_R * B_B
        # M1
        U[a00_u][col] += 1; U[a11_u][col] += 1
        V[b00_v][col] += 1; V[b11_v][col] += 1
        W[c00_w][col] += 1; W[c11_w][col] += 1
        col += 1
        
        # M2
        U[a10_u][col] += 1; U[a11_u][col] += 1
        V[b00_v][col] += 1
        W[c10_w][col] += 1; W[c11_w][col] -= 1
        col += 1
        
        # M3
        U[a00_u][col] += 1
        V[b10_v][col] += 1; V[b11_v][col] -= 1
        W[c10_w][col] += 1; W[c11_w][col] -= 1
        col += 1
        
        # M4
        U[a11_u][col] += 1
        V[b01_v][col] += 1; V[b00_v][col] -= 1
        W[c00_w][col] -= 1; W[c01_w][col] += 1
        col += 1
        
        # M5
        U[a00_u][col] += 1; U[a01_u][col] += 1
        V[b11_v][col] += 1
        W[c00_w][col] += 1; W[c01_w][col] -= 1
        col += 1
        
        # M6
        U[a10_u][col] += 1; U[a00_u][col] -= 1
        V[b00_v][col] += 1; V[b01_v][col] += 1
        W[c11_w][col] += 1
        col += 1
        
        # M7
        U[a01_u][col] += 1; U[a11_u][col] -= 1
        V[b10_v][col] += 1; V[b11_v][col] += 1
        W[c00_w][col] -= 1; W[c01_w][col] += 1
        col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
