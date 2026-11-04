import time
import numpy as np

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 88 decomposition
    # Top 2x8 part: 4 blocks of 2x2 output.
    # Each block is sum of two 2x2 multiplications.
    # Each 2x2 mult has rank 7.
    # Total 4 * 2 * 7 = 56.
    
    # Bottom 1x8 part: 4 blocks of 1x2 output.
    # Each block is 1x4 by 4x2 multiplication.
    # Split 1x4 by 4x2 into two 1x2 by 2x2 multiplications.
    # Each 1x2 by 2x2 has rank 4 (assumed).
    # Total 4 * 2 * 4 = 32.
    
    # Total rank 88.
    
    rank = 88
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen for 2x2 by 2x2
    # M1 = A11 + A22, M2 = A21 + A22, M3 = A11, M4 = A22
    # M5 = A11 + A12, M6 = A21 - A11, M7 = A12 + A22
    # P1 = M1 * (B11 + B22)
    # P2 = M2 * B22
    # P3 = M3 * B11
    # P4 = M4 * B22
    # P5 = M5 * (B12 - B22)
    # P6 = M6 * (B11 + B12)
    # P7 = M7 * (B21 - B11)
    # C11 = P1 + P4 - P2 - P7
    # C12 = P3 + P5
    # C21 = P6 + P4
    # C22 = P1 - P3 + P2 - P6
    
    def add_strassen_2x2(u_offset, v_offset, w_offset, i_base, j_base, k_base):
        nonlocal col
        # u_offset: index in U for A block
        # v_offset: index in V for B block
        # w_offset: index in W for C block
        # i_base, j_base, k_base: base indices in original matrices
        
        # A is 2x2, B is 2x2, C is 2x2
        # A indices: (i_base, j_base), (i_base, j_base+1), (i_base+1, j_base), (i_base+1, j_base+1)
        # U row index: i * m + j
        # V row index: j * p + k
        # W row index: k * n + i
        
        a00 = u_offset + 0 * m + 0
        a01 = u_offset + 0 * m + 1
        a10 = u_offset + 1 * m + 0
        a11 = u_offset + 1 * m + 1
        
        b00 = v_offset + 0 * p + 0
        b01 = v_offset + 0 * p + 1
        b10 = v_offset + 1 * p + 0
        b11 = v_offset + 1 * p + 1
        
        c00 = w_offset + 0 * n + 0
        c01 = w_offset + 0 * n + 1
        c10 = w_offset + 1 * n + 0
        c11 = w_offset + 1 * n + 1
        
        # M1 = A11 + A22 (00 + 11)
        U[a00][col] = 1
        U[a11][col] = 1
        V[b00][col] = 1
        V[b11][col] = 1
        W[c00][col] = 1
        W[c11][col] = 1
        W[c10][col] = -1
        W[c01][col] = -1
        col += 1
        
        # M2 = A21 + A22 (10 + 11)
        U[a10][col] = 1
        U[a11][col] = 1
        V[b11][col] = 1
        W[c00][col] = -1
        W[c11][col] = 1
        col += 1
        
        # M3 = A11 (00)
        U[a00][col] = 1
        V[b00][col] = 1
        W[c01][col] = 1
        W[c10][col] = 1
        col += 1
        
        # M4 = A22 (11)
        U[a11][col] = 1
        V[b11][col] = 1
        W[c00][col] = 1
        W[c10][col] = 1
        W[c11][col] = -1
        col += 1
        
        # M5 = A11 + A12 (00 + 01)
        U[a00][col] = 1
        U[a01][col] = 1
        V[b01][col] = 1
        V[b11][col] = -1
        W[c01][col] = 1
        col += 1
        
        # M6 = A21 - A11 (10 - 00)
        U[a10][col] = 1
        U[a00][col] = -1
        V[b00][col] = 1
        V[b01][col] = 1
        W[c10][col] = 1
        W[c11][col] = -1
        col += 1
        
        # M7 = A12 + A22 (01 + 11)
        U[a01][col] = 1
        U[a11][col] = 1
        V[b10][col] = 1
        V[b00][col] = -1
        W[c11][col] = 1
        col += 1
        
        return col

    # Top part: I in {0,1}, K in {0,1,2,3} (blocks of size 2)
    # J in {0,1} (blocks of size 2)
    
    # Block (0,0): C[0:2, 0:2] = A[0:2, 0:2] B[0:2, 0:2] + A[0:2, 2:4] B[2:4, 0:2]
    # A1: i=0:2, j=0:2. A2: i=0:2, j=2:4.
    # B1: j=0:2, k=0:2. B2: j=2:4, k=0:2.
    
    # Strassen 1 for A1 B1
    col = add_strassen_2x2(0, 0, 0, 0, 0, 0)
    
    # Strassen 2 for A2 B2
    # A2 offset: i=0:2, j=2:4. U index: i*m + j. j starts at 2.
    # V offset: j=2:4, k=0:2. V index: j*p + k. j starts at 2.
    # W offset: k=0:2, i=0:2. W index: k*n + i. k starts at 0.
    col = add_strassen_2x2(0 + 2, 2 * 8, 0, 0, 2, 0)
    
    # Block (0,1): C[0:2, 2:4] = A[0:2, 0:2] B[0:2, 2:4] + A[0:2, 2:4] B[2:4, 2:4]
    col = add_strassen_2x2(0, 0 + 2, 0 + 2 * 3, 0, 0, 2)
    col = add_strassen_2x2(0 + 2, 2 * 8 + 2, 0 + 2 * 3, 0, 2, 2)
    
    # Block (0,2): C[0:2, 4:6]
    col = add_strassen_2x2(0, 0 + 4, 0 + 4 * 3, 0, 0, 4)
    col = add_strassen_2x2(0 + 2, 2 * 8 + 4, 0 + 4 * 3, 0, 2, 4)
    
    # Block (0,3): C[0:2, 6:8]
    col = add_strassen_2x2(0, 0 + 6, 0 + 6 * 3, 0, 0, 6)
    col = add_strassen_2x2(0 + 2, 2 * 8 + 6, 0 + 6 * 3, 0, 2, 6)
    
    # Bottom part: I=2, K in {0,1,2,3}
    # C[2, k:k+2] = A[2, 0:2] B[0:2, k:k+2] + A[2, 2:4] B[2:4, k:k+2]
    # A1: i=2, j=0:2. 1x2.
    # B1: j=0:2, k:k+2. 2x2.
    # Product: 1x2.
    # Rank 4 decomposition for 1x2 by 2x2.
    # C0 = A0 B00 + A1 B10
    # C1 = A0 B01 + A1 B11
    # T1 = A0 (B00 + B10) -> C0? No.
    # Standard rank 4:
    # P1 = A0 B00 -> C0
    # P2 = A0 B01 -> C1
    # P3 = A1 B10 -> C0
    # P4 = A1 B11 -> C1
    
    def add_1x2_2x2(u_offset, v_offset, w_offset):
        nonlocal col
        # u_offset: index for A[2, j]
        # v_offset: index for B[j, k]
        # w_offset: index for C[2, k]
        
        a0 = u_offset + 0
        a1 = u_offset + 1
        
        b00 = v_offset + 0 * p + 0
        b01 = v_offset + 0 * p + 1
        b10 = v_offset + 1 * p + 0
        b11 = v_offset + 1 * p + 1
        
        c0 = w_offset + 0 * n + 2
        c1 = w_offset + 1 * n + 2
        
        # P1 = A0 B00
        U[a0][col] = 1
        V[b00][col] = 1
        W[c0][col] = 1
        col += 1
        
        # P2 = A0 B01
        U[a0][col] = 1
        V[b01][col] = 1
        W[c1][col] = 1
        col += 1
        
        # P3 = A1 B10
        U[a1][col] = 1
        V[b10][col] = 1
        W[c0][col] = 1
        col += 1
        
        # P4 = A1 B11
        U[a1][col] = 1
        V[b11][col] = 1
        W[c1][col] = 1
        col += 1
        
        return col

    # Block (2,0): K=0:2
    # A1: j=0:2. U index: 2*4 + 0 = 8.
    # B1: j=0:2, k=0:2. V index: 0*8 + 0 = 0.
    # W: k=0:2, i=2. W index: 0*3 + 2 = 2.
    col = add_1x2_2x2(8, 0, 0)
    # A2: j=2:4. U index: 2*4 + 2 = 10.
    # B2: j=2:4, k=0:2. V index: 2*8 + 0 = 16.
    col = add_1x2_2x2(10, 16, 0)
    
    # Block (2,1): K=2:4
    col = add_1x2_2x2(8, 0 + 2, 2)
    col = add_1x2_2x2(10, 16 + 2, 2)
    
    # Block (2,2): K=4:6
    col = add_1x2_2x2(8, 0 + 4, 4)
    col = add_1x2_2x2(10, 16 + 4, 4)
    
    # Block (2,3): K=6:8
    col = add_1x2_2x2(8, 0 + 6, 6)
    col = add_1x2_2x2(10, 16 + 6, 6)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
