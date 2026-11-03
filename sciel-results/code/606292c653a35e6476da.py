"""Find rank-minimal exact tensor decomposition for 2x3 * 3x4 matrix multiplication."""
import numpy as np
import time

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    # Rank 22 decomposition using Strassen-like blocking
    rank = 22
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Helper to add a rank-1 term
    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val, col):
        U[u_idx, col] += u_val
        V[v_idx, col] += v_val
        W[w_idx, col] += w_val

    col = 0
    
    # Part 1: A[:, 0] * B[0, :]
    # A is 2x3, B is 3x4
    # A[:, 0] is 2x1. B[0, :] is 1x4.
    # 8 terms
    for i in range(2):
        for k in range(4):
            # U index: i*m + 0
            # V index: 0*p + k
            # W index: k*n + i
            add_term(i * m + 0, 1.0, 0 * p + k, 1.0, k * n + i, 1.0, col)
            col += 1
            
    # Part 2a: A[:, 1:3] * B[1:3, 0:2]
    # A_block is 2x2 (rows 0,1; cols 1,2)
    # B_block is 2x2 (rows 1,2; cols 0,1)
    # Strassen 7 terms
    
    # Local A indices: (0,0)->(0,1), (0,1)->(0,2), (1,0)->(1,1), (1,1)->(1,2)
    # Global A idx: 0*m+1=1, 0*m+2=2, 1*m+1=4, 1*m+2=5
    a00, a01, a10, a11 = 1, 2, 4, 5
    
    # Local B indices: (0,0)->(1,0), (0,1)->(1,1), (1,0)->(2,0), (1,1)->(2,1)
    # Global B idx: 1*p+0=4, 1*p+1=5, 2*p+0=8, 2*p+1=9
    b00, b01, b10, b11 = 4, 5, 8, 9
    
    # Local C indices: (0,0)->(0,0), (0,1)->(0,1), (1,0)->(1,0), (1,1)->(1,1)
    # Global W idx: 0*n+0=0, 0*n+1=1, 1*n+0=2, 1*n+1=3
    c00, c01, c10, c11 = 0, 1, 2, 3
    
    # P1 = (A00 + A11)(B00 + B11)
    add_term(a00, 1.0, b00, 1.0, c00, 1.0, col)
    add_term(a11, 1.0, b00, 1.0, c00, 1.0, col) # Wait, add_term adds to same col.
    # My add_term signature is (u_idx, u_val, v_idx, v_val, w_idx, w_val, col)
    # It adds u_val to U[u_idx, col].
    # So for P1:
    U[a00, col] = 1.0
    U[a11, col] = 1.0
    V[b00, col] = 1.0
    V[b11, col] = 1.0
    W[c00, col] = 1.0
    W[c11, col] = 1.0
    col += 1
    
    # P2 = (A10 + A11)B00
    U[a10, col] = 1.0
    U[a11, col] = 1.0
    V[b00, col] = 1.0
    W[c10, col] = 1.0
    col += 1
    
    # P3 = A00(B01 - B11)
    U[a00, col] = 1.0
    V[b01, col] = 1.0
    V[b11, col] = -1.0
    W[c01, col] = 1.0
    W[c11, col] = -1.0
    col += 1
    
    # P4 = A11(B10 - B00)
    U[a11, col] = 1.0
    V[b10, col] = 1.0
    V[b00, col] = -1.0
    W[c00, col] = -1.0
    W[c10, col] = 1.0
    col += 1
    
    # P5 = (A00 + A01)B11
    U[a00, col] = 1.0
    U[a01, col] = 1.0
    V[b11, col] = 1.0
    W[c01, col] = 1.0
    col += 1
    
    # P6 = (A10 - A00)(B00 + B01)
    U[a10, col] = 1.0
    U[a00, col] = -1.0
    V[b00, col] = 1.0
    V[b01, col] = 1.0
    W[c00, col] = 1.0
    W[c10, col] = 1.0
    col += 1
    
    # P7 = (A01 - A11)(B10 + B11)
    U[a01, col] = 1.0
    U[a11, col] = -1.0
    V[b10, col] = 1.0
    V[b11, col] = 1.0
    W[c11, col] = 1.0
    col += 1
    
    # Part 2b: A[:, 1:3] * B[1:3, 2:4]
    # A_block same as 2a
    # B_block rows 1,2; cols 2,3
    # Global B idx: 1*p+2=6, 1*p+3=7, 2*p+2=10, 2*p+3=11
    b00, b01, b10, b11 = 6, 7, 10, 11
    
    # Local C indices: (0,0)->(0,2), (0,1)->(0,3), (1,0)->(1,2), (1,1)->(1,3)
    # Global W idx: 2*n+0=4, 2*n+1=5, 3*n+0=6, 3*n+1=7
    c00, c01, c10, c11 = 4, 5, 6, 7
    
    # P1
    U[a00, col] = 1.0
    U[a11, col] = 1.0
    V[b00, col] = 1.0
    V[b11, col] = 1.0
    W[c00, col] = 1.0
    W[c11, col] = 1.0
    col += 1
    
    # P2
    U[a10, col] = 1.0
    U[a11, col] = 1.0
    V[b00, col] = 1.0
    W[c10, col] = 1.0
    col += 1
    
    # P3
    U[a00, col] = 1.0
    V[b01, col] = 1.0
    V[b11, col] = -1.0
    W[c01, col] = 1.0
    W[c11, col] = -1.0
    col += 1
    
    # P4
    U[a11, col] = 1.0
    V[b10, col] = 1.0
    V[b00, col] = -1.0
    W[c00, col] = -1.0
    W[c10, col] = 1.0
    col += 1
    
    # P5
    U[a00, col] = 1.0
    U[a01, col] = 1.0
    V[b11, col] = 1.0
    W[c01, col] = 1.0
    col += 1
    
    # P6
    U[a10, col] = 1.0
    U[a00, col] = -1.0
    V[b00, col] = 1.0
    V[b01, col] = 1.0
    W[c00, col] = 1.0
    W[c10, col] = 1.0
    col += 1
    
    # P7
    U[a01, col] = 1.0
    U[a11, col] = -1.0
    V[b10, col] = 1.0
    V[b11, col] = 1.0
    W[c11, col] = 1.0
    col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
