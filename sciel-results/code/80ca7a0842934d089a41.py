import numpy as np
import time

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # We use a Strassen-based decomposition.
    # 2x4x6 is split into two 2x2x6 multiplications.
    # Each 2x2x6 is split into three 2x2x2 multiplications.
    # Each 2x2x2 uses Strassen (rank 7).
    # Total rank = 2 * 3 * 7 = 42.
    
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # Indices for 2x2x2 block:
    # A is 2x2, B is 2x2.
    # U indices: 0,1,2,3 (i*m+j)
    # V indices: 0,1,2,3 (j*p+k)
    # W indices: 0,1,2,3 (k*n+i)
    
    # Strassen terms (u, v, w) where u,v,w are vectors of length 4
    # u is for A, v for B, w for C.
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We need to map these to the global indices.
    
    def add_strassen_block(u_offset, v_offset, w_offset, col_start):
        # u_offset: index in U for the 2x2 A block
        # v_offset: index in V for the 2x2 B block
        # w_offset: index in W for the 2x2 C block
        
        # Term 1: M1
        # A: (0,0), (1,1) -> u[0], u[3]
        # B: (0,0), (1,1) -> v[0], v[3]
        # C: (0,0), (1,1) -> w[0], w[3]
        # Coeffs:
        # u: [1, 0, 0, 1]
        # v: [1, 0, 0, 1]
        # w: [1, 0, 0, 1]
        U[u_offset][col_start] = 1
        U[u_offset+3][col_start] = 1
        V[v_offset][col_start] = 1
        V[v_offset+3][col_start] = 1
        W[w_offset][col_start] = 1
        W[w_offset+3][col_start] = 1
        
        # Term 2: M2
        # A: (1,0), (1,1) -> u[2], u[3]
        # B: (0,0) -> v[0]
        # C: (1,0) -> w[2]
        # u: [0, 0, 1, 1]
        # v: [1, 0, 0, 0]
        # w: [0, 0, 1, 0]
        U[u_offset+2][col_start+1] = 1
        U[u_offset+3][col_start+1] = 1
        V[v_offset][col_start+1] = 1
        W[w_offset+2][col_start+1] = 1
        
        # Term 3: M3
        # A: (0,0) -> u[0]
        # B: (0,1), (1,1) -> v[1], v[3]
        # C: (0,1) -> w[1]
        # u: [1, 0, 0, 0]
        # v: [0, 1, 0, -1]
        # w: [0, 1, 0, 0]
        U[u_offset][col_start+2] = 1
        V[v_offset+1][col_start+2] = 1
        V[v_offset+3][col_start+2] = -1
        W[w_offset+1][col_start+2] = 1
        
        # Term 4: M4
        # A: (1,1) -> u[3]
        # B: (1,0), (0,0) -> v[2], v[0]
        # C: (1,0) -> w[2]
        # u: [0, 0, 0, 1]
        # v: [1, 0, -1, 0]
        # w: [0, 0, 1, 0]
        U[u_offset+3][col_start+3] = 1
        V[v_offset][col_start+3] = 1
        V[v_offset+2][col_start+3] = -1
        W[w_offset+2][col_start+3] = 1
        
        # Term 5: M5
        # A: (0,0), (0,1) -> u[0], u[1]
        # B: (1,1) -> v[3]
        # C: (0,1) -> w[1]
        # u: [1, 1, 0, 0]
        # v: [0, 0, 0, 1]
        # w: [0, 1, 0, 0]
        U[u_offset][col_start+4] = 1
        U[u_offset+1][col_start+4] = 1
        V[v_offset+3][col_start+4] = 1
        W[w_offset+1][col_start+4] = 1
        
        # Term 6: M6
        # A: (1,0), (0,0) -> u[2], u[0]
        # B: (0,0), (0,1) -> v[0], v[1]
        # C: (1,1) -> w[3]
        # u: [1, 0, -1, 0]
        # v: [1, 1, 0, 0]
        # w: [0, 0, 0, 1]
        U[u_offset][col_start+5] = 1
        U[u_offset+2][col_start+5] = -1
        V[v_offset][col_start+5] = 1
        V[v_offset+1][col_start+5] = 1
        W[w_offset+3][col_start+5] = 1
        
        # Term 7: M7
        # A: (0,1), (1,1) -> u[1], u[3]
        # B: (1,0), (1,1) -> v[2], v[3]
        # C: (1,1) -> w[3]
        # u: [0, 1, 0, -1]
        # v: [0, 0, 1, 1]
        # w: [0, 0, 0, 1]
        U[u_offset+1][col_start+6] = 1
        U[u_offset+3][col_start+6] = -1
        V[v_offset+2][col_start+6] = 1
        V[v_offset+3][col_start+6] = 1
        W[w_offset+3][col_start+6] = 1

    # Block 1: A1 (rows 0-1, cols 0-1) * B1 (rows 0-1, cols 0-5)
    # A1 is 2x2. B1 is 2x6.
    # Split B1 into B1a (cols 0-1), B1b (cols 2-3), B1c (cols 4-5).
    
    # A1 U offset: 0 (indices 0,1,2,3)
    # B1a V offset: 0 (indices 0,1,2,3)
    # C1a W offset: 0 (indices 0,1,2,3) -> C rows 0-1, cols 0-1
    add_strassen_block(0, 0, 0, 0)
    
    # B1b V offset: 4 (indices 4,5,6,7)
    # C1b W offset: 4 (indices 4,5,6,7) -> C rows 0-1, cols 2-3
    # Note: W index is k*n + i.
    # For C1a (cols 0-1), k=0,1. W indices 0..3.
    # For C1b (cols 2-3), k=2,3. W indices 4..7.
    # For C1c (cols 4-5), k=4,5. W indices 8..11.
    add_strassen_block(0, 4, 4, 7)
    
    # B1c V offset: 8 (indices 8,9,10,11)
    # C1c W offset: 8
    add_strassen_block(0, 8, 8, 14)

    # Block 2: A2 (rows 0-1, cols 2-3) * B2 (rows 2-3, cols 0-5)
    # A2 U offset: 4 (indices 4,5,6,7)
    # B2a V offset: 12 (indices 12,13,14,15)
    # C2a W offset: 0 (Same C rows/cols as C1a)
    add_strassen_block(4, 12, 0, 21)
    
    # B2b V offset: 16
    # C2b W offset: 4
    add_strassen_block(4, 16, 4, 28)
    
    # B2c V offset: 20
    # C2c W offset: 8
    add_strassen_block(4, 20, 8, 35)

    return {"U": U, "V": V, "W": W}
