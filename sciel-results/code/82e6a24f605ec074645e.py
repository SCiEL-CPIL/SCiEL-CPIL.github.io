import time
import numpy as np

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 18 decomposition
    # A (2x3) = [A1 (2x2) | A2 (2x1)]
    # B (3x4) = [B1 (2x4); B2 (1x4)]
    # AB = A1 B1 + A2 B2
    # B1 (2x4) = [B11 (2x2) | B12 (2x2)]
    # A1 B1 = A1 B11 + A1 B12
    # Rank(A1 B11) = 7 (Strassen)
    # Rank(A1 B12) = 7 (Strassen)
    # Rank(A2 B2) = 4 (Outer product 2x1 and 1x4)
    # Total Rank = 7 + 7 + 4 = 18
    
    rank = 18
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Helper to add Strassen decomposition for 2x2 * 2x2
    # Indices in U, V, W for the sub-blocks
    # A1 is rows 0-1, cols 0-1 of A (indices in U: 0,1,2,3)
    # B11 is rows 0-1, cols 0-1 of B (indices in V: 0,1,4,5)
    # Output is 2x2 (indices in W: 0,1,2,3)
    
    def add_strassen(u_base, v_base, w_base, offset):
        # u_base: list of 4 indices for A11, A12, A21, A22
        # v_base: list of 4 indices for B11, B12, B21, B22
        # w_base: list of 4 indices for C11, C12, C21, C22
        # Strassen 7 multiplications
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # M1
        U[u_base[0], col+offset] = 1
        U[u_base[3], col+offset] = 1
        V[v_base[0], col+offset] = 1
        V[v_base[3], col+offset] = 1
        W[w_base[0], col+offset] = 1
        W[w_base[1], col+offset] = 1
        W[w_base[2], col+offset] = 1
        W[w_base[3], col+offset] = 1
        # C11 += M1, C12 -= M1, C21 -= M1, C22 += M1
        # Wait, standard Strassen:
        # C11 = M1 - M4 + M5 - M7
        # C12 = M3 + M7
        # C21 = M2 + M4
        # C22 = M1 - M3 + M2 - M6
        
        # Let's just use the standard coefficients
        # M1
        U[u_base[0], col+offset] = 1
        U[u_base[3], col+offset] = 1
        V[v_base[0], col+offset] = 1
        V[v_base[3], col+offset] = 1
        W[w_base[0], col+offset] = 1
        W[w_base[1], col+offset] = -1
        W[w_base[2], col+offset] = -1
        W[w_base[3], col+offset] = 1
        
        # M2
        U[u_base[2], col+offset+1] = 1
        U[u_base[3], col+offset+1] = 1
        V[v_base[0], col+offset+1] = 1
        W[w_base[2], col+offset+1] = 1
        W[w_base[3], col+offset+1] = 1
        
        # M3
        U[u_base[0], col+offset+2] = 1
        V[v_base[1], col+offset+2] = 1
        V[v_base[3], col+offset+2] = -1
        W[w_base[1], col+offset+2] = 1
        W[w_base[3], col+offset+2] = -1
        
        # M4
        U[u_base[3], col+offset+3] = 1
        V[v_base[2], col+offset+3] = 1
        V[v_base[0], col+offset+3] = -1
        W[w_base[0], col+offset+3] = -1
        W[w_base[2], col+offset+3] = 1
        
        # M5
        U[u_base[0], col+offset+4] = 1
        U[u_base[1], col+offset+4] = 1
        V[v_base[3], col+offset+4] = 1
        W[w_base[0], col+offset+4] = 1
        W[w_base[2], col+offset+4] = -1
        
        # M6
        U[u_base[2], col+offset+5] = 1
        U[u_base[0], col+offset+5] = -1
        V[v_base[0], col+offset+5] = 1
        V[v_base[1], col+offset+5] = 1
        W[w_base[3], col+offset+5] = -1
        
        # M7
        U[u_base[1], col+offset+6] = 1
        U[u_base[3], col+offset+6] = -1
        V[v_base[2], col+offset+6] = 1
        V[v_base[3], col+offset+6] = 1
        W[w_base[1], col+offset+6] = 1
        W[w_base[3], col+offset+6] = 1
        
        return col + offset + 7

    # 1. A1 B11 (2x2 * 2x2)
    # A1 indices in U: 
    # A is 2x3. A1 is top-left 2x2.
    # U indices: i*m + j.
    # i=0, j=0 -> 0; i=0, j=1 -> 1; i=1, j=0 -> 3; i=1, j=1 -> 4.
    # Wait, A is 2x3.
    # Row 0: j=0,1,2 -> indices 0,1,2
    # Row 1: j=0,1,2 -> indices 3,4,5
    # A1 (2x2) uses j=0,1.
    # A11 (0,0) -> U[0]
    # A12 (0,1) -> U[1]
    # A21 (1,0) -> U[3]
    # A22 (1,1) -> U[4]
    u_A1 = [0, 1, 3, 4]
    
    # B11 (2x2) is top-left of B1 (2x4).
    # B is 3x4.
    # B1 is top 2 rows.
    # B11 is cols 0,1.
    # V indices: j*p + k.
    # j=0, k=0 -> 0; j=0, k=1 -> 1; j=1, k=0 -> 4; j=1, k=1 -> 5.
    v_B11 = [0, 1, 4, 5]
    
    # Output 2x2 is top-left of result.
    # W indices: k*n + i.
    # k=0, i=0 -> 0; k=0, i=1 -> 1; k=1, i=0 -> 2; k=1, i=1 -> 3.
    w_C11 = [0, 1, 2, 3]
    
    col = add_strassen(u_A1, v_B11, w_C11, 0)
    
    # 2. A1 B12 (2x2 * 2x2)
    # B12 is top-right of B1. Cols 2,3.
    # j=0, k=2 -> 2; j=0, k=3 -> 3; j=1, k=2 -> 6; j=1, k=3 -> 7.
    v_B12 = [2, 3, 6, 7]
    
    # Output is top-right 2x2 of result. Cols 2,3.
    # k=2, i=0 -> 4; k=2, i=1 -> 5; k=3, i=0 -> 6; k=3, i=1 -> 7.
    w_C12 = [4, 5, 6, 7]
    
    col = add_strassen(u_A1, v_B12, w_C12, col)
    
    # 3. A2 B2 (2x1 * 1x4)
    # A2 is right column of A. j=2.
    # i=0, j=2 -> 2; i=1, j=2 -> 5.
    # B2 is bottom row of B. j=2.
    # k=0 -> 8; k=1 -> 9; k=2 -> 10; k=3 -> 11.
    
    # Outer product rank 4.
    # M_k = A2_i * B2_k.
    # For each k in 0..3:
    for k in range(4):
        U[2, col+k] = 1
        U[5, col+k] = 1
        V[8+k, col+k] = 1
        W[k*2+0, col+k] = 1
        W[k*2+1, col+k] = 1
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
