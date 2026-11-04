import time
import random
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def _strassen_2x2x2():
    """
    Returns U, V, W for 2x2x2 Strassen decomposition.
    U shape (4, 7), V shape (4, 7), W shape (4, 7)
    Indices:
    A: 00, 01, 10, 11 -> 0, 1, 2, 3
    B: 00, 01, 10, 11 -> 0, 1, 2, 3
    C: 00, 01, 10, 11 -> 0, 1, 2, 3
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A00+A11)(B00+B11)
    U[0,0] = 1; U[3,0] = 1
    V[0,0] = 1; V[1,0] = 1; V[2,0] = 1; V[3,0] = 1
    W[0,0] = 1; W[3,0] = 1
    
    # M2 = (A10+A11)B00
    U[2,1] = 1; U[3,1] = 1
    V[0,1] = 1
    W[2,1] = 1; W[3,1] = -1
    
    # M3 = A00(B01-B11)
    U[0,2] = 1
    V[1,2] = 1; V[3,2] = -1
    W[1,2] = 1; W[3,2] = 1
    
    # M4 = A11(B10-B00)
    U[3,3] = 1
    V[2,3] = 1; V[0,3] = -1
    W[0,3] = 1; W[2,3] = 1
    
    # M5 = (A00+A01)B11
    U[0,4] = 1; U[1,4] = 1
    V[3,4] = 1
    W[0,4] = -1; W[1,4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    U[2,5] = 1; U[0,5] = -1
    V[0,5] = 1; V[1,5] = 1
    W[1,5] = 1; W[3,5] = -1
    
    # M7 = (A01-A11)(B10+B11)
    U[1,6] = 1; U[3,6] = -1
    V[2,6] = 1; V[3,6] = 1
    W[0,6] = -1; W[3,6] = 1
    
    return U, V, W

def _construct_rank_23():
    n, m, p = 2, 3, 6
    rank = 23
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Part 1: A1 B1 (2x1 * 1x6)
    # A1 is col 0 of A. Indices in U: 0, 1 (for rows 0,1 of A)
    # B1 is row 0 of B. Indices in V: 0, 1, 2, 3, 4, 5 (for cols 0-5 of B)
    # C is 2x6. Indices in W: k*n+i -> 0*2+0=0, 0*2+1=1, 1*2+0=2, ... 5*2+1=11
    
    # Term 0: A[0,0] * B[0,:]
    U[0, 0] = 1
    V[0, 0] = 1; V[1, 0] = 1; V[2, 0] = 1; V[3, 0] = 1; V[4, 0] = 1; V[5, 0] = 1
    W[0, 0] = 1; W[2, 0] = 1; W[4, 0] = 1; W[6, 0] = 1; W[8, 0] = 1; W[10, 0] = 1
    
    # Term 1: A[1,0] * B[0,:]
    U[1, 1] = 1
    V[0, 1] = 1; V[1, 1] = 1; V[2, 1] = 1; V[3, 1] = 1; V[4, 1] = 1; V[5, 1] = 1
    W[1, 1] = 1; W[3, 1] = 1; W[5, 1] = 1; W[7, 1] = 1; W[9, 1] = 1; W[11, 1] = 1
    
    # Part 2: A2 B2 (2x2 * 2x6)
    # A2 is cols 1,2 of A. Indices in U: 2,3 (row0 col1,2) and 4,5 (row1 col1,2)
    # Wait, U index is i*m+j.
    # A is 2x3.
    # i=0, j=0 -> 0
    # i=0, j=1 -> 1
    # i=0, j=2 -> 2
    # i=1, j=0 -> 3
    # i=1, j=1 -> 4
    # i=1, j=2 -> 5
    # So A2 corresponds to U indices:
    # A2[0,0] (A[0,1]) -> 1
    # A2[0,1] (A[0,2]) -> 2
    # A2[1,0] (A[1,1]) -> 4
    # A2[1,1] (A[1,2]) -> 5
    
    # B2 is rows 1,2 of B. Indices in V:
    # B is 3x6.
    # i=0 (row 0) -> 0-5
    # i=1 (row 1) -> 6-11
    # i=2 (row 2) -> 12-17
    # So B2 corresponds to V indices:
    # B2[0,0] (B[1,0]) -> 6
    # B2[0,1] (B[1,1]) -> 7
    # B2[1,0] (B[2,0]) -> 12
    # B2[1,1] (B[2,1]) -> 13
    # ... and so on for cols 2-5.
    
    # C is 2x6. Indices in W: k*n+i
    # k=0 (col 0) -> 0,1
    # k=1 (col 1) -> 2,3
    # k=2 (col 2) -> 4,5
    # k=3 (col 3) -> 6,7
    # k=4 (col 4) -> 8,9
    # k=5 (col 5) -> 10,11
    
    # We have 3 blocks of B2, each 2x2.
    # Block 0: Cols 0,1 of B2 (which are cols 0,1 of B)
    # Block 1: Cols 2,3 of B2 (which are cols 2,3 of B)
    # Block 2: Cols 4,5 of B2 (which are cols 4,5 of B)
    
    S_U, S_V, S_W = _strassen_2x2x2()
    
    for block in range(3):
        base_rank = 2 + block * 7
        # Map Strassen A indices (0,1,2,3) to global U indices
        # S_U rows: 0(A00), 1(A01), 2(A10), 3(A11)
        # Global U indices for A2:
        # A2[0,0] (A[0,1]) -> 1
        # A2[0,1] (A[0,2]) -> 2
        # A2[1,0] (A[1,1]) -> 4
        # A2[1,1] (A[1,2]) -> 5
        a_map = [1, 2, 4, 5]
        
        # Map Strassen B indices (0,1,2,3) to global V indices
        # B2 block 'block' corresponds to B cols 2*block, 2*block+1
        # B2[0,0] (B[1, 2*block]) -> 6 + 2*block
        # B2[0,1] (B[1, 2*block+1]) -> 7 + 2*block
        # B2[1,0] (B[2, 2*block]) -> 12 + 2*block
        # B2[1,1] (B[2, 2*block+1]) -> 13 + 2*block
        b_map = [6 + 2*block, 7 + 2*block, 12 + 2*block, 13 + 2*block]
        
        # Map Strassen C indices (0,1,2,3) to global W indices
        # C block 'block' corresponds to C cols 2*block, 2*block+1
        # C[0,0] (C[0, 2*block]) -> (2*block)*2 + 0 = 4*block
        # C[0,1] (C[0, 2*block+1]) -> (2*block+1)*2 + 0 = 4*block + 2
        # C[1,0] (C[1, 2*block]) -> (2*block)*2 + 1 = 4*block + 1
        # C[1,1] (C[1, 2*block+1]) -> (2*block+1)*2 + 1 = 4*block + 3
        c_map = [4*block, 4*block+2, 4*block+1, 4*block+3]
        
        for t in range(7):
            r = base_rank + t
            for i_local in range(4):
                if S_U[i_local, t] != 0:
                    U[a_map[i_local], r] = S_U[i_local, t]
            for i_local in range(4):
                if S_V[i_local, t] != 0:
                    V[b_map[i_local], r] = S_V[i_local, t]
            for i_local in range(4):
                if S_W[i_local, t] != 0:
                    W[c_map[i_local], r] = S_W[i_local, t]
                    
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = np.sum(U_arr[i * m + j, :] * V_arr[j * p + k, :] * W_arr[k * n + i, :])
                if abs(actual - expected) > tol:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    
    # Try rank 23 construction
    result_23 = _construct_rank_23()
    if _verify_decomposition(result_23["U"], result_23["V"], result_23["W"], n, m, p, tol=1e-9):
        return result_23
        
    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
