"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
import numpy as np

def _schoolbook(n, m, p):
    """
    Return U, V, W for schoolbook decomposition of n x m by m x p multiplication.
    U shape (n*m, n*m*p)
    V shape (m*p, n*m*p)
    W shape (p*n, n*m*p)
    """
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _strassen_2x2():
    """
    Return U, V, W for Strassen decomposition of 2x2 by 2x2 multiplication.
    U shape (4, 7)
    V shape (4, 7)
    W shape (4, 7)
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A11 + A22)(B12 + B21)
    U[0, 0] = 1; U[3, 0] = 1
    V[1, 0] = 1; V[2, 0] = 1
    W[1, 0] = 1; W[2, 0] = 1
    
    # M2 = (A21 + A22)B11
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = 1
    
    # M3 = A11(B21 - B22)
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[0, 2] = 1; W[3, 2] = 1
    
    # M4 = A22(B11 - B12)
    U[3, 3] = 1
    V[0, 3] = 1; V[1, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    # M5 = (A11 + A12)B22
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[1, 4] = 1; W[3, 4] = 1
    
    # M6 = (A21 - A11)(B11 + B12)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[2, 5] = 1; W[1, 5] = 1
    
    # M7 = (A12 - A22)(B21 + B22)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1; W[2, 6] = 1
    
    return U, V, W

def _embed_block(U_sub, V_sub, W_sub, 
                 U_rows, V_rows, W_rows, 
                 U_cols, V_cols, W_cols,
                 U_start, V_start, W_start,
                 U_end, V_end, W_end):
    """
    Helper to place a sub-decomposition into the larger matrices.
    """
    U_block = np.zeros((U_end - U_start, U_cols - U_cols)) # Placeholder, need correct shapes
    # Actually, let's just construct the final matrices directly in construct()
    # This helper is for clarity but I will implement the logic directly in construct()
    pass

def construct():
    """
    Construct a rank-39 exact tensor decomposition for 2x3 by 3x7 matrix multiplication.
    
    Strategy:
    Split A (2x3) into A1 (2x1) and A2 (2x2).
    Split B (3x7) into B1 (1x7) and B2 (2x7).
    C = A1*B1 + A2*B2.
    
    Part 1: A1 (2x1) * B1 (1x7). Rank = 2*1*7 = 14.
    Part 2: A2 (2x2) * B2 (2x7).
        Split B2 into B2a (2x3) and B2b (2x4).
        A2*B2a:
            Split B2a into B2a1 (2x2) and B2a2 (2x1).
            A2*B2a1: Strassen 2x2*2x2. Rank 7.
            A2*B2a2: Schoolbook 2x2*2x1. Rank 4.
            Total 11.
        A2*B2b:
            Split B2b into B2b1 (2x2) and B2b2 (2x2).
            A2*B2b1: Strassen 2x2*2x2. Rank 7.
            A2*B2b2: Strassen 2x2*2x2. Rank 7.
            Total 14.
        Total for Part 2: 25.
    
    Total Rank: 14 + 25 = 39.
    """
    n, m, p = DIMENSIONS
    
    # Global shapes
    U_total = np.zeros((n * m, 39))  # 6 x 39
    V_total = np.zeros((m * p, 39))  # 21 x 39
    W_total = np.zeros((p * n, 39))  # 14 x 39
    
    col = 0
    
    # --- Part 1: A1 (2x1) * B1 (1x7) ---
    # A1 corresponds to rows 0,1 of A (col 0 of original 3 cols)
    # B1 corresponds to row 0 of B (col 0 of original 3 rows)
    # U1: 2x14, V1: 7x14, W1: 14x14
    U1, V1, W1 = _schoolbook(2, 1, 7)
    
    # Map indices:
    # U1 rows: 0,1 -> U_total rows 0,1
    # V1 rows: 0..6 -> V_total rows 0..6 (since B row 0 is first)
    # W1 rows: 0..13 -> W_total rows 0..13
    
    U_total[0:2, col:col+14] = U1
    V_total[0:7, col:col+14] = V1
    W_total[0:14, col:col+14] = W1
    col += 14
    
    # --- Part 2: A2 (2x2) * B2 (2x7) ---
    # A2 corresponds to rows 0,1 of A (cols 1,2 of original)
    # B2 corresponds to rows 1,2 of B
    
    # Sub-part 2a: A2 * B2a (2x3)
    # B2a corresponds to B rows 1,2 and cols 0,1,2
    
    # Sub-sub-part 2a1: A2 * B2a1 (2x2) -> Strassen
    # B2a1 corresponds to B rows 1,2 and cols 0,1
    U2a1, V2a1, W2a1 = _strassen_2x2()
    
    # Map indices for 2a1:
    # U2a1 rows 0..3 -> A2 indices. 
    # A2 local row 0 = A global (0,1) -> U_total row 1
    # A2 local row 1 = A global (0,2) -> U_total row 2
    # A2 local row 2 = A global (1,1) -> U_total row 3
    # A2 local row 3 = A global (1,2) -> U_total row 4
    # So U_total rows [1, 2, 3, 4]
    
    # V2a1 rows 0..3 -> B2a1 indices.
    # B2a1 local col 0 = B global (1,0) -> V_total row 1*7 + 0 = 7
    # B2a1 local col 1 = B global (1,1) -> V_total row 1*7 + 1 = 8
    # B2a1 local col 2 = B global (2,0) -> V_total row 2*7 + 0 = 14
    # B2a1 local col 3 = B global (2,1) -> V_total row 2*7 + 1 = 15
    # So V_total rows [7, 8, 14, 15]
    
    # W2a1 rows 0..3 -> C indices.
    # C local (0,0) = C global (0,0) -> W_total row 0*2 + 0 = 0
    # C local (0,1) = C global (0,1) -> W_total row 0*2 + 1 = 1
    # C local (1,0) = C global (1,0) -> W_total row 1*2 + 0 = 2
    # C local (1,1) = C global (1,1) -> W_total row 1*2 + 1 = 3
    # So W_total rows [0, 1, 2, 3]
    
    U_rows_2a1 = [1, 2, 3, 4]
    V_rows_2a1 = [7, 8, 14, 15]
    W_rows_2a1 = [0, 1, 2, 3]
    
    U_total[U_rows_2a1, col:col+7] = U2a1
    V_total[V_rows_2a1, col:col+7] = V2a1
    W_total[W_rows_2a1, col:col+7] = W2a1
    col += 7
    
    # Sub-sub-part 2a2: A2 * B2a2 (2x1) -> Schoolbook
    # B2a2 corresponds to B rows 1,2 and col 2
    U2a2, V2a2, W2a2 = _schoolbook(2, 2, 1)
    
    # Map indices for 2a2:
    # U2a2 rows 0..3 -> U_total rows [1, 2, 3, 4] (same as above)
    # V2a2 rows 0..3 -> B2a2 indices.
    # B2a2 local col 0 = B global (1,2) -> V_total row 1*7 + 2 = 9
    # B2a2 local col 1 = B global (2,2) -> V_total row 2*7 + 2 = 16
    # Wait, schoolbook for 2x2 by 2x1 has V shape (2*1, rank). 
    # My _schoolbook returns V shape (m*p, rank). Here m=2, p=1. So V is 2x4.
    # But B2a2 has 2 rows (from B) and 1 col. So the "m" in schoolbook is the middle dim of the sub-problem.
    # The sub-problem is A2 (2x2) * B2a2 (2x1).
    # So m_sub = 2, p_sub = 1.
    # V2a2 shape is (2*1, 4) = (2, 4).
    # V2a2 row 0 corresponds to B2a2 row 0, col 0 -> B global (1,2) -> V_total 9
    # V2a2 row 1 corresponds to B2a2 row 1, col 0 -> B global (2,2) -> V_total 16
    
    # W2a2 rows 0..3 -> C indices.
    # C local (0,0) = C global (0,2) -> W_total row 0*2 + 2 = 2
    # C local (1,0) = C global (1,2) -> W_total row 1*2 + 2 = 4
    # Wait, W shape is (p*n, rank). p_sub=1, n_sub=2. So W is 2x4.
    # W2a2 row 0 corresponds to C local (0,0) -> W_total 2
    # W2a2 row 1 corresponds to C local (1,0) -> W_total 4
    
    U_rows_2a2 = [1, 2, 3, 4]
    V_rows_2a2 = [9, 16]
    W_rows_2a2 = [2, 4]
    
    U_total[U_rows_2a2, col:col+4] = U2a2
    V_total[V_rows_2a2, col:col+4] = V2a2
    W_total[W_rows_2a2, col:col+4] = W2a2
    col += 4
    
    # Sub-part 2b: A2 * B2b (2x4)
    # B2b corresponds to B rows 1,2 and cols 3,4,5,6
    
    # Sub-sub-part 2b1: A2 * B2b1 (2x2) -> Strassen
    # B2b1 corresponds to B rows 1,2 and cols 3,4
    U2b1, V2b1, W2b1 = _strassen_2x2()
    
    # Map indices for 2b1:
    # U2b1 rows 0..3 -> U_total rows [1, 2, 3, 4]
    # V2b1 rows 0..3 -> B2b1 indices.
    # B2b1 local col 0 = B global (1,3) -> V_total row 1*7 + 3 = 10
    # B2b1 local col 1 = B global (1,4) -> V_total row 1*7 + 4 = 11
    # B2b1 local col 2 = B global (2,3) -> V_total row 2*7 + 3 = 17
    # B2b1 local col 3 = B global (2,4) -> V_total row 2*7 + 4 = 18
    # So V_total rows [10, 11, 17, 18]
    
    # W2b1 rows 0..3 -> C indices.
    # C local (0,0) = C global (0,3) -> W_total row 0*2 + 3 = 3
    # C local (0,1) = C global (0,4) -> W_total row 0*2 + 4 = 4
    # C local (1,0) = C global (1,3) -> W_total row 1*2 + 3 = 5
    # C local (1,1) = C global (1,4) -> W_total row 1*2 + 4 = 6
    # So W_total rows [3, 4, 5, 6]
    
    U_rows_2b1 = [1, 2, 3, 4]
    V_rows_2b1 = [10, 11, 17, 18]
    W_rows_2b1 = [3, 4, 5, 6]
    
    U_total[U_rows_2b1, col:col+7] = U2b1
    V_total[V_rows_2b1, col:col+7] = V2b1
    W_total[W_rows_2b1, col:col+7] = W2b1
    col += 7
    
    # Sub-sub-part 2b2: A2 * B2b2 (2x2) -> Strassen
    # B2b2 corresponds to B rows 1,2 and cols 5,6
    U2b2, V2b2, W2b2 = _strassen_2x2()
    
    # Map indices for 2b2:
    # U2b2 rows 0..3 -> U_total rows [1, 2, 3, 4]
    # V2b2 rows 0..3 -> B2b2 indices.
    # B2b2 local col 0 = B global (1,5) -> V_total row 1*7 + 5 = 12
    # B2b2 local col 1 = B global (1,6) -> V_total row 1*7 + 6 = 13
    # B2b2 local col 2 = B global (2,5) -> V_total row 2*7 + 5 = 19
    # B2b2 local col 3 = B global (2,6) -> V_total row 2*7 + 6 = 20
    # So V_total rows [12, 13, 19, 20]
    
    # W2b2 rows 0..3 -> C indices.
    # C local (0,0) = C global (0,5) -> W_total row 0*2 + 5 = 5
    # C local (0,1) = C global (0,6) -> W_total row 0*2 + 6 = 6
    # C local (1,0) = C global (1,5) -> W_total row 1*2 + 5 = 7
    # C local (1,1) = C global (1,6) -> W_total row 1*2 + 6 = 8
    # So W_total rows [5, 6, 7, 8]
    
    U_rows_2b2 = [1, 2, 3, 4]
    V_rows_2b2 = [12, 13, 19, 20]
    W_rows_2b2 = [5, 6, 7, 8]
    
    U_total[U_rows_2b2, col:col+7] = U2b2
    V_total[V_rows_2b2, col:col+7] = V2b2
    W_total[W_rows_2b2, col:col+7] = W2b2
    col += 7
    
    # Verify col is 39
    # 14 + 7 + 4 + 7 + 7 = 39
    
    return {"U": U_total.tolist(), "V": V_total.tolist(), "W": W_total.tolist()}

# EVOLVE-BLOCK-END
