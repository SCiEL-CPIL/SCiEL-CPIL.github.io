import numpy as np
import time

DIMENSIONS = [3, 5, 6]

def construct():
    """
    Constructs a rank-88 tensor decomposition for 3x5 by 5x6 matrix multiplication.
    Uses Strassen for 2x2x2 blocks within the 3x3x3 sub-problems.
    """
    n, m, p = DIMENSIONS
    # We aim for rank 88.
    # Start with empty lists for U, V, W terms
    # Each term is (U_row, V_row, W_row, sign)
    # Actually, we build the matrices directly.
    # Rank estimate: 36 (Part 1) + 26 (Part 2a) + 26 (Part 2b) = 88.
    
    rank = 88
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Helper to add a schoolbook term
    def add_schoolbook_term(i, j, k):
        nonlocal col
        if col >= rank:
            raise ValueError("Rank overflow")
        u_idx = i * m + j
        v_idx = j * p + k
        w_idx = k * n + i
        U[u_idx, col] = 1
        V[v_idx, col] = 1
        W[w_idx, col] = 1
        col += 1
        
    # Helper to add Strassen terms for 2x2x2 block at offset (i_off, j_off, k_off)
    def add_strassen_term(i_off, j_off, k_off):
        nonlocal col
        # A is at i_off, i_off+1 and j_off, j_off+1
        # B is at j_off, j_off+1 and k_off, k_off+1
        # C is at k_off, k_off+1 and i_off, i_off+1
        
        # M1 = (A00+A11)(B00+B11)
        if col >= rank: raise ValueError("Rank overflow")
        U[i_off*m + j_off, col] = 1
        U[(i_off+1)*m + (j_off+1), col] = 1
        V[j_off*p + k_off, col] = 1
        V[(j_off+1)*p + (k_off+1), col] = 1
        W[k_off*n + i_off, col] = 1   # C00 += M1
        W[(k_off+1)*n + i_off, col] = 1 # C01 += M1 (Wait, C01 is M1 - M2 + M3 + M6? No)
        # Let's re-verify Strassen C mapping.
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # So M1 contributes to C00 (+1) and C11 (+1).
        W[k_off*n + i_off, col] = 1
        W[(k_off+1)*n + (i_off+1), col] = 1
        col += 1
        
        # M2 = (A10+A11)B00
        if col >= rank: raise ValueError("Rank overflow")
        U[(i_off+1)*m + j_off, col] = 1
        U[(i_off+1)*m + (j_off+1), col] = 1
        V[j_off*p + k_off, col] = 1
        W[(k_off)*n + (i_off+1), col] = 1 # C10 += M2
        W[(k_off+1)*n + (i_off+1), col] = -1 # C11 += -M2
        col += 1
        
        # M3 = A00(B10-B11)
        if col >= rank: raise ValueError("Rank overflow")
        U[i_off*m + j_off, col] = 1
        V[(j_off+1)*p + k_off, col] = 1
        V[(j_off+1)*p + (k_off+1), col] = -1
        W[(k_off+1)*n + i_off, col] = 1 # C01 += M3
        W[(k_off+1)*n + (i_off+1), col] = 1 # C11 += M3
        col += 1
        
        # M4 = A11(B01-B00)
        if col >= rank: raise ValueError("Rank overflow")
        U[(i_off+1)*m + (j_off+1), col] = 1
        V[j_off*p + (k_off+1), col] = 1
        V[j_off*p + k_off, col] = -1
        W[(k_off)*n + i_off, col] = 1 # C00 += M4
        W[(k_off)*n + (i_off+1), col] = 1 # C10 += M4
        col += 1
        
        # M5 = (A00+A01)B11
        if col >= rank: raise ValueError("Rank overflow")
        U[i_off*m + j_off, col] = 1
        U[i_off*m + (j_off+1), col] = 1
        V[(j_off+1)*p + (k_off+1), col] = 1
        W[(k_off)*n + i_off, col] = -1 # C00 += -M5
        W[(k_off+1)*n + i_off, col] = 1 # C01 += M5
        col += 1
        
        # M6 = (A10-A00)(B00+B01)
        if col >= rank: raise ValueError("Rank overflow")
        U[(i_off+1)*m + j_off, col] = 1
        U[i_off*m + j_off, col] = -1
        V[j_off*p + k_off, col] = 1
        V[j_off*p + (k_off+1), col] = 1
        W[(k_off+1)*n + (i_off+1), col] = 1 # C11 += M6
        col += 1
        
        # M7 = (A01-A11)(B10+B11)
        if col >= rank: raise ValueError("Rank overflow")
        U[i_off*m + (j_off+1), col] = 1
        U[(i_off+1)*m + (j_off+1), col] = -1
        V[(j_off+1)*p + k_off, col] = 1
        V[(j_off+1)*p + (k_off+1), col] = 1
        W[(k_off)*n + i_off, col] = 1 # C00 += M7
        col += 1

    # Part 1: 3 x 2 x 6 (i:0..2, j:0..1, k:0..5)
    for i in range(3):
        for j in range(2):
            for k in range(6):
                add_schoolbook_term(i, j, k)
    
    # Part 2: 3 x 3 x 6 (i:0..2, j:2..4, k:0..5)
    # Split k into 0..2 and 3..5
    # Sub-problem 2a: i:0..2, j:2..4, k:0..2
    # Sub-problem 2b: i:0..2, j:2..4, k:3..5
    
    # Inside 2a and 2b, we use the 2+1 split for 3x3x3
    # Local indices for 3x3x3: i_l:0..2, j_l:0..2, k_l:0..2
    # Global i = i_l
    # Global j = j_off + j_l (j_off=2)
    # Global k = k_off + k_l
    
    for k_off in [0, 3]:
        j_off = 2
        i_off = 0
        
        # Block 1: 2x2x2 (i_l 0..1, j_l 0..1, k_l 0..1) -> Strassen
        add_strassen_term(i_off, j_off, k_off)
        
        # Block 2: 2x2x1 (i_l 0..1, j_l 0..1, k_l 2) -> Schoolbook
        for i in range(2):
            for j in range(2):
                add_schoolbook_term(i_off + i, j_off + j, k_off + 2)
        
        # Block 3: 2x1x2 (i_l 0..1, j_l 2, k_l 0..1) -> Schoolbook
        for i in range(2):
            for k in range(2):
                add_schoolbook_term(i_off + i, j_off + 2, k_off + k)
        
        # Block 4: 2x1x1 (i_l 0..1, j_l 2, k_l 2) -> Schoolbook
        for i in range(2):
            add_schoolbook_term(i_off + i, j_off + 2, k_off + 2)
        
        # Block 5: 1x2x2 (i_l 2, j_l 0..1, k_l 0..1) -> Schoolbook
        for j in range(2):
            for k in range(2):
                add_schoolbook_term(i_off + 2, j_off + j, k_off + k)
        
        # Block 6: 1x2x1 (i_l 2, j_l 0..1, k_l 2) -> Schoolbook
        for j in range(2):
            add_schoolbook_term(i_off + 2, j_off + j, k_off + 2)
        
        # Block 7: 1x1x2 (i_l 2, j_l 2, k_l 0..1) -> Schoolbook
        for k in range(2):
            add_schoolbook_term(i_off + 2, j_off + 2, k_off + k)
        
        # Block 8: 1x1x1 (i_l 2, j_l 2, k_l 2) -> Schoolbook
        add_schoolbook_term(i_off + 2, j_off + 2, k_off + 2)

    if col != rank:
        print(f"Warning: Used {col} terms out of {rank}.")
        
    return {"U": U, "V": V, "W": W}
