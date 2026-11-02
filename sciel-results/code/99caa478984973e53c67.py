import numpy as np

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We will construct a rank-50 decomposition.
    # Split m=4 into two blocks of 2.
    # Split p=7 into three blocks of 2 and one block of 1.
    # For each A-block (2x2) and B-block (2x2 or 2x1), we use Strassen or Schoolbook.
    
    # Total rank = 2 * (3 * 7 + 1 * 4) = 2 * 25 = 50.
    rank = 50
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    # Helper to add a local decomposition to global matrices
    # local_a: 2x2 array of indices into U (i*m + j)
    # local_b: 2x2 or 2x1 array of indices into V (j*p + k)
    # local_c: 2x2 or 2x1 array of indices into W (k*n + i)
    # type: 'strassen' or 'schoolbook'
    
    def add_strassen(a_idx, b_idx, c_idx, start_col):
        # a_idx: dict {(0,0): idx, ...}
        # b_idx: dict {(0,0): idx, ...}
        # c_idx: dict {(0,0): idx, ...}
        
        # M1 = (A00 + A11)(B00 + B11)
        # U: A00, A11; V: B00, B11
        U[a_idx[(0,0)], start_col] = 1.0
        U[a_idx[(1,1)], start_col] = 1.0
        V[b_idx[(0,0)], start_col] = 1.0
        V[b_idx[(1,1)], start_col] = 1.0
        # W: C00 (+1), C11 (+1)
        W[c_idx[(0,0)], start_col] = 1.0
        W[c_idx[(1,1)], start_col] = 1.0
        col = start_col + 1
        
        # M2 = (A10 + A11)B00
        U[a_idx[(1,0)], col] = 1.0
        U[a_idx[(1,1)], col] = 1.0
        V[b_idx[(0,0)], col] = 1.0
        # W: C10 (+1), C11 (-1)
        W[c_idx[(1,0)], col] = 1.0
        W[c_idx[(1,1)], col] = -1.0
        col += 1
        
        # M3 = A00(B01 - B11)
        U[a_idx[(0,0)], col] = 1.0
        V[b_idx[(0,1)], col] = 1.0
        V[b_idx[(1,1)], col] = -1.0
        # W: C01 (+1), C11 (+1)
        W[c_idx[(0,1)], col] = 1.0
        W[c_idx[(1,1)], col] = 1.0
        col += 1
        
        # M4 = A11(B10 - B00)
        U[a_idx[(1,1)], col] = 1.0
        V[b_idx[(1,0)], col] = 1.0
        V[b_idx[(0,0)], col] = -1.0
        # W: C00 (+1), C10 (+1)
        W[c_idx[(0,0)], col] = 1.0
        W[c_idx[(1,0)], col] = 1.0
        col += 1
        
        # M5 = (A00 + A01)B11
        U[a_idx[(0,0)], col] = 1.0
        U[a_idx[(0,1)], col] = 1.0
        V[b_idx[(1,1)], col] = 1.0
        # W: C00 (-1), C01 (+1)
        W[c_idx[(0,0)], col] = -1.0
        W[c_idx[(0,1)], col] = 1.0
        col += 1
        
        # M6 = (A10 - A00)(B00 + B01)
        U[a_idx[(1,0)], col] = 1.0
        U[a_idx[(0,0)], col] = -1.0
        V[b_idx[(0,0)], col] = 1.0
        V[b_idx[(0,1)], col] = 1.0
        # W: C11 (+1)
        W[c_idx[(1,1)], col] = 1.0
        col += 1
        
        # M7 = (A01 - A11)(B10 + B11)
        U[a_idx[(0,1)], col] = 1.0
        U[a_idx[(1,1)], col] = -1.0
        V[b_idx[(1,0)], col] = 1.0
        V[b_idx[(1,1)], col] = 1.0
        # W: C00 (+1)
        W[c_idx[(0,0)], col] = 1.0
        col += 1
        
        return col

    def add_schoolbook_221(a_idx, b_idx, c_idx, start_col):
        # b_idx: dict {(0,0): idx, (1,0): idx} for 2x1
        # c_idx: dict {(0,0): idx, (1,0): idx} for 2x1
        
        # N1 = A00 B0
        U[a_idx[(0,0)], start_col] = 1.0
        V[b_idx[(0,0)], start_col] = 1.0
        W[c_idx[(0,0)], start_col] = 1.0
        col = start_col + 1
        
        # N2 = A01 B1
        U[a_idx[(0,1)], col] = 1.0
        V[b_idx[(1,0)], col] = 1.0
        W[c_idx[(0,0)], col] = 1.0
        col += 1
        
        # N3 = A10 B0
        U[a_idx[(1,0)], col] = 1.0
        V[b_idx[(0,0)], col] = 1.0
        W[c_idx[(1,0)], col] = 1.0
        col += 1
        
        # N4 = A11 B1
        U[a_idx[(1,1)], col] = 1.0
        V[b_idx[(1,0)], col] = 1.0
        W[c_idx[(1,0)], col] = 1.0
        col += 1
        
        return col

    # Block 1: A1 (cols 0,1 of A) and B1 (rows 0,1 of B)
    # A1 indices in U:
    # (0,0) -> 0*4+0 = 0
    # (0,1) -> 0*4+1 = 1
    # (1,0) -> 1*4+0 = 4
    # (1,1) -> 1*4+1 = 5
    a1_idx = {(0,0): 0, (0,1): 1, (1,0): 4, (1,1): 5}
    
    # B1 indices in V:
    # B is 4x7. Rows 0,1. Cols 0..6.
    # V index = j*p + k.
    # (0,0) -> 0*7+0 = 0
    # (0,1) -> 0*7+1 = 1
    # (1,0) -> 1*7+0 = 7
    # (1,1) -> 1*7+1 = 8
    # ...
    
    # C1 indices in W:
    # C is 2x7. Rows 0,1. Cols 0..6.
    # W index = k*n + i.
    # (0,0) -> 0*2+0 = 0
    # (0,1) -> 1*2+0 = 2
    # (1,0) -> 0*2+1 = 1
    # (1,1) -> 1*2+1 = 3
    
    # Process B1 blocks
    # B1a: Cols 0,1. 2x2.
    b1a_idx = {(0,0): 0, (0,1): 1, (1,0): 7, (1,1): 8}
    c1a_idx = {(0,0): 0, (0,1): 2, (1,0): 1, (1,1): 3}
    col = add_strassen(a1_idx, b1a_idx, c1a_idx, col)
    
    # B1b: Cols 2,3. 2x2.
    b1b_idx = {(0,0): 2, (0,1): 3, (1,0): 9, (1,1): 10}
    c1b_idx = {(0,0): 4, (0,1): 6, (1,0): 5, (1,1): 7}
    col = add_strassen(a1_idx, b1b_idx, c1b_idx, col)
    
    # B1c: Cols 4,5. 2x2.
    b1c_idx = {(0,0): 4, (0,1): 5, (1,0): 11, (1,1): 12}
    c1c_idx = {(0,0): 8, (0,1): 10, (1,0): 9, (1,1): 11}
    col = add_strassen(a1_idx, b1c_idx, c1c_idx, col)
    
    # B1d: Col 6. 2x1.
    b1d_idx = {(0,0): 6, (1,0): 13}
    c1d_idx = {(0,0): 12, (1,0): 13}
    col = add_schoolbook_221(a1_idx, b1d_idx, c1d_idx, col)
    
    # Block 2: A2 (cols 2,3 of A) and B2 (rows 2,3 of B)
    # A2 indices in U:
    # (0,0) -> 0*4+2 = 2
    # (0,1) -> 0*4+3 = 3
    # (1,0) -> 1*4+2 = 6
    # (1,1) -> 1*4+3 = 7
    a2_idx = {(0,0): 2, (0,1): 3, (1,0): 6, (1,1): 7}
    
    # B2 indices in V:
    # Rows 2,3.
    # (0,0) -> 2*7+0 = 14
    # (0,1) -> 2*7+1 = 15
    # (1,0) -> 3*7+0 = 21
    # (1,1) -> 3*7+1 = 22
    # ...
    
    # C2 indices in W: Same C indices as C1? No, C is the sum.
    # The output C is the same matrix. The W matrix accumulates contributions.
    # So c2_idx is the same as c1_idx.
    
    # B2a: Cols 0,1.
    b2a_idx = {(0,0): 14, (0,1): 15, (1,0): 21, (1,1): 22}
    c2a_idx = {(0,0): 0, (0,1): 2, (1,0): 1, (1,1): 3}
    col = add_strassen(a2_idx, b2a_idx, c2a_idx, col)
    
    # B2b: Cols 2,3.
    b2b_idx = {(0,0): 16, (0,1): 17, (1,0): 23, (1,1): 24}
    c2b_idx = {(0,0): 4, (0,1): 6, (1,0): 5, (1,1): 7}
    col = add_strassen(a2_idx, b2b_idx, c2b_idx, col)
    
    # B2c: Cols 4,5.
    b2c_idx = {(0,0): 18, (0,1): 19, (1,0): 25, (1,1): 26}
    c2c_idx = {(0,0): 8, (0,1): 10, (1,0): 9, (1,1): 11}
    col = add_strassen(a2_idx, b2c_idx, c2c_idx, col)
    
    # B2d: Col 6.
    b2d_idx = {(0,0): 20, (1,0): 27}
    c2d_idx = {(0,0): 12, (1,0): 13}
    col = add_schoolbook_221(a2_idx, b2d_idx, c2d_idx, col)
    
    # Check if col == rank
    if col != rank:
        # Fallback or error
        pass
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
