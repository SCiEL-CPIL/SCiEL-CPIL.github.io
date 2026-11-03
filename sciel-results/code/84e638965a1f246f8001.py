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
    
    def add_strassen(a_idx, b_idx, c_idx, start_col):
        U[a_idx[(0,0)], start_col] = 1.0
        U[a_idx[(1,1)], start_col] = 1.0
        V[b_idx[(0,0)], start_col] = 1.0
        V[b_idx[(1,1)], start_col] = 1.0
        W[c_idx[(0,0)], start_col] = 1.0
        W[c_idx[(1,1)], start_col] = 1.0
        col = start_col + 1
        
        U[a_idx[(1,0)], col] = 1.0
        U[a_idx[(1,1)], col] = 1.0
        V[b_idx[(0,0)], col] = 1.0
        W[c_idx[(1,0)], col] = 1.0
        W[c_idx[(1,1)], col] = -1.0
        col += 1
        
        U[a_idx[(0,0)], col] = 1.0
        V[b_idx[(0,1)], col] = 1.0
        V[b_idx[(1,1)], col] = -1.0
        W[c_idx[(0,1)], col] = 1.0
        W[c_idx[(1,1)], col] = 1.0
        col += 1
        
        U[a_idx[(1,1)], col] = 1.0
        V[b_idx[(1,0)], col] = 1.0
        V[b_idx[(0,0)], col] = -1.0
        W[c_idx[(0,0)], col] = 1.0
        W[c_idx[(1,0)], col] = 1.0
        col += 1
        
        U[a_idx[(0,0)], col] = 1.0
        U[a_idx[(0,1)], col] = 1.0
        V[b_idx[(1,1)], col] = 1.0
        W[c_idx[(0,0)], col] = -1.0
        W[c_idx[(0,1)], col] = 1.0
        col += 1
        
        U[a_idx[(1,0)], col] = 1.0
        U[a_idx[(0,0)], col] = -1.0
        V[b_idx[(0,0)], col] = 1.0
        V[b_idx[(0,1)], col] = 1.0
        W[c_idx[(1,1)], col] = 1.0
        col += 1
        
        U[a_idx[(0,1)], col] = 1.0
        U[a_idx[(1,1)], col] = -1.0
        V[b_idx[(1,0)], col] = 1.0
        V[b_idx[(1,1)], col] = 1.0
        W[c_idx[(0,0)], col] = 1.0
        col += 1
        
        return col

    def add_schoolbook_221(a_idx, b_idx, c_idx, start_col):
        U[a_idx[(0,0)], start_col] = 1.0
        V[b_idx[(0,0)], start_col] = 1.0
        W[c_idx[(0,0)], start_col] = 1.0
        col = start_col + 1
        
        U[a_idx[(0,1)], col] = 1.0
        V[b_idx[(1,0)], col] = 1.0
        W[c_idx[(0,0)], col] = 1.0
        col += 1
        
        U[a_idx[(1,0)], col] = 1.0
        V[b_idx[(0,0)], col] = 1.0
        W[c_idx[(1,0)], col] = 1.0
        col += 1
        
        U[a_idx[(1,1)], col] = 1.0
        V[b_idx[(1,0)], col] = 1.0
        W[c_idx[(1,0)], col] = 1.0
        col += 1
        
        return col

    a1_idx = {(0,0): 0, (0,1): 1, (1,0): 4, (1,1): 5}
    b1a_idx = {(0,0): 0, (0,1): 1, (1,0): 7, (1,1): 8}
    c1a_idx = {(0,0): 0, (0,1): 2, (1,0): 1, (1,1): 3}
    col = add_strassen(a1_idx, b1a_idx, c1a_idx, col)
    
    b1b_idx = {(0,0): 2, (0,1): 3, (1,0): 9, (1,1): 10}
    c1b_idx = {(0,0): 4, (0,1): 6, (1,0): 5, (1,1): 7}
    col = add_strassen(a1_idx, b1b_idx, c1b_idx, col)
    
    b1c_idx = {(0,0): 4, (0,1): 5, (1,0): 11, (1,1): 12}
    c1c_idx = {(0,0): 8, (0,1): 10, (1,0): 9, (1,1): 11}
    col = add_strassen(a1_idx, b1c_idx, c1c_idx, col)
    
    b1d_idx = {(0,0): 6, (1,0): 13}
    c1d_idx = {(0,0): 12, (1,0): 13}
    col = add_schoolbook_221(a1_idx, b1d_idx, c1d_idx, col)
    
    a2_idx = {(0,0): 2, (0,1): 3, (1,0): 6, (1,1): 7}
    
    b2a_idx = {(0,0): 14, (0,1): 15, (1,0): 21, (1,1): 22}
    c2a_idx = {(0,0): 0, (0,1): 2, (1,0): 1, (1,1): 3}
    col = add_strassen(a2_idx, b2a_idx, c2a_idx, col)
    
    b2b_idx = {(0,0): 16, (0,1): 17, (1,0): 23, (1,1): 24}
    c2b_idx = {(0,0): 4, (0,1): 6, (1,0): 5, (1,1): 7}
    col = add_strassen(a2_idx, b2b_idx, c2b_idx, col)
    
    b2c_idx = {(0,0): 18, (0,1): 19, (1,0): 25, (1,1): 26}
    c2c_idx = {(0,0): 8, (0,1): 10, (1,0): 9, (1,1): 11}
    col = add_strassen(a2_idx, b2c_idx, c2c_idx, col)
    
    b2d_idx = {(0,0): 20, (1,0): 27}
    c2d_idx = {(0,0): 12, (1,0): 13}
    col = add_schoolbook_221(a2_idx, b2d_idx, c2d_idx, col)
    
    if col != rank:
        pass
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
