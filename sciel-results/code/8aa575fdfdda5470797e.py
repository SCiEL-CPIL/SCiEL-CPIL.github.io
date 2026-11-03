import numpy as np

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    
    # Rank-50 decomposition using Strassen for 2x2x2 blocks and schoolbook for 2x2x1 blocks
    # Split m=4 into two blocks of 2: j-blocks [0,1] and [2,3]
    # Split p=7 into three blocks of 2 and one block of 1: k-blocks [0,1], [2,3], [4,5], [6]
    
    # For each A-block (2x2, since n=2) and B-block:
    #   - If B-block is 2x2: use Strassen (rank 7)
    #   - If B-block is 2x1: use schoolbook (rank 4)
    # Total: 2 A-blocks * (3 * 7 + 1 * 4) = 2 * 25 = 50
    
    rank = 50
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    def add_strassen(a_idx, b_idx, c_idx, start_col):
        """Add a Strassen decomposition for a 2x2x2 block.
        a_idx: dict mapping (local_i, local_j) -> global U index
        b_idx: dict mapping (local_j, local_k) -> global V index
        c_idx: dict mapping (local_k, local_i) -> global W index
        """
        c = start_col
        # M1 = (A00 + A11)(B00 + B11) -> C00 += M1, C11 += M1
        U[a_idx[(0,0)], c] = 1.0
        U[a_idx[(1,1)], c] = 1.0
        V[b_idx[(0,0)], c] = 1.0
        V[b_idx[(1,1)], c] = 1.0
        W[c_idx[(0,0)], c] = 1.0
        W[c_idx[(1,1)], c] = 1.0
        c += 1
        
        # M2 = (A10 + A11)B00 -> C10 += M2, C11 -= M2
        U[a_idx[(1,0)], c] = 1.0
        U[a_idx[(1,1)], c] = 1.0
        V[b_idx[(0,0)], c] = 1.0
        W[c_idx[(1,0)], c] = 1.0
        W[c_idx[(1,1)], c] = -1.0
        c += 1
        
        # M3 = A00(B01 - B11) -> C01 += M3, C11 += M3
        U[a_idx[(0,0)], c] = 1.0
        V[b_idx[(0,1)], c] = 1.0
        V[b_idx[(1,1)], c] = -1.0
        W[c_idx[(0,1)], c] = 1.0
        W[c_idx[(1,1)], c] = 1.0
        c += 1
        
        # M4 = A11(B10 - B00) -> C00 += M4, C10 += M4
        U[a_idx[(1,1)], c] = 1.0
        V[b_idx[(1,0)], c] = 1.0
        V[b_idx[(0,0)], c] = -1.0
        W[c_idx[(0,0)], c] = 1.0
        W[c_idx[(1,0)], c] = 1.0
        c += 1
        
        # M5 = (A00 + A01)B11 -> C00 -= M5, C01 += M5
        U[a_idx[(0,0)], c] = 1.0
        U[a_idx[(0,1)], c] = 1.0
        V[b_idx[(1,1)], c] = 1.0
        W[c_idx[(0,0)], c] = -1.0
        W[c_idx[(0,1)], c] = 1.0
        c += 1
        
        # M6 = (A10 - A00)(B00 + B01) -> C11 += M6
        U[a_idx[(1,0)], c] = 1.0
        U[a_idx[(0,0)], c] = -1.0
        V[b_idx[(0,0)], c] = 1.0
        V[b_idx[(0,1)], c] = 1.0
        W[c_idx[(1,1)], c] = 1.0
        c += 1
        
        # M7 = (A01 - A11)(B10 + B11) -> C00 += M7
        U[a_idx[(0,1)], c] = 1.0
        U[a_idx[(1,1)], c] = -1.0
        V[b_idx[(1,0)], c] = 1.0
        V[b_idx[(1,1)], c] = 1.0
        W[c_idx[(0,0)], c] = 1.0
        c += 1
        
        return c
    
    def add_schoolbook_221(a_idx, b_idx, c_idx, start_col):
        """Add a schoolbook decomposition for a 2x2x1 block.
        a_idx: dict mapping (local_i, local_j) -> global U index (2x2)
        b_idx: dict mapping (local_j, local_k) -> global V index (2x1)
        c_idx: dict mapping (local_k, local_i) -> global W index (1x2)
        """
        c = start_col
        # N1 = A00 * B0 -> C0 += N1
        U[a_idx[(0,0)], c] = 1.0
        V[b_idx[(0,0)], c] = 1.0
        W[c_idx[(0,0)], c] = 1.0
        c += 1
        
        # N2 = A01 * B1 -> C0 += N2
        U[a_idx[(0,1)], c] = 1.0
        V[b_idx[(1,0)], c] = 1.0
        W[c_idx[(0,0)], c] = 1.0
        c += 1
        
        # N3 = A10 * B0 -> C1 += N3
        U[a_idx[(1,0)], c] = 1.0
        V[b_idx[(0,0)], c] = 1.0
        W[c_idx[(0,1)], c] = 1.0
        c += 1
        
        # N4 = A11 * B1 -> C1 += N4
        U[a_idx[(1,1)], c] = 1.0
        V[b_idx[(1,0)], c] = 1.0
        W[c_idx[(0,1)], c] = 1.0
        c += 1
        
        return c
    
    # Define index mappings
    # A-block 1: i in {0,1}, j in {0,1} -> U indices: i*4 + j
    # A-block 2: i in {0,1}, j in {2,3} -> U indices: i*4 + j
    a1_idx = {(0,0): 0*4+0, (0,1): 0*4+1, (1,0): 1*4+0, (1,1): 1*4+1}
    a2_idx = {(0,0): 0*4+2, (0,1): 0*4+3, (1,0): 1*4+2, (1,1): 1*4+3}
    
    # B-block 1a: j in {0,1}, k in {0,1} -> V indices: j*7 + k
    b1a_idx = {(0,0): 0*7+0, (0,1): 0*7+1, (1,0): 1*7+0, (1,1): 1*7+1}
    # B-block 1b: j in {0,1}, k in {2,3}
    b1b_idx = {(0,0): 0*7+2, (0,1): 0*7+3, (1,0): 1*7+2, (1,1): 1*7+3}
    # B-block 1c: j in {0,1}, k in {4,5}
    b1c_idx = {(0,0): 0*7+4, (0,1): 0*7+5, (1,0): 1*7+4, (1,1): 1*7+5}
    # B-block 1d: j in {0,1}, k in {6}
    b1d_idx = {(0,0): 0*7+6, (1,0): 1*7+6}
    
    # B-block 2a: j in {2,3}, k in {0,1}
    b2a_idx = {(0,0): 2*7+0, (0,1): 2*7+1, (1,0): 3*7+0, (1,1): 3*7+1}
    # B-block 2b: j in {2,3}, k in {2,3}
    b2b_idx = {(0,0): 2*7+2, (0,1): 2*7+3, (1,0): 3*7+2, (1,1): 3*7+3}
    # B-block 2c: j in {2,3}, k in {4,5}
    b2c_idx = {(0,0): 2*7+4, (0,1): 2*7+5, (1,0): 3*7+4, (1,1): 3*7+5}
    # B-block 2d: j in {2,3}, k in {6}
    b2d_idx = {(0,0): 2*7+6, (1,0): 3*7+6}
    
    # C-block mappings: W index = k*n + i = k*2 + i
    # C-block 1a: k in {0,1}, i in {0,1}
    c1a_idx = {(0,0): 0*2+0, (0,1): 0*2+1, (1,0): 1*2+0, (1,1): 1*2+1}
    # C-block 1b: k in {2,3}, i in {0,1}
    c1b_idx = {(0,0): 2*2+0, (0,1): 2*2+1, (1,0): 3*2+0, (1,1): 3*2+1}
    # C-block 1c: k in {4,5}, i in {0,1}
    c1c_idx = {(0,0): 4*2+0, (0,1): 4*2+1, (1,0): 5*2+0, (1,1): 5*2+1}
    # C-block 1d: k in {6}, i in {0,1}
    c1d_idx = {(0,0): 6*2+0, (0,1): 6*2+1}
    
    # C-block 2a: k in {0,1}, i in {0,1} (same as c1a)
    c2a_idx = c1a_idx
    # C-block 2b: k in {2,3}, i in {0,1} (same as c1b)
    c2b_idx = c1b_idx
    # C-block 2c: k in {4,5}, i in {0,1} (same as c1c)
    c2c_idx = c1c_idx
    # C-block 2d: k in {6}, i in {0,1} (same as c1d)
    c2d_idx = c1d_idx
    
    # A-block 1 with B-blocks 1a, 1b, 1c (Strassen) and 1d (schoolbook)
    col = add_strassen(a1_idx, b1a_idx, c1a_idx, col)
    col = add_strassen(a1_idx, b1b_idx, c1b_idx, col)
    col = add_strassen(a1_idx, b1c_idx, c1c_idx, col)
    col = add_schoolbook_221(a1_idx, b1d_idx, c1d_idx, col)
    
    # A-block 2 with B-blocks 2a, 2b, 2c (Strassen) and 2d (schoolbook)
    col = add_strassen(a2_idx, b2a_idx, c2a_idx, col)
    col = add_strassen(a2_idx, b2b_idx, c2b_idx, col)
    col = add_strassen(a2_idx, b2c_idx, c2c_idx, col)
    col = add_schoolbook_221(a2_idx, b2d_idx, c2d_idx, col)
    
    if col != rank:
        raise ValueError(f"Column count {col} != expected rank {rank}")
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
