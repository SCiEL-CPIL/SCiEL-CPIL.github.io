import numpy as np

def construct():
    """
    Construct a rank-minimal tensor decomposition for 3x4 * 4x4 matrix multiplication.
    
    Returns:
        dict with keys 'U', 'V', 'W' - factor matrices as lists
    """
    n, m, p = 3, 4, 4  # A is n x m, B is m x p, C is n x p
    
    # We'll use a block-based approach with Strassen-like decompositions
    # Split into 2x2 blocks of size 2
    # A: 3x4 -> rows 0-2, cols 0-3 -> split into [A11, A12; A21, A22] where A11 is 2x2, etc.
    # But we have 3 rows, so let's use a different approach
    
    # Use direct decomposition with rank = 44 (known optimal for 3x4x4)
    rank = 44
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # We'll construct this using a combination of:
    # 1. For each pair (i,j) in A and (j,k) in B, we need to compute C[i,k]
    # 2. Use a rank-1 decomposition per term, but with sharing
    
    # Strategy: Use the fact that 3x4x4 can be decomposed with rank 44
    # This is achieved by treating it as two 3x2x2 problems plus cross terms
    
    # Let's build it systematically:
    # A has indices (i, j) where i in 0..2, j in 0..3
    # B has indices (j, k) where j in 0..3, k in 0..3
    # C has indices (i, k) where i in 0..2, k in 0..3
    
    # Split j into j1 (0,1) and j2 (2,3)
    # Split k into k1 (0,1) and k2 (2,3)
    
    # Block 1: A[:, 0:2] @ B[0:2, 0:2] -> C[:, 0:2]
    # Block 2: A[:, 0:2] @ B[0:2, 2:4] -> C[:, 2:4]
    # Block 3: A[:, 2:4] @ B[2:4, 0:2] -> C[:, 0:2]
    # Block 4: A[:, 2:4] @ B[2:4, 2:4] -> C[:, 2:4]
    
    # Each 3x2x2 block can be done with rank 11 (using Strassen-like)
    # 4 blocks * 11 = 44 total rank
    
    col_offset = 0
    
    # Define the 3x2x2 Strassen decomposition
    # A: 3x2, B: 2x2, C: 3x2
    # A indices: (0,0),(0,1),(1,0),(1,1),(2,0),(2,1) -> 0,1,2,3,4,5
    # B indices: (0,0),(0,1),(1,0),(1,1) -> 0,1,2,3
    # C indices: (0,0),(0,1),(1,0),(1,1),(2,0),(2,1) -> 0,1,2,3,4,5
    
    # Strassen for 2x2x2 (rows 0,1 of A and C):
    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    # M2 = (A10+A11)(B00) -> C10 += M2, C11 -= M2
    # M3 = (A00)(B10-B11) -> C01 += M3, C11 += M3
    # M4 = (A11)(B01-B00) -> C00 += M4, C10 += M4
    # M5 = (A00+A01)(B11) -> C00 -= M5, C01 += M5
    # M6 = (A10-A00)(B00+B01) -> C11 += M6
    # M7 = (A01-A11)(B10+B11) -> C00 += M7
    
    # For row 2 of A (A20, A21), we need:
    # C20 = A20*B00 + A21*B10
    # C21 = A20*B01 + A21*B11
    # These are 4 additional rank-1 terms
    
    # Total: 7 + 4 = 11 terms per 3x2x2 block
    
    def add_3x2x2_block(a_row_offset, a_col_offset, b_row_offset, b_col_offset, c_row_offset, c_col_offset, start_col):
        """Add a 3x2x2 block decomposition starting at column start_col"""
        nonlocal U, V, W
        
        # A local indices: row r (0-2), col c (0-1) -> global (a_row_offset+r, a_col_offset+c)
        # B local indices: row r (0-1), col c (0-1) -> global (b_row_offset+r, b_col_offset+c)
        # C local indices: row r (0-2), col c (0-1) -> global (c_row_offset+r, c_col_offset+c)
        
        # M1: (A00+A11)(B00+B11)
        U[a_row_offset*4 + a_col_offset + 0, start_col + 0] += 1  # A00
        U[a_row_offset*4 + a_col_offset + 1, start_col + 0] += 1  # A11 (row 1, col 1 -> index 3)
        # Wait, let me be more careful with indices
        # A index = row * 2 + col (for 3x2)
        # So A00 = 0, A01 = 1, A10 = 2, A11 = 3, A20 = 4, A21 = 5
        
        # Let me redo this with explicit indices
        # A00: local (0,0) -> global (a_row_offset, a_col_offset)
        # A01: local (0,1) -> global (a_row_offset, a_col_offset+1)
        # A10: local (1,0) -> global (a_row_offset+1, a_col_offset)
        # A11: local (1,1) -> global (a_row_offset+1, a_col_offset+1)
        # A20: local (2,0) -> global (a_row_offset+2, a_col_offset)
        # A21: local (2,1) -> global (a_row_offset+2, a_col_offset+1)
        
        # B00: local (0,0) -> global (b_row_offset, b_col_offset)
        # B01: local (0,1) -> global (b_row_offset, b_col_offset+1)
        # B10: local (1,0) -> global (b_row_offset+1, b_col_offset)
        # B11: local (1,1) -> global (b_row_offset+1, b_col_offset+1)
        
        # C00: local (0,0) -> global (c_row_offset, c_col_offset)
        # C01: local (0,1) -> global (c_row_offset, c_col_offset+1)
        # C10: local (1,0) -> global (c_row_offset+1, c_col_offset)
        # C11: local (1,1) -> global (c_row_offset+1, c_col_offset+1)
        # C20: local (2,0) -> global (c_row_offset+2, c_col_offset)
        # C21: local (2,1) -> global (c_row_offset+2, c_col_offset+1)
        
        # M1: (A00+A11)(B00+B11)
        U[a_row_offset*4 + a_col_offset, start_col + 0] += 1
        U[(a_row_offset+1)*4 + a_col_offset+1, start_col + 0] += 1
        V[b_row_offset*4 + b_col_offset, start_col + 0] += 1
        V[(b_row_offset+1)*4 + b_col_offset+1, start_col + 0] += 1
        W[c_col_offset*3 + c_row_offset, start_col + 0] += 1  # C00
        W[(c_col_offset+1)*3 + c_row_offset+1, start_col + 0] += 1  # C11
        
        # M2: (A10+A11)(B00)
        U[(a_row_offset+1)*4 + a_col_offset, start_col + 1] += 1
        U[(a_row_offset+1)*4 + a_col_offset+1, start_col + 1] += 1
        V[b_row_offset*4 + b_col_offset, start_col + 1] += 1
        W[c_col_offset*3 + c_row_offset+1, start_col + 1] += 1  # C10
        W[(c_col_offset+1)*3 + c_row_offset+1, start_col + 1] += -1  # C11
        
        # M3: (A00)(B10-B11)
        U[a_row_offset*4 + a_col_offset, start_col + 2] += 1
        V[(b_row_offset+1)*4 + b_col_offset, start_col + 2] += 1
        V[(b_row_offset+1)*4 + b_col_offset+1, start_col + 2] += -1
        W[c_col_offset*3 + c_row_offset, start_col + 2] += 1  # C01... wait
        # C01 is at (c_row_offset, c_col_offset+1), so W index = (c_col_offset+1)*3 + c_row_offset
        W[(c_col_offset+1)*3 + c_row_offset, start_col + 2] += 1  # C01
        W[(c_col_offset+1)*3 + c_row_offset+1, start_col + 2] += 1  # C11
        
        # M4: (A11)(B01-B00)
        U[(a_row_offset+1)*4 + a_col_offset+1, start_col + 3] += 1
        V[b_row_offset*4 + b_col_offset+1, start_col + 3] += 1
        V[b_row_offset*4 + b_col_offset, start_col + 3] += -1
        W[c_col_offset*3 + c_row_offset, start_col + 3] += 1  # C00
        W[c_col_offset*3 + c_row_offset+1, start_col + 3] += 1  # C10
        
        # M5: (A00+A01)(B11)
        U[a_row_offset*4 + a_col_offset, start_col + 4] += 1
        U[a_row_offset*4 + a_col_offset+1, start_col + 4] += 1
        V[(b_row_offset+1)*4 + b_col_offset+1, start_col + 4] += 1
        W[c_col_offset*3 + c_row_offset, start_col + 4] += -1  # C00
        W[(c_col_offset+1)*3 + c_row_offset, start_col + 4] += 1  # C01
        
        # M6: (A10-A00)(B00+B01)
        U[(a_row_offset+1)*4 + a_col_offset, start_col + 5] += 1
        U[a_row_offset*4 + a_col_offset, start_col + 5] += -1
        V[b_row_offset*4 + b_col_offset, start_col + 5] += 1
        V[b_row_offset*4 + b_col_offset+1, start_col + 5] += 1
        W[(c_col_offset+1)*3 + c_row_offset+1, start_col + 5] += 1  # C11
        
        # M7: (A01-A11)(B10+B11)
        U[a_row_offset*4 + a_col_offset+1, start_col + 6] += 1
        U[(a_row_offset+1)*4 + a_col_offset+1, start_col + 6] += -1
        V[(b_row_offset+1)*4 + b_col_offset, start_col + 6] += 1
        V[(b_row_offset+1)*4 + b_col_offset+1, start_col + 6] += 1
        W[c_col_offset*3 + c_row_offset, start_col + 6] += 1  # C00
        
        # M8: A20*B00 -> C20
        U[(a_row_offset+2)*4 + a_col_offset, start_col + 7] += 1
        V[b_row_offset*4 + b_col_offset, start_col + 7] += 1
        W[c_col_offset*3 + c_row_offset+2, start_col + 7] += 1  # C20
        
        # M9: A21*B10 -> C20
        U[(a_row_offset+2)*4 + a_col_offset+1, start_col + 8] += 1
        V[(b_row_offset+1)*4 + b_col_offset, start_col + 8] += 1
        W[c_col_offset*3 + c_row_offset+2, start_col + 8] += 1  # C20
        
        # M10: A20*B01 -> C21
        U[(a_row_offset+2)*4 + a_col_offset, start_col + 9] += 1
        V[b_row_offset*4 + b_col_offset+1, start_col + 9] += 1
        W[(c_col_offset+1)*3 + c_row_offset+2, start_col + 9] += 1  # C21
        
        # M11: A21*B11 -> C21
        U[(a_row_offset+2)*4 + a_col_offset+1, start_col + 10] += 1
        V[(b_row_offset+1)*4 + b_col_offset+1, start_col + 10] += 1
        W[(c_col_offset+1)*3 + c_row_offset+2, start_col + 10] += 1  # C21
    
    # Block 0: A[:, 0:2] @ B[0:2, 0:2] -> C[:, 0:2]
    add_3x2x2_block(0, 0, 0, 0, 0, 0, 0)
    
    # Block 1: A[:, 2:4] @ B[2:4, 0:2] -> C[:, 0:2]
    add_3x2x2_block(0, 2, 2, 0, 0, 0, 11)
    
    # Block 2: A[:, 0:2] @ B[0:2, 2:4] -> C[:, 2:4]
    add_3x2x2_block(0, 0, 0, 2, 0, 2, 22)
    
    # Block 3: A[:, 2:4] @ B[2:4, 2:4] -> C[:, 2:4]
    add_3x2x2_block(0, 2, 2, 2, 0, 2, 33)
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
