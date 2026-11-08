import numpy as np
import time

# EVOLVE-BLOCK-START
_N, _M, _P = 4, 4, 4
_RANK = 49
_START_TIME = time.time()
_SAFETY_MARGIN = 10  # seconds before the 120s limit

def _construct_decomposition():
    """
    Build a rank-49 tensor decomposition for 4x4 matrix multiplication
    using a recursive Strassen-like approach.
    """
    U = np.zeros((_N * _M, _RANK), dtype=float)
    V = np.zeros((_M * _P, _RANK), dtype=float)
    W = np.zeros((_P * _N, _RANK), dtype=float)
    
    # Helper to get index in flattened 4x4 matrix
    def idx4(i, j):
        return i * 4 + j

    # Helper to get index in flattened 2x2 block (relative to block start)
    # But we are working with global indices.
    
    # Define Strassen patterns for 2x2 matrices.
    # Input matrices A, B (2x2). Output C (2x2).
    # A is indexed A[0,0], A[0,1], A[1,0], A[1,1]
    # We represent the 7 multiplication terms as linear combinations of elements.
    
    # Inner Strassen patterns
    # Each term m (0..6) is defined by:
    # x_coeffs: dict (r,c) -> coeff for X matrix (left operand)
    # y_coeffs: dict (r,c) -> coeff for Y matrix (right operand)
    # z_coeffs: dict (r,c) -> coeff for Z matrix (result)
    
    # Standard Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 - M2 + M4 + M6
    # C01 = M3 + M5
    # C10 = M2 + M7
    # C11 = M1 - M5 + M3 + M7
    
    # We define the patterns for the inner 2x2 multiplication.
    # x_patterns[m] is a list of (row, col, coeff) for the left matrix.
    # y_patterns[m] is a list of (row, col, coeff) for the right matrix.
    # z_patterns[m] is a list of (row, col, coeff) for the result matrix.
    
    x_patterns = [
        [(0,0,1), (1,1,1)],  # M1
        [(1,0,1), (1,1,1)],  # M2
        [(0,0,1)],           # M3
        [(1,1,1)],           # M4
        [(0,0,1), (0,1,1)],  # M5
        [(1,0,1), (0,0,-1)], # M6
        [(0,1,1), (1,1,-1)]  # M7
    ]
    
    y_patterns = [
        [(0,0,1), (1,1,1)],  # M1
        [(0,0,1)],           # M2
        [(0,1,1), (1,1,-1)], # M3
        [(1,0,1), (0,0,-1)], # M4
        [(1,1,1)],           # M5
        [(0,0,1), (0,1,1)],  # M6
        [(1,0,1), (1,1,1)]   # M7
    ]
    
    z_patterns = [
        [(0,0,1), (1,1,1)],  # M1 contributes to C00 and C11
        [(1,0,1), (1,1,-1)], # M2 contributes to C10 and C11
        [(0,1,1), (1,1,1)],  # M3 contributes to C01 and C11
        [(0,0,1), (1,0,1)],  # M4 contributes to C00 and C10
        [(0,0,-1), (0,1,1)], # M5 contributes to C00 (-1) and C01
        [(1,1,1)],           # M6 contributes to C11
        [(0,0,1)]            # M7 contributes to C00
    ]
    
    # Outer Strassen structure
    # We view the 4x4 matrices as 2x2 blocks of 2x2 matrices.
    # Block indices: (0,0), (0,1), (1,0), (1,1)
    # Global row i = block_row * 2 + local_row
    # Global col j = block_col * 2 + local_col
    
    # Outer patterns:
    # For each outer term k (0..6), define which blocks of A and B are involved
    # and with what coefficients.
    # A_blocks[k]: list of (block_r, block_c, coeff)
    # B_blocks[k]: list of (block_r, block_c, coeff)
    # C_blocks[k]: list of (block_r, block_c, coeff)
    
    A_blocks = [
        [(0,0,1), (1,1,1)],  # M1: A00 + A11
        [(1,0,1), (1,1,1)],  # M2: A10 + A11
        [(0,0,1)],           # M3: A00
        [(1,1,1)],           # M4: A11
        [(0,0,1), (0,1,1)],  # M5: A00 + A01
        [(1,0,1), (0,0,-1)], # M6: A10 - A00
        [(0,1,1), (1,1,-1)]  # M7: A01 - A11
    ]
    
    B_blocks = [
        [(0,0,1), (1,1,1)],  # M1: B00 + B11
        [(0,0,1)],           # M2: B00
        [(0,1,1), (1,1,-1)], # M3: B01 - B11
        [(1,0,1), (0,0,-1)], # M4: B10 - B00
        [(1,1,1)],           # M5: B11
        [(0,0,1), (0,1,1)],  # M6: B00 + B01
        [(1,0,1), (1,1,1)]   # M7: B10 + B11
    ]
    
    C_blocks = [
        [(0,0,1), (1,1,1)],  # M1 -> C00, C11
        [(1,0,1), (1,1,-1)], # M2 -> C10, C11
        [(0,1,1), (1,1,1)],  # M3 -> C01, C11
        [(0,0,1), (1,0,1)],  # M4 -> C00, C10
        [(0,0,-1), (0,1,1)], # M5 -> C00, C01
        [(1,1,1)],           # M6 -> C11
        [(0,0,1)]            # M7 -> C00
    ]
    
    # Construct the decomposition
    for k in range(7):
        # Get the outer block definitions for term k
        a_blk_defs = A_blocks[k]
        b_blk_defs = B_blocks[k]
        c_blk_defs = C_blocks[k]
        
        for p in range(7):
            rank_idx = k * 7 + p
            
            # For this rank-1 term, we combine the outer block selection
            # with the inner Strassen pattern p.
            
            # U coefficients
            # For each block (br, bc) in a_blk_defs with coeff ca:
            #   For each local element (lr, lc) in x_patterns[p] with coeff cx:
            #     Global A index: (br*2+lr, bc*2+lc)
            #     Coeff: ca * cx
            for br, bc, ca in a_blk_defs:
                for lr, lc, cx in x_patterns[p]:
                    gi = br * 2 + lr
                    gj = bc * 2 + lc
                    U[idx4(gi, gj), rank_idx] += ca * cx
            
            # V coefficients
            # For each block (br, bc) in b_blk_defs with coeff cb:
            #   For each local element (lr, lc) in y_patterns[p] with coeff cy:
            #     Global B index: (br*2+lr, bc*2+lc)
            #     Coeff: cb * cy
            for br, bc, cb in b_blk_defs:
                for lr, lc, cy in y_patterns[p]:
                    gi = br * 2 + lr
                    gj = bc * 2 + lc
                    V[idx4(gi, gj), rank_idx] += cb * cy
            
            # W coefficients
            # For each block (br, bc) in c_blk_defs with coeff cc:
            #   For each local element (lr, lc) in z_patterns[p] with coeff cz:
            #     Global C index: (br*2+lr, bc*2+lc)
            #     Coeff: cc * cz
            for br, bc, cc in c_blk_defs:
                for lr, lc, cz in z_patterns[p]:
                    gi = br * 2 + lr
                    gj = bc * 2 + lc
                    W[idx4(gi, gj), rank_idx] += cc * cz
    
    return {"U": U, "V": V, "W": W}

# Precompute at module load time
_RESULT = _construct_decomposition()

def construct():
    """
    Return the rank-49 tensor decomposition for 4x4 matrix multiplication.
    
    Returns:
        dict with keys 'U', 'V', 'W' as numpy arrays of shape (16, 49)
    """
    return _RESULT
# EVOLVE-BLOCK-END
