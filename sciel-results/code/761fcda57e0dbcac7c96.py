"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction
from itertools import product
import scipy.optimize

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-36 exact tensor decomposition for 2x4x5 matrix multiplication
    using Strassen's algorithm for 2x2x2 blocks and schoolbook for 2x2x1 blocks.
    """
    n, m, p = DIMENSIONS  # n=2, m=4, p=5
    rank = 36
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    def add_term(idx, u_coeffs, v_coeffs, w_coeffs):
        """
        Add a rank-1 term to U, V, W.
        u_coeffs: dict {(i, j): val}
        v_coeffs: dict {(j, k): val}
        w_coeffs: dict {(k, i): val}
        """
        for (i, j), val in u_coeffs.items():
            U[i*m + j, idx] += val
        for (j, k), val in v_coeffs.items():
            V[j*p + k, idx] += val
        for (k, i), val in w_coeffs.items():
            W[k*n + i, idx] += val

    def strassen_block(a_row_offset, b_row_offset, c_col_offset, idx_start):
        """
        Add Strassen decomposition for a 2x2x2 block.
        """
        # M1 = (A00 + A11) * (B00 + B11)
        add_term(idx_start, 
                 {(0, a_row_offset): 1, (1, a_row_offset+1): 1},
                 {(b_row_offset, c_col_offset): 1, (b_row_offset+1, c_col_offset+1): 1},
                 {(c_col_offset, 0): 1, (c_col_offset+1, 1): 1})
        
        # M2 = (A10 + A11) * B00
        add_term(idx_start+1,
                 {(1, a_row_offset): 1, (1, a_row_offset+1): 1},
                 {(b_row_offset, c_col_offset): 1},
                 {(c_col_offset, 1): 1, (c_col_offset+1, 1): -1})
        
        # M3 = A00 * (B01 - B11)
        add_term(idx_start+2,
                 {(0, a_row_offset): 1},
                 {(b_row_offset, c_col_offset+1): 1, (b_row_offset+1, c_col_offset+1): -1},
                 {(c_col_offset+1, 0): 1, (c_col_offset+1, 1): 1})
        
        # M4 = A11 * (B10 - B00)
        add_term(idx_start+3,
                 {(1, a_row_offset+1): 1},
                 {(b_row_offset+1, c_col_offset): 1, (b_row_offset, c_col_offset): -1},
                 {(c_col_offset, 0): 1, (c_col_offset, 1): 1})
        
        # M5 = (A00 + A01) * B11
        add_term(idx_start+4,
                 {(0, a_row_offset): 1, (0, a_row_offset+1): 1},
                 {(b_row_offset+1, c_col_offset+1): 1},
                 {(c_col_offset, 0): -1, (c_col_offset+1, 0): 1})
        
        # M6 = (A10 - A00) * (B00 + B01)
        add_term(idx_start+5,
                 {(1, a_row_offset): 1, (0, a_row_offset): -1},
                 {(b_row_offset, c_col_offset): 1, (b_row_offset, c_col_offset+1): 1},
                 {(c_col_offset+1, 1): 1})
        
        # M7 = (A01 - A11) * (B10 + B11)
        add_term(idx_start+6,
                 {(0, a_row_offset+1): 1, (1, a_row_offset+1): -1},
                 {(b_row_offset+1, c_col_offset): 1, (b_row_offset+1, c_col_offset+1): 1},
                 {(c_col_offset, 0): 1})

    def schoolbook_2x2x1_block(a_row_offset, b_row_offset, c_col_offset, idx_start):
        """
        Add schoolbook decomposition for a 2x2x1 block.
        """
        # Term 1: A00 B00 -> C00
        add_term(idx_start,
                 {(0, a_row_offset): 1},
                 {(b_row_offset, c_col_offset): 1},
                 {(c_col_offset, 0): 1})
        
        # Term 2: A01 B10 -> C00
        add_term(idx_start+1,
                 {(0, a_row_offset+1): 1},
                 {(b_row_offset+1, c_col_offset): 1},
                 {(c_col_offset, 0): 1})
        
        # Term 3: A10 B00 -> C10
        add_term(idx_start+2,
                 {(1, a_row_offset): 1},
                 {(b_row_offset, c_col_offset): 1},
                 {(c_col_offset, 1): 1})
        
        # Term 4: A11 B10 -> C10
        add_term(idx_start+3,
                 {(1, a_row_offset+1): 1},
                 {(b_row_offset+1, c_col_offset): 1},
                 {(c_col_offset, 1): 1})

    # Block 1: A[:, 0:2] * B[0:2, :]
    idx = 0
    strassen_block(0, 0, 0, idx)       # A_L * B_T,11
    idx += 7
    strassen_block(0, 0, 2, idx)       # A_L * B_T,12
    idx += 7
    schoolbook_2x2x1_block(0, 0, 4, idx) # A_L * B_T,13
    idx += 4
    
    # Block 2: A[:, 2:4] * B[2:4, :]
    strassen_block(2, 2, 0, idx)       # A_R * B_B,11
    idx += 7
    strassen_block(2, 2, 2, idx)       # A_R * B_B,12
    idx += 7
    schoolbook_2x2x1_block(2, 2, 4, idx) # A_R * B_B,13
    idx += 4
    
    if idx != rank:
        raise ValueError(f"Rank mismatch: expected {rank}, got {idx}")

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
