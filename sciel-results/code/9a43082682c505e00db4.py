"""Tensor decomposition for 2x4 * 4x6 matrix multiplication."""
import numpy as np

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=6
    
    # We decompose M(2, 4, 6) into two M(2, 2, 6) sub-tensors.
    # Each M(2, 2, 6) is decomposed into three M(2, 2, 2) sub-tensors using Strassen.
    # Total rank = 2 * 3 * 7 = 42.
    
    rank = 42
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    # Helper to add a 2x2x2 Strassen block
    # a_offset: row offset in A (0 or 2)
    # b_offset: row offset in B (0 or 2)
    # c_offset: col offset in C (0, 2, 4)
    def add_strassen_block(a_off, b_off, c_off):
        nonlocal col
        # Indices in U: (i, j) -> i*m + j. A is 2x4.
        # a_off is 0 or 2.
        # j_local is 0 or 1. j_global = a_off + j_local.
        
        # Indices in V: (j, k) -> j*p + k. B is 4x6.
        # b_off is 0 or 2.
        # j_local is 0 or 1. j_global = b_off + j_local.
        # k_local is 0 or 1. k_global = c_off + k_local.
        
        # Indices in W: (k, i) -> k*n + i. C is 2x6.
        # c_off is 0, 2, 4.
        # k_local is 0 or 1. k_global = c_off + k_local.
        # i is 0 or 1.
        
        # Strassen M1 = (a00 + a11) * (b00 + b11)
        U[0*m + (a_off+0), col] += 1
        U[1*m + (a_off+1), col] += 1
        V[(b_off+0)*p + (c_off+0), col] += 1
        V[(b_off+1)*p + (c_off+1), col] += 1
        W[(c_off+0)*n + 0, col] += 1
        W[(c_off+1)*n + 1, col] += 1
        col += 1
        
        # M2 = (a10 + a11) * b00
        U[1*m + (a_off+0), col] += 1
        U[1*m + (a_off+1), col] += 1
        V[(b_off+0)*p + (c_off+0), col] += 1
        W[(c_off+0)*n + 1, col] += 1
        W[(c_off+1)*n + 1, col] -= 1
        col += 1
        
        # M3 = a00 * (b01 - b11)
        U[0*m + (a_off+0), col] += 1
        V[(b_off+0)*p + (c_off+1), col] += 1
        V[(b_off+1)*p + (c_off+1), col] -= 1
        W[(c_off+1)*n + 0, col] += 1
        W[(c_off+1)*n + 1, col] += 1
        col += 1
        
        # M4 = a11 * (b10 - b00)
        U[1*m + (a_off+1), col] += 1
        V[(b_off+1)*p + (c_off+0), col] += 1
        V[(b_off+0)*p + (c_off+0), col] -= 1
        W[(c_off+0)*n + 0, col] += 1
        W[(c_off+0)*n + 1, col] += 1
        col += 1
        
        # M5 = (a00 + a01) * b11
        U[0*m + (a_off+0), col] += 1
        U[0*m + (a_off+1), col] += 1
        V[(b_off+1)*p + (c_off+1), col] += 1
        W[(c_off+0)*n + 0, col] -= 1
        W[(c_off+1)*n + 0, col] += 1
        col += 1
        
        # M6 = (a10 - a00) * (b00 + b01)
        U[1*m + (a_off+0), col] += 1
        U[0*m + (a_off+0), col] -= 1
        V[(b_off+0)*p + (c_off+0), col] += 1
        V[(b_off+0)*p + (c_off+1), col] += 1
        W[(c_off+1)*n + 1, col] += 1
        col += 1
        
        # M7 = (a01 - a11) * (b10 + b11)
        U[0*m + (a_off+1), col] += 1
        U[1*m + (a_off+1), col] -= 1
        V[(b_off+1)*p + (c_off+0), col] += 1
        V[(b_off+1)*p + (c_off+1), col] += 1
        W[(c_off+0)*n + 0, col] += 1
        col += 1

    # Process A_left (rows 0-1 of A, which are cols 0-1) * B_top (rows 0-1 of B)
    # A_left is A[:, 0:2]. In U, this corresponds to indices 0,1 (j=0,1).
    # B_top is B[0:2, :]. In V, this corresponds to rows 0,1 (j=0,1).
    # C contributions are to all 6 columns of C.
    # We split C's 6 columns into 3 blocks of 2.
    
    for c_block in range(3):
        c_off = c_block * 2
        add_strassen_block(0, 0, c_off)
        
    # Process A_right (rows 0-1 of A, which are cols 2-3) * B_bottom (rows 2-3 of B)
    # A_right is A[:, 2:4]. In U, this corresponds to indices 2,3 (j=2,3).
    # B_bottom is B[2:4, :]. In V, this corresponds to rows 2,3 (j=2,3).
    
    for c_block in range(3):
        c_off = c_block * 2
        add_strassen_block(2, 2, c_off)

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
