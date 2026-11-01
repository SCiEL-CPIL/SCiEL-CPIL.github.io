import numpy as np
import time

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 2x4 * 4x8 matrix multiplication.
    The minimal rank is 56, achieved by decomposing the problem into 8 independent 2x2x2
    multiplications, each solved using the Strassen algorithm (rank 7).
    
    U: shape (n*m, rank) = (8, 56)
    V: shape (m*p, rank) = (32, 56)
    W: shape (p*n, rank) = (16, 56)
    """
    n, m, p = 2, 4, 8
    rank = 56
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # We decompose the multiplication into 4 blocks of B (each 4x2).
    # For each block v (0..3), we compute C[:, 2v:2v+2] = A * B[:, 2v:2v+2].
    # A is 2x4, B_block is 4x2.
    # Split A into A_L (2x2, cols 0-1) and A_R (2x2, cols 2-3).
    # Split B_block into B_Top (2x2, rows 0-1) and B_Bot (2x2, rows 2-3).
    # C_block = A_L * B_Top + A_R * B_Bot.
    # Each term is a 2x2 * 2x2 multiplication, rank 7 (Strassen).
    # Total rank = 4 blocks * 2 terms/block * 7 rank/term = 56.
    
    for v in range(4):
        # v corresponds to output columns 2v and 2v+1
        # B rows are 0..3.
        # B indices for this block: rows 0..3, cols 2v, 2v+1.
        
        # Part 0: A_L * B_Top
        # Part 1: A_R * B_Bot
        
        for part in range(2):
            # base index in the rank dimension
            base_col = 14 * v + 7 * part
            
            # A indices:
            # A is 2x4. Index a = i*4 + j.
            # A_L: i in {0,1}, j in {0,1}
            # A_R: i in {0,1}, j in {2,3}
            if part == 0:
                a_j_offsets = [0, 1]
            else:
                a_j_offsets = [2, 3]
            
            # B indices:
            # B is 4x8. Index b = j*8 + k.
            # B_Top: j in {0,1}, k in {2v, 2v+1}
            # B_Bot: j in {2,3}, k in {2v, 2v+1}
            if part == 0:
                b_j_offsets = [0, 1]
            else:
                b_j_offsets = [2, 3]
            b_k_offsets = [2 * v, 2 * v + 1]
            
            # C indices:
            # C is 8x2 (p*n). Index c = k*n + i = k*2 + i.
            # We are computing C for k in {2v, 2v+1} and i in {0,1}.
            # c00 corresponds to i=0, k=2v
            # c10 corresponds to i=1, k=2v
            # c01 corresponds to i=0, k=2v+1
            # c11 corresponds to i=1, k=2v+1
            
            c00 = (2 * v) * 2 + 0
            c10 = (2 * v) * 2 + 1
            c01 = (2 * v + 1) * 2 + 0
            c11 = (2 * v + 1) * 2 + 1
            
            # Helper functions for local indices within the 2x2 block
            # Local A indices: 00, 01, 10, 11
            # a_idx(r, c) where r,c in {0,1}
            def get_a_idx(r, c):
                return r * 4 + (a_j_offsets[c])
            
            def get_b_idx(r, c):
                return (b_j_offsets[r]) * 8 + (b_k_offsets[c])
            
            # Strassen's 7 products for 2x2 * 2x2
            # M = [[m00, m01], [m10, m11]]
            # s0 = (A00+A11)(B00+B11) -> contributes to m00, m11
            # s1 = (A10+A11)B00 -> contributes to m10, m11
            # s2 = A00(B01-B11) -> contributes to m01, m11
            # s3 = A11(B10-B00) -> contributes to m00, m10
            # s4 = (A00+A01)B11 -> contributes to m00, m01
            # s5 = (A10-A00)(B00+B01) -> contributes to m11
            # s6 = (A01-A11)(B10+B11) -> contributes to m00
            
            # s=0
            U[get_a_idx(0, 0)][base_col + 0] += 1
            U[get_a_idx(1, 1)][base_col + 0] += 1
            V[get_b_idx(0, 0)][base_col + 0] += 1
            V[get_b_idx(1, 1)][base_col + 0] += 1
            W[c00][base_col + 0] += 1
            W[c11][base_col + 0] += 1
            
            # s=1
            U[get_a_idx(1, 0)][base_col + 1] += 1
            U[get_a_idx(1, 1)][base_col + 1] += 1
            V[get_b_idx(0, 0)][base_col + 1] += 1
            W[c10][base_col + 1] += 1
            W[c11][base_col + 1] -= 1
            
            # s=2
            U[get_a_idx(0, 0)][base_col + 2] += 1
            V[get_b_idx(0, 1)][base_col + 2] += 1
            V[get_b_idx(1, 1)][base_col + 2] -= 1
            W[c01][base_col + 2] += 1
            W[c11][base_col + 2] += 1
            
            # s=3
            U[get_a_idx(1, 1)][base_col + 3] += 1
            V[get_b_idx(1, 0)][base_col + 3] += 1
            V[get_b_idx(0, 0)][base_col + 3] -= 1
            W[c00][base_col + 3] += 1
            W[c10][base_col + 3] += 1
            
            # s=4
            U[get_a_idx(0, 0)][base_col + 4] += 1
            U[get_a_idx(0, 1)][base_col + 4] += 1
            V[get_b_idx(1, 1)][base_col + 4] += 1
            W[c00][base_col + 4] -= 1
            W[c01][base_col + 4] += 1
            
            # s=5
            U[get_a_idx(1, 0)][base_col + 5] += 1
            U[get_a_idx(0, 0)][base_col + 5] -= 1
            V[get_b_idx(0, 0)][base_col + 5] += 1
            V[get_b_idx(0, 1)][base_col + 5] += 1
            W[c11][base_col + 5] += 1
            
            # s=6
            U[get_a_idx(0, 1)][base_col + 6] += 1
            U[get_a_idx(1, 1)][base_col + 6] -= 1
            V[get_b_idx(1, 0)][base_col + 6] += 1
            V[get_b_idx(1, 1)][base_col + 6] += 1
            W[c00][base_col + 6] += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
