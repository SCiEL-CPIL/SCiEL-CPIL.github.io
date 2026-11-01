"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    # Dimensions
    n, m, p = 2, 4, 4
    
    # Rank is 28 (4 blocks of Strassen rank 7)
    rank_28 = 28
    U = np.zeros((n * m, rank_28), dtype=np.float64)
    V = np.zeros((m * p, rank_28), dtype=np.float64)
    W = np.zeros((p * n, rank_28), dtype=np.float64)
    
    # Strassen decomposition for 2x2 matrix multiplication
    # Inputs: A (2x2), B (2x2) -> C (2x2)
    # Indices in U: A elements
    # Indices in V: B elements
    # Indices in W: C elements
    # 
    # M1 = (A00 + A11) * (B00 + B11) -> C00, C11
    # M2 = (A10 + A11) * B00          -> C10, C11
    # M3 = A00 * (B01 - B11)          -> C01, C11
    # M4 = A11 * (B10 - B00)          -> C00, C10
    # M5 = (A00 + A01) * B11          -> C00, C01
    # M6 = (A10 - A00) * (B00 + B01)  -> C11
    # M7 = (A11 - A01) * (B10 + B11)  -> C00
    
    def add_strassen_block(offset, a_row_base, a_col_base, b_row_base, b_col_base, c_row_base, c_col_base):
        # A indices in U (flattened 2x4 matrix A, row i, col k)
        # A is 2x4. a_row_base is the row index in the 2x4 block (0 or 1).
        # a_col_base is the column index in the 2x4 block (0, 1, 2, or 3).
        # The 2x2 sub-block for Strassen uses columns a_col_base and a_col_base+1.
        
        a00_u = a_row_base * m + a_col_base
        a01_u = a_row_base * m + a_col_base + 1
        a10_u = (a_row_base + 1) * m + a_col_base
        a11_u = (a_row_base + 1) * m + a_col_base + 1
        
        # B indices in V (flattened 4x4 matrix B, row k, col j)
        # b_row_base is the row index in the 4x4 block (0, 1, 2, or 3).
        # b_col_base is the column index in the 4x4 block (0, 1, 2, or 3).
        # The 2x2 sub-block for Strassen uses rows b_row_base and b_row_base+1, 
        # and columns b_col_base and b_col_base+1.
        
        b00_v = b_row_base * p + b_col_base
        b01_v = b_row_base * p + b_col_base + 1
        b10_v = (b_row_base + 1) * p + b_col_base
        b11_v = (b_row_base + 1) * p + b_col_base + 1
        
        # C indices in W (flattened 2x4 matrix C, row i, col j)
        # Convention: W[k*n+i, r] ? 
        # Task says: "output factor follows the official trace convention W[k*n+i,r]"
        # Wait, the tensor is usually T[a,b,c]. U is index a, V is index b, W is index c.
        # The decomposition is sum_r U[a,r] V[b,r] W[c,r] = T[a,b,c].
        # The index c corresponds to the output matrix C.
        # C is n x p (2x4). 
        # The problem statement says W has shape (p*n, rank).
        # And "W[k*n+i,r]". This usually implies index c = k*n + i.
        # In standard matrix mult C = AB, C[i,j] = sum_k A[i,k] B[k,j].
        # Here n=2, m=4, p=4.
        # A is 2x4. B is 4x4. C is 2x4.
        # Index a in U corresponds to A[i,k] -> i*m + k. (n*m = 8).
        # Index b in V corresponds to B[k,j] -> k*p + j. (m*p = 16).
        # Index c in W corresponds to C[i,j].
        # The convention W[k*n+i,r] is a bit unusual if c is just the index in the vector.
        # Let's check the seed code.
        # Seed code: c_idx = j * n + i.
        # So c = j*n + i. This matches "W[j*n+i, r]" if we rename j to k in the prompt's notation?
        # Prompt: "W[k*n+i,r]". If i is row (0..n-1) and k is col (0..p-1), then index is k*n+i.
        # So c_index = col * n + row.
        
        c00_w = c_col_base * n + c_row_base
        c01_w = (c_col_base + 1) * n + c_row_base
        c10_w = c_col_base * n + (c_row_base + 1)
        c11_w = (c_col_base + 1) * n + (c_row_base + 1)
        
        # M1
        U[a00_u, offset + 0] += 1
        U[a11_u, offset + 0] += 1
        V[b00_v, offset + 0] += 1
        V[b11_v, offset + 0] += 1
        W[c00_w, offset + 0] += 1
        W[c11_w, offset + 0] += 1
        
        # M2
        U[a10_u, offset + 1] += 1
        U[a11_u, offset + 1] += 1
        V[b00_v, offset + 1] += 1
        W[c10_w, offset + 1] += 1
        W[c11_w, offset + 1] -= 1
        
        # M3
        U[a00_u, offset + 2] += 1
        V[b01_v, offset + 2] += 1
        V[b11_v, offset + 2] -= 1
        W[c01_w, offset + 2] += 1
        W[c11_w, offset + 2] += 1
        
        # M4
        U[a11_u, offset + 3] += 1
        V[b10_v, offset + 3] += 1
        V[b00_v, offset + 3] -= 1
        W[c00_w, offset + 3] += 1
        W[c10_w, offset + 3] += 1
        
        # M5
        U[a00_u, offset + 4] += 1
        U[a01_u, offset + 4] += 1
        V[b11_v, offset + 4] += 1
        W[c00_w, offset + 4] -= 1
        W[c01_w, offset + 4] += 1
        
        # M6
        U[a10_u, offset + 5] += 1
        U[a00_u, offset + 5] -= 1
        V[b00_v, offset + 5] += 1
        V[b01_v, offset + 5] += 1
        W[c11_w, offset + 5] += 1
        
        # M7
        U[a01_u, offset + 6] += 1
        U[a11_u, offset + 6] -= 1
        V[b10_v, offset + 6] += 1
        V[b11_v, offset + 6] += 1
        W[c00_w, offset + 6] += 1

    # We need to cover the entire 2x4 * 4x4 multiplication.
    # A (2x4) * B (4x4) -> C (2x4).
    # C[i,j] = sum_k A[i,k] B[k,j].
    # We can decompose this into 4 independent 2x2x2 Strassen blocks.
    # The "inner" dimension k is 4. We split it into two pairs: (0,1) and (2,3).
    # For each pair of k, we have a 2x2 block of A (rows 0,1; cols k1,k2) and a 2x4 block of B (rows k1,k2; cols 0..3).
    # Then we split the 4 columns of B into two pairs: (0,1) and (2,3).
    # So we have 4 combinations of (k_pair, j_pair).
    
    # Block 1: k in {0,1}, j in {0,1}
    # a_row_base=0, a_col_base=0
    # b_row_base=0, b_col_base=0
    # c_row_base=0, c_col_base=0
    add_strassen_block(0, 0, 0, 0, 0, 0, 0)
    
    # Block 2: k in {0,1}, j in {2,3}
    # a_row_base=0, a_col_base=0
    # b_row_base=0, b_col_base=2
    # c_row_base=0, c_col_base=2
    add_strassen_block(7, 0, 0, 0, 2, 0, 2)
    
    # Block 3: k in {2,3}, j in {0,1}
    # a_row_base=0, a_col_base=2
    # b_row_base=2, b_col_base=0
    # c_row_base=0, c_col_base=0
    add_strassen_block(14, 0, 2, 2, 0, 0, 0)
    
    # Block 4: k in {2,3}, j in {2,3}
    # a_row_base=0, a_col_base=2
    # b_row_base=2, b_col_base=2
    # c_row_base=0, c_col_base=2
    add_strassen_block(21, 0, 2, 2, 2, 0, 2)

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
