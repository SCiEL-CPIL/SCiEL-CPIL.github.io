"""
Rank-189 exact tensor decomposition for 6x6 matrix multiplication via 3x3 tiling of 2x2 Strassen blocks.
"""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s limit
    
    # We use the tiling approach: split 6x6 into 3x3 blocks of 2x2.
    # Each pair of 2x2 blocks (A_ij, B_jk) -> C_ik is handled by rank-7 Strassen.
    # There are 3 (i) * 3 (j) * 3 (k) = 27 pairs.
    # Each pair uses 7 scalar multiplications (rank 7 terms).
    # Total rank = 27 * 7 = 189.
    
    rank = 189
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen 2x2 decomposition terms.
    # Local indices (r, c) in {0, 1} x {0, 1}
    # U terms act on A, V terms act on B, W terms act on C.
    # Coefficients are integers.
    # The 7 terms are defined as (u_list, v_list, w_list)
    # where u_list is a list of (r, c, coeff), etc.
    
    strassen_terms = [
        # M1 = (A00+A11)(B00+B11)
        (
            [(0, 0, 1), (1, 1, 1)],
            [(0, 0, 1), (1, 1, 1)],
            [(0, 0, 1), (1, 1, 1)]
        ),
        # M2 = (A10+A11)B00
        (
            [(1, 0, 1), (1, 1, 1)],
            [(0, 0, 1)],
            [(1, 0, 1), (1, 1, -1)]
        ),
        # M3 = A00(B01-B11)
        (
            [(0, 0, 1)],
            [(0, 1, 1), (1, 1, -1)],
            [(0, 1, 1), (1, 1, 1)]
        ),
        # M4 = A11(B10-B00)
        (
            [(1, 1, 1)],
            [(1, 0, 1), (0, 0, -1)],
            [(0, 0, 1), (1, 0, 1)]
        ),
        # M5 = (A00+A01)B11
        (
            [(0, 0, 1), (0, 1, 1)],
            [(1, 1, 1)],
            [(0, 0, -1), (0, 1, 1)]
        ),
        # M6 = (A10-A00)(B00+B01)
        (
            [(1, 0, 1), (0, 0, -1)],
            [(0, 0, 1), (0, 1, 1)],
            [(1, 1, 1)]
        ),
        # M7 = (A01-A11)(B10+B11)
        (
            [(0, 1, 1), (1, 1, -1)],
            [(1, 0, 1), (1, 1, 1)],
            [(0, 0, 1)]
        )
    ]
    
    # Iterate over all 27 block pairs
    # i: row block of A (0..2)
    # j: col block of A / row block of B (0..2)
    # k: col block of B (0..2)
    
    for i in range(3):
        for j in range(3):
            for k in range(3):
                # Block positions
                # A block is at rows 2*i..2*i+1, cols 2*j..2*j+1
                a_r0 = 2 * i
                a_c0 = 2 * j
                
                # B block is at rows 2*j..2*j+1, cols 2*k..2*k+1
                b_r0 = 2 * j
                b_c0 = 2 * k
                
                # C block is at rows 2*i..2*i+1, cols 2*k..2*k+1
                c_r0 = 2 * i
                c_c0 = 2 * k
                
                # Add 7 Strassen terms
                for t in range(7):
                    col_idx = (i * 9 + j * 3 + k) * 7 + t
                    u_terms, v_terms, w_terms = strassen_terms[t]
                    
                    for r, c, coeff in u_terms:
                        row = a_r0 + r
                        col = a_c0 + c
                        U[row * m + col, col_idx] += coeff
                    
                    for r, c, coeff in v_terms:
                        row = b_r0 + r
                        col = b_c0 + c
                        V[row * p + col, col_idx] += coeff
                    
                    for r, c, coeff in w_terms:
                        row = c_r0 + r
                        col = c_c0 + c
                        # W index convention: W[k*n+i, r] where k is col of C, i is row of C
                        # So index is col * n + row
                        W[col * n + row, col_idx] += coeff
    
    # Convert to lists to match the expected interface if needed, 
    # but numpy arrays are often accepted or convertible. 
    # The task says "factor matrices of shapes... with integer/half-integer entries".
    # Returning numpy arrays is usually fine, but let's return lists of lists to be safe 
    # if the evaluator expects native Python types, or just arrays if it's flexible.
    # Given the seed used lists, let's convert to lists.
    
    elapsed = time.time() - start_time
    if elapsed > deadline:
        # Should not happen, but return what we have
        pass
    
    U = U.tolist()
    V = V.tolist()
    W = W.tolist()
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
