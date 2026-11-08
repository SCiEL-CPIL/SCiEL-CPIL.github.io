"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Block decomposition: 6x6 matrices split into 3x3 blocks of 2x2.
    # Each 2x2 block multiplication uses Strassen (rank 7).
    # Total blocks: 3 * 3 * 3 = 27.
    # Total rank: 27 * 7 = 189.
    
    bs = 2  # block size
    nb = 3  # number of blocks per dimension
    
    total_rank = (nb ** 3) * 7
    
    # Initialize factor matrices with zeros
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Define Strassen terms for 2x2 matrix multiplication.
    # Each term is (A_indices, B_indices, C_indices)
    # Indices are (row, col) within the 2x2 block.
    # Values are implicitly 1 unless specified, but here we use explicit coefficients for clarity.
    # Strassen terms:
    # M0 = (A00+A11)(B00+B11) -> C00 += M0, C11 += M0
    # M1 = (A10+A11)B00 -> C10 += M1, C11 -= M1
    # M2 = A00(B01-B11) -> C01 += M2, C11 += M2
    # M3 = A11(B10-B00) -> C00 += M3, C10 += M3
    # M4 = (A00+A01)B11 -> C00 -= M4, C01 += M4
    # M5 = (A10-A00)(B00+B01) -> C11 += M5
    # M6 = (A01-A11)(B10+B11) -> C00 += M6
    
    strassen_terms = [
        # (A_coeffs, B_coeffs, C_coeffs)
        # Coeffs are lists of (r, c, val)
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
        ([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1), (1,1,-1)]),
        ([(0,0,1)], [(0,1,1), (1,1,-1)], [(0,1,1), (1,1,1)]),
        ([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (1,0,1)]),
        ([(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,-1), (0,1,1)]),
        ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)]),
        ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1)]),
    ]
    
    col = 0
    for bi in range(nb):
        for bj in range(nb):
            for bk in range(nb):
                for a_terms, b_terms, c_terms in strassen_terms:
                    # Update U (Input A)
                    for r, c, val in a_terms:
                        row_a = bi * bs + r
                        col_a = bj * bs + c
                        U[row_a * m + col_a][col] += val
                    
                    # Update V (Input B)
                    for r, c, val in b_terms:
                        row_b = bj * bs + r
                        col_b = bk * bs + c
                        V[row_b * p + col_b][col] += val
                    
                    # Update W (Output C)
                    # Output C is indexed by [k*n + i] where k is row of C (from B's col) and i is col of C (from A's row)?
                    # Wait, standard matrix multiplication C = A * B.
                    # C[i, k] = sum_j A[i, j] B[j, k].
                    # In the problem statement: W[k*n+i, r].
                    # This implies the output index is (k, i) where k is the row of B (which becomes col of A? No).
                    # Let's trace indices carefully.
                    # A is n x m. B is m x p. C is n x p.
                    # C[i, k] = sum_j A[i, j] B[j, k].
                    # U has shape (n*m, rank). Index (i*m + j).
                    # V has shape (m*p, rank). Index (j*p + k).
                    # W has shape (p*n, rank). Index (k*n + i).
                    # So W index is k*n + i.
                    
                    # In our block loop:
                    # bi corresponds to block row of A (and C). So i = bi*bs + r_a.
                    # bj corresponds to block col of A (and row of B). So j = bj*bs + c_a.
                    # bk corresponds to block col of B (and C). So k = bk*bs + c_b.
                    
                    # In the Strassen term:
                    # a_terms gives (r_a, c_a).
                    # b_terms gives (r_b, c_b). Note r_b should match c_a in block space?
                    # Yes, the inner dimension j is shared.
                    # In our block structure:
                    # A block is at (bi, bj). Size 2x2.
                    # B block is at (bj, bk). Size 2x2.
                    # C block is at (bi, bk). Size 2x2.
                    
                    # The Strassen terms operate on local 2x2 coordinates.
                    # For A: local (r, c) maps to global (bi*2+r, bj*2+c).
                    # For B: local (r, c) maps to global (bj*2+r, bk*2+c).
                    # For C: local (r, c) maps to global (bi*2+r, bk*2+c).
                    
                    # Wait, the W index is k*n + i.
                    # i = bi*2 + r_c.
                    # k = bk*2 + c_c.
                    
                    for r_c, c_c, val in c_terms:
                        i_global = bi * bs + r_c
                        k_global = bk * bs + c_c
                        W[k_global * n + i_global][col] += val
                        
                    col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
