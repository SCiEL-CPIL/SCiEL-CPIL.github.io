"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Approach: 3x3 grid of 2x2 blocks, each using Strassen (rank 7).
    # Total rank = 27 * 7 = 189.
    
    block_size = 2
    blocks_per_dim = n // block_size # 3
    
    rank = (blocks_per_dim ** 3) * 7
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Define Strassen terms for 2x2 multiplication.
    # Each term is (u_dict, v_dict, w_dict)
    # u_dict: {(row, col): coeff} for A
    # v_dict: {(row, col): coeff} for B
    # w_dict: {(row, col): coeff} for C
    # Note: C[k, i] = sum A[i, j] B[j, k]
    # In the tensor decomposition, W is indexed by (k, i) -> k*n + i
    # U is indexed by (i, j) -> i*m + j
    # V is indexed by (j, k) -> j*p + k
    
    strassen_terms = [
        # M0 = (A00+A11)(B00+B11) -> C00 += M0, C11 += M0
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # M1 = (A10+A11)B00 -> C10 += M1, C11 -= M1
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        # M2 = A00(B01-B11) -> C01 += M2, C11 += M2
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        # M3 = A11(B10-B00) -> C00 += M3, C10 += M3
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        # M4 = (A00+A01)B11 -> C00 -= M4, C01 += M4
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        # M5 = (A10-A00)(B00+B01) -> C11 += M5
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # M6 = (A01-A11)(B10+B11) -> C00 += M6
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    col_idx = 0
    for I in range(blocks_per_dim):
        for J in range(blocks_per_dim):
            for K in range(blocks_per_dim):
                for u_coeffs, v_coeffs, w_coeffs in strassen_terms:
                    # Process U (A matrix)
                    for (ri, ci), val in u_coeffs.items():
                        global_i = I * block_size + ri
                        global_j = J * block_size + ci
                        U[global_i * m + global_j][col_idx] += val
                    
                    # Process V (B matrix)
                    for (ri, ci), val in v_coeffs.items():
                        global_j = J * block_size + ri
                        global_k = K * block_size + ci
                        V[global_j * p + global_k][col_idx] += val
                    
                    # Process W (C matrix)
                    # C[k, i] where k is col of C (from B's row), i is row of C (from A's col)
                    # Wait, standard matrix mult: C = A B.
                    # C[i, k] = sum_j A[i, j] B[j, k].
                    # The task says: "output factor follows the official trace convention W[k*n+i,r]"
                    # This means W is indexed by (k, i) -> k*n + i.
                    # So W[k*n + i] corresponds to C[i, k].
                    # In Strassen, we compute C entries.
                    # The w_coeffs dict maps (row, col) in the 2x2 block C to a value.
                    # Let (ri, ci) be the local indices in the 2x2 C block.
                    # Global row i = I * block_size + ri
                    # Global col k = K * block_size + ci
                    # So we update W[k * n + i]
                    for (ri, ci), val in w_coeffs.items():
                        global_i = I * block_size + ri
                        global_k = K * block_size + ci
                        W[global_k * n + global_i][col_idx] += val
                    
                    col_idx += 1
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
