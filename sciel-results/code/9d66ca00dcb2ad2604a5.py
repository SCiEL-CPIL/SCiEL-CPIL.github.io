"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # 6x6x6 matrix multiplication.
    # We use a 3x3 block decomposition where each 2x2 block is computed
    # using the 7-term Strassen algorithm.
    # This gives 9 blocks * 7 = 63 rank terms.
    #
    # Wait - that's not right. Each block multiplication C_ij = sum_k A_ik * B_kj
    # requires n*m*p block multiplications = 3*3*3 = 27 block mults.
    # Each 2x2 block mult uses Strassen with 7 terms.
    # Total rank = 27 * 7 = 189.
    #
    # The idea in the strategy was "3x3 block decomposition where each 2x2 block
    # is computed using the 7-term Strassen algorithm, yielding a total rank of 63".
    # That would only work if we could compute each C_ij with just 7 operations total,
    # which isn't how block matrix multiplication works. The 63 claim appears to be
    # incorrect - you still need 27 block multiplications each of rank 7, giving 189.
    #
    # However, let me reconsider. Perhaps there's a different interpretation:
    # If we view the 6x6 matrices as having 3x3 blocks of size 2x2, then:
    # C[i][j] = sum_k A[i][k] * B[k][j] for i,j,k in 0..2
    # Each A[i][k] and B[k][j] are 2x2 matrices.
    # We need 27 such 2x2x2x2 multiplications.
    # Each has rank 7 via Strassen.
    # Total rank = 189.
    #
    # Let me try a different approach: use the 3x3 block structure more cleverly.
    # Actually, let me just verify the rank-189 solution works correctly.
    
    block_size = 2
    blocks_per_dim = 3  # 6 / 2
    
    rank = (blocks_per_dim ** 3) * 7  # 27 * 7 = 189
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen patterns for 2x2 matrices
    # M0 = (A00+A11)(B00+B11)
    # M1 = (A10+A11)B00
    # M2 = A00(B01-B11)
    # M3 = A11(B10-B00)
    # M4 = (A00+A01)B11
    # M5 = (A10-A00)(B00+B01)
    # M6 = (A01-A11)(B10+B11)
    #
    # C00 = M0 + M3 - M4 - M6
    # C01 = M2 + M4
    # C10 = M1 + M5
    # C11 = M0 - M1 + M5 + M6
    
    strassen_patterns = [
        # (u_coeffs, v_coeffs, w_coeffs)
        # M0 = (A00+A11)(B00+B11)
        # U: A00, A11 -> coeffs at (0,0) and (1,1)
        # V: B00, B11 -> coeffs at (0,0) and (1,1)
        # W: C00 += 1, C11 += 1 -> at (0,0) and (1,1)
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        
        # M1 = (A10+A11)B00
        # U: A10, A11 -> (1,0) and (1,1)
        # V: B00 -> (0,0)
        # W: C10 += 1, C11 -= 1 -> (1,0) and (1,1)
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        
        # M2 = A00(B01-B11)
        # U: A00 -> (0,0)
        # V: B01, -B11 -> (0,1) and (1,1)
        # W: C01 += 1 -> (0,1)
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1}),
        
        # M3 = A11(B10-B00)
        # U: A11 -> (1,1)
        # V: B10, -B00 -> (1,0) and (0,0)
        # W: C00 += 1 -> (0,0)
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1}),
        
        # M4 = (A00+A01)B11
        # U: A00, A01 -> (0,0) and (0,1)
        # V: B11 -> (1,1)
        # W: C00 -= 1, C01 += 1 -> (0,0) and (0,1)
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        
        # M5 = (A10-A00)(B00+B01)
        # U: A10, -A00 -> (1,0) and (0,0)
        # V: B00, B01 -> (0,0) and (0,1)
        # W: C11 += 1 -> (1,1)
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        
        # M6 = (A01-A11)(B10+B11)
        # U: A01, -A11 -> (0,1) and (1,1)
        # V: B10, B11 -> (1,0) and (1,1)
        # W: C00 -= 1 -> (0,0)
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): -1}),
    ]
    
    col_idx = 0
    for I in range(blocks_per_dim):
        for J in range(blocks_per_dim):
            for K in range(blocks_per_dim):
                for u_coeffs, v_coeffs, w_coeffs in strassen_patterns:
                    # Update U: A is n x m = 6 x 6
                    # A[I*bs + pi][J*bs + pj]
                    for (pi, pj), val in u_coeffs.items():
                        r = I * block_size + pi
                        c = J * block_size + pj
                        U[r * m + c][col_idx] += val
                    
                    # Update V: B is m x p = 6 x 6
                    # B[J*bs + pi][K*bs + pj]
                    for (pi, pj), val in v_coeffs.items():
                        r = J * block_size + pi
                        c = K * block_size + pj
                        V[r * p + c][col_idx] += val
                    
                    # Update W: C is p x n = 6 x 6, indexed as W[k*n + i]
                    # C[i][k] where i is row, k is col
                    # C[I*bs + pi][K*bs + pj]
                    for (pi, pj), val in w_coeffs.items():
                        k = K * block_size + pj
                        i = I * block_size + pi
                        W[k * n + i][col_idx] += val
                    
                    col_idx += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

