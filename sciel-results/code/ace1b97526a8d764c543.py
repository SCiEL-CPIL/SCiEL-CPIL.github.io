"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
import time

def construct():
    """Block-Strassen decomposition for 6x6x6 matrix multiplication.
    
    Uses 3x3 grid of 2x2 blocks, each computed via Strassen (rank 7).
    Total rank: 27 * 7 = 189.
    """
    n, m, p = DIMENSIONS  # all 6
    block_size = 2
    num_blocks = n // block_size  # 3
    
    # Strassen coefficients for 2x2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For each block position (bi, bj, bk) in the 3x3 grid:
    # U index: (bi * block_size + ai) * m + (bj * block_size + aj)
    # V index: (bj * block_size + aj) * p + (bk * block_size + ak)
    # W index: (bk * block_size + ak) * n + (bi * block_size + ai)
    # where ai, aj, ak are within-block indices (0 or 1)
    
    rank = num_blocks ** 3 * 7  # 27 * 7 = 189
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficient table:
    # For each of the 7 multiplications, the coefficients for U, V, W
    # U coeffs: for the two A elements involved
    # V coeffs: for the two B elements involved
    # W coeffs: for the C element that this M contributes to
    
    # M1: A11+A22, B11+B22 -> contributes to C11 (+1) and C22 (+1)
    # M2: A21+A22, B11 -> contributes to C21 (+1) and C22 (-1)
    # M3: A11, B21-B22 -> contributes to C12 (+1) and C22 (+1)
    # M4: A22, B12-B11 -> contributes to C11 (+1) and C21 (+1)
    # M5: A11+A12, B22 -> contributes to C11 (-1) and C12 (+1)
    # M6: A21-A11, B11+B12 -> contributes to C22 (+1)
    # M7: A12-A22, B21+B22 -> contributes to C11 (+1)
    
    # Each M has: (a_coeff_list, b_coeff_list, c_contribution_list)
    # a_coeff_list: list of (local_a_index, coeff)
    # b_coeff_list: list of (local_b_index, coeff)
    # c_contribution_list: list of (local_c_index, coeff)
    
    strassen_terms = [
        # M1: (A11+A22)(B11+B22)
        {
            'a': [(0, 1), (3, 1)],  # A11 is (0,0)->idx0, A22 is (1,1)->idx3
            'b': [(0, 1), (3, 1)],  # B11 is (0,0)->idx0, B22 is (1,1)->idx3
            'c': [(0, 1), (3, 1)]   # C11 is (0,0)->idx0, C22 is (1,1)->idx3
        },
        # M2: (A21+A22)B11
        {
            'a': [(2, 1), (3, 1)],  # A21 is (1,0)->idx2, A22 is (1,1)->idx3
            'b': [(0, 1)],          # B11 is (0,0)->idx0
            'c': [(2, 1), (3, -1)]  # C21 is (1,0)->idx2, C22 is (1,1)->idx3
        },
        # M3: A11(B21-B22)
        {
            'a': [(0, 1)],          # A11 is (0,0)->idx0
            'b': [(2, 1), (3, -1)], # B21 is (1,0)->idx2, B22 is (1,1)->idx3
            'c': [(1, 1), (3, 1)]   # C12 is (0,1)->idx1, C22 is (1,1)->idx3
        },
        # M4: A22(B12-B11)
        {
            'a': [(3, 1)],          # A22 is (1,1)->idx3
            'b': [(1, 1), (0, -1)], # B12 is (0,1)->idx1, B11 is (0,0)->idx0
            'c': [(0, 1), (2, 1)]   # C11 is (0,0)->idx0, C21 is (1,0)->idx2
        },
        # M5: (A11+A12)B22
        {
            'a': [(0, 1), (1, 1)],  # A11 is (0,0)->idx0, A12 is (0,1)->idx1
            'b': [(3, 1)],          # B22 is (1,1)->idx3
            'c': [(0, -1), (1, 1)]  # C11 is (0,0)->idx0, C12 is (0,1)->idx1
        },
        # M6: (A21-A11)(B11+B12)
        {
            'a': [(2, 1), (0, -1)], # A21 is (1,0)->idx2, A11 is (0,0)->idx0
            'b': [(0, 1), (1, 1)],  # B11 is (0,0)->idx0, B12 is (0,1)->idx1
            'c': [(3, 1)]           # C22 is (1,1)->idx3
        },
        # M7: (A12-A22)(B21+B22)
        {
            'a': [(1, 1), (3, -1)], # A12 is (0,1)->idx1, A22 is (1,1)->idx3
            'b': [(2, 1), (3, 1)],  # B21 is (1,0)->idx2, B22 is (1,1)->idx3
            'c': [(0, 1)]           # C11 is (0,0)->idx0
        },
    ]
    
    column = 0
    for bi in range(num_blocks):
        for bj in range(num_blocks):
            for bk in range(num_blocks):
                for term in strassen_terms:
                    # For each element in the U coefficient list
                    for (local_a_idx, a_coeff) in term['a']:
                        ai, aj = divmod(local_a_idx, 2)  # row, col within block
                        u_row = (bi * block_size + ai) * m + (bj * block_size + aj)
                        U[u_row][column] += a_coeff
                    
                    for (local_b_idx, b_coeff) in term['b']:
                        aj, ak = divmod(local_b_idx, 2)  # row, col within block
                        v_row = (bj * block_size + aj) * p + (bk * block_size + ak)
                        V[v_row][column] += b_coeff
                    
                    for (local_c_idx, c_coeff) in term['c']:
                        ak, ai = divmod(local_c_idx, 2)  # row, col within block
                        w_row = (bk * block_size + ak) * n + (bi * block_size + ai)
                        W[w_row][column] += c_coeff
                    
                    column += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

