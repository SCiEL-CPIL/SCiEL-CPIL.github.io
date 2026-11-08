"""Matrix multiplication 2x3x10 tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying
    a 2x3 matrix by a 3x10 matrix.
    
    Strategy: Split p=10 into 5 blocks of size 2. Each block is a 2x3x2
    multiplication decomposed with 7 Strassen terms (for the 2x2x2 part
    using columns j=0,1) plus 4 outer-product terms (for column j=2).
    Total rank = 5 * 11 = 55.
    
    The output factor W follows the convention W[k*n+i, r].
    """
    deadline = time.time() + 140.0

    n, m, p = DIMENSIONS
    num_blocks = p // 2  # 5 blocks of size 2
    block_size = 2
    rank_per_block = 11
    total_rank = num_blocks * rank_per_block  # 55

    U = np.zeros((n * m, total_rank), dtype=np.float64)   # 6 x 55
    V = np.zeros((m * p, total_rank), dtype=np.float64)   # 30 x 55
    W = np.zeros((p * n, total_rank), dtype=np.float64)   # 20 x 55

    # Index conventions (local to a 2x3x2 block):
    #   A[i,j]  -> U index  i*m + j   (i in 0..1, j in 0..2)  => i*3+j
    #   B[j,k]  -> V index  j*p + k   (j in 0..2, k in 0..1)  => j*10+k
    #   C[i,k]  -> W index  k*n + i   (k in 0..1, i in 0..1)  => k*2+i
    #
    # Local indices within a block of k in {0,1}:
    #   k_local in {0,1};  k_global = b*2 + k_local
    #   V local index for B[j, k_local] = j*2 + k_local  (we remap to global)
    #   W local index for C[i, k_local] = k_local*2 + i

    # Corrected Strassen decomposition for 2x2x2:
    #   M1 = (A00+A11)(B00+B11)
    #   M2 = (A10+A11)B00
    #   M3 = A00(B01-B11)
    #   M4 = A11(B10-B00)
    #   M5 = (A00+A01)B11
    #   M6 = (A10-A00)(B00+B01)
    #   M7 = (A01-A11)(B10+B11)
    #
    #   C00 = M1 + M4 - M5 + M7
    #   C01 = M3 + M5
    #   C10 = M2 + M4
    #   C11 = M1 - M2 + M3 + M6
    #
    # Each term is a rank-1 contribution to the 2x3x2 tensor.
    # For each term we list (local_U_index, coeff), (local_V_index, coeff), (local_W_index, coeff)
    # where local_V_index = j*2 + k_local and local_W_index = k_local*2 + i.

    terms_11 = [
        # M1 = (A00+A11)(B00+B11): C00 += 1, C11 += 1
        {'U': [(0, 1), (4, 1)], 'V': [(0, 1), (3, 1)], 'W': [(0, 1), (3, 1)]},
        # M2 = (A10+A11)B00: C10 += 1
        {'U': [(3, 1), (4, 1)], 'V': [(0, 1)], 'W': [(1, 1)]},
        # M3 = A00(B01-B11): C01 += 1
        {'U': [(0, 1)], 'V': [(1, 1), (3, -1)], 'W': [(2, 1)]},
        # M4 = A11(B10-B00): C00 += 1, C10 += 1
        {'U': [(4, 1)], 'V': [(2, 1), (0, -1)], 'W': [(0, 1), (1, 1)]},
        # M5 = (A00+A01)B11: C00 -= 1, C01 += 1
        {'U': [(0, 1), (1, 1)], 'V': [(3, 1)], 'W': [(0, -1), (2, 1)]},
        # M6 = (A10-A00)(B00+B01): C11 += 1
        {'U': [(3, 1), (0, -1)], 'V': [(0, 1), (1, 1)], 'W': [(3, 1)]},
        # M7 = (A01-A11)(B10+B11): C00 += 1
        {'U': [(1, 1), (4, -1)], 'V': [(2, 1), (3, 1)], 'W': [(0, 1)]},
        # Outer: A02 * B20 -> C00 += 1
        {'U': [(2, 1)], 'V': [(4, 1)], 'W': [(0, 1)]},
        # Outer: A02 * B21 -> C01 += 1
        {'U': [(2, 1)], 'V': [(5, 1)], 'W': [(2, 1)]},
        # Outer: A12 * B20 -> C10 += 1
        {'U': [(5, 1)], 'V': [(4, 1)], 'W': [(1, 1)]},
        # Outer: A12 * B21 -> C11 += 1
        {'U': [(5, 1)], 'V': [(5, 1)], 'W': [(3, 1)]},
    ]

    col_offset = 0
    for b in range(num_blocks):
        if time.time() > deadline:
            break
        for term in terms_11:
            col = col_offset
            # Fill U: index is i*3 + j (global, same as local since A doesn't depend on k)
            for u_idx, val in term['U']:
                U[u_idx, col] += val
            # Fill V: local index = j*2 + k_local -> global V index = j*p + (b*2 + k_local)
            for v_local, val in term['V']:
                j = v_local // 2
                k_local = v_local % 2
                k_global = b * block_size + k_local
                V[j * p + k_global, col] += val
            # Fill W: local index = k_local*2 + i -> global W index = (b*2 + k_local)*n + i
            for w_local, val in term['W']:
                k_local = w_local // 2
                i = w_local % 2
                k_global = b * block_size + k_local
                W[k_global * n + i, col] += val
            col_offset += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
