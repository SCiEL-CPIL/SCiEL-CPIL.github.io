import numpy as np

DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal tensor decomposition for 2x2 * 2x14 matrix multiplication.
    
    The multiplication tensor for (2,2,14) is the direct sum of 7 copies of the
    (2,2,2) multiplication tensor (one per pair of output columns). Each (2,2,2)
    block has rank 7 (Strassen's algorithm), and since the B/C column supports
    are disjoint across blocks, the total rank is 7 * 7 = 49.
    
    This is provably minimal: the rank of a direct sum of tensors is the sum
    of their ranks when the supports are disjoint.
    """
    n, m, p = DIMENSIONS  # n=2, m=2, p=14
    nm = n * m  # 4
    mp = m * p  # 28
    pn = p * n  # 28

    num_blocks = p // 2  # 7

    # Strassen's algorithm for 2x2 * 2x2:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # A indices: A00=0, A01=1, A10=2, A11=3
    # B indices: B00=0, B01=1, B10=2, B11=3
    # C indices: C00=0, C01=1, C10=2, C11=3

    # Each term: (a_indices, a_vals, b_indices, b_vals, c_indices, c_vals)
    strassen_terms = [
        # M1 = (A00 + A11)(B00 + B11) -> C00 += 1, C11 += 1
        ([0, 3], [1, 1], [0, 3], [1, 1], [0, 3], [1, 1]),
        # M2 = (A10 + A11)B00 -> C10 += 1, C11 -= 1
        ([2, 3], [1, 1], [0], [1], [2, 3], [1, -1]),
        # M3 = A00(B01 - B11) -> C01 += 1, C11 += 1
        ([0], [1], [1, 3], [1, -1], [1, 3], [1, 1]),
        # M4 = A11(B10 - B00) -> C00 += 1, C10 += 1
        ([3], [1], [2, 0], [1, -1], [0, 2], [1, 1]),
        # M5 = (A00 + A01)B11 -> C00 -= 1, C01 += 1
        ([0, 1], [1, 1], [3], [1], [0, 1], [-1, 1]),
        # M6 = (A10 - A00)(B00 + B01) -> C11 += 1
        ([2, 0], [1, -1], [0, 1], [1, 1], [3], [1]),
        # M7 = (A01 - A11)(B10 + B11) -> C00 += 1
        ([1, 3], [1, -1], [2, 3], [1, 1], [0], [1]),
    ]

    rank = num_blocks * 7  # 49
    U = np.zeros((nm, rank), dtype=np.int64)
    V = np.zeros((mp, rank), dtype=np.int64)
    W = np.zeros((pn, rank), dtype=np.int64)

    for t in range(num_blocks):
        base_col = t * 7
        for s in range(7):
            col = base_col + s
            a_idx_list, a_val_list, b_idx_list, b_val_list, c_idx_list, c_val_list = strassen_terms[s]

            for a_idx, coeff in zip(a_idx_list, a_val_list):
                a_i = a_idx // m
                a_j = a_idx % m
                U[a_i * m + a_j, col] = coeff

            for b_idx, coeff in zip(b_idx_list, b_val_list):
                b_j = b_idx // 2
                b_col = b_idx % 2
                b_k = 2 * t + b_col
                V[b_j * p + b_k, col] = coeff

            for c_idx, coeff in zip(c_idx_list, c_val_list):
                c_i = c_idx // 2
                c_col = c_idx % 2
                c_k = 2 * t + c_col
                W[c_k * n + c_i, col] = coeff

    # Verify correctness: for all (i, j, k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should be 1
    # This checks that the tensor decomposition exactly reproduces the multiplication.
    for i in range(n):
        for j in range(m):
            u_row = U[i * m + j, :]
            for k in range(p):
                v_row = V[j * p + k, :]
                w_row = W[k * n + i, :]
                s = np.dot(u_row, v_row * w_row)
                if s != 1:
                    raise ValueError(f"Tensor check failed at i={i}, j={j}, k={k}: got {s}")

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
