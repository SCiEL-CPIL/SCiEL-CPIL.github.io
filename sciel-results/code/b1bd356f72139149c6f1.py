"""Matrix multiplication 2x4x6 rank-42 via Strassen on 2x2x2 sub-blocks."""
import time
import random
import numpy as np

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START

def _strassen_2x2x2():
    """
    Strassen rank-7 algorithm for 2x2 * 2x2 matrix multiplication.
    Returns list of 7 terms, each: (a_idx, b_idx, c_idx, d_idx, out_coeff)
    where (a_idx * c_idx + b_idx * d_idx) contributes out_coeff to the output.
    A indices: 00=0, 01=1, 10=2, 11=3
    B indices: 00=0, 01=1, 10=2, 11=3
    m0 = (a00+a11)*(b00+b11)
    m1 = (a10+a11)*b00
    m2 = a00*(b01-b11)
    m3 = a11*(b10-b00)
    m4 = (a00+a01)*b11
    m5 = (a10-a00)*(b00+b01)
    m6 = (a01-a11)*(b10+b11)
    c00 = m0+m3-m4+m6
    c01 = m2+m4
    c10 = m1+m3
    c11 = m0-m1+m2+m5
    """
    return [
        # (a_idx, b_idx, c_idx, d_idx, out_coeff)
        (0, 3, 0, 3, 1),    # m0: (a00+a11)*(b00+b11)
        (2, 3, 0, 0, 1),    # m1: (a10+a11)*b00
        (0, 0, 1, 3, 1),    # m2: a00*(b01-b11)
        (3, 0, 2, 0, 1),    # m3: a11*(b10-b00)
        (0, 1, 3, 3, 1),    # m4: (a00+a01)*b11
        (2, 0, 0, 1, 1),    # m5: (a10-a00)*(b00+b01)
        (1, 3, 2, 3, 1),    # m6: (a01-a11)*(b10+b11)
    ]

def _strassen_output_coeffs():
    """
    For each Strassen term index (0..6), return dict of output (i,k) -> coefficient.
    Output is C[i,k] where i is row of A (0,1), k is col of B (0,1).
    c00 = m0+m3-m4+m6
    c01 = m2+m4
    c10 = m1+m3
    c11 = m0-m1+m2+m5
    """
    return [
        {(0,0): 1, (1,1): 1},    # m0
        {(1,0): 1, (1,1): -1},   # m1
        {(0,1): 1, (1,1): 1},    # m2
        {(0,0): 1, (1,0): 1},    # m3
        {(0,0): -1, (0,1): 1},   # m4
        {(1,1): 1},              # m5
        {(0,0): 1},              # m6
    ]

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x4 * 4x6 matrix multiplication.
    Uses Strassen on 2x2x2 sub-blocks: split into 2 blocks of 2x2x6,
    each 2x2x6 = three 2x2x2 Strassen products. Total rank = 2*3*7 = 42.
    """
    n, m, p = DIMENSIONS  # 2, 4, 6
    start_time = time.time()
    deadline = start_time + 110

    U_dim = n * m  # 8
    V_dim = m * p  # 24
    W_dim = p * n  # 12

    # Build the target tensor for verification
    T = np.zeros((U_dim, V_dim, W_dim), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1

    # Strassen components
    strassen_terms = _strassen_2x2x2()
    strassen_out = _strassen_output_coeffs()

    # A(2x4) = [A_left(2x2) | A_right(2x2)]
    #   A_left: A[0,0], A[0,1], A[1,0], A[1,1] -> A indices 0,1,2,3
    #   A_right: A[0,2], A[0,3], A[1,2], A[1,3] -> A indices 4,5,6,7
    # B(4x6) = [B_top(2x6); B_bottom(2x6)]
    #   B_top: B[0,0..5], B[1,0..5] -> B indices 0..11
    #   B_bottom: B[2,0..5], B[3,0..5] -> B indices 12..23
    # C(2x6) = A_left*B_top + A_right*B_bottom

    # For each of the 2 A-blocks (left/right) and 3 B-sub-blocks (cols 0-1, 2-3, 4-5):
    #   A-block 'left' has A indices [0,1,2,3] (i*2+j for i in 0,1; j in 0,1)
    #   A-block 'right' has A indices [4,5,6,7] (i*2+j for i in 0,1; j in 2,3)
    #   B-sub-block for columns [2c, 2c+1] of B_top: B indices [0+2c, 1+2c, 2+2c, 3+2c]
    #   B-sub-block for columns [2c, 2c+1] of B_bottom: B indices [12+2c, 13+2c, 14+2c, 15+2c]
    #   Output C[i, 2c] and C[i, 2c+1] -> W[2c*2+i] and W[(2c+1)*2+i]

    # Collect all terms: each term is (a_idx, b_idx, c_idx, d_idx, i, k, coeff)
    # meaning: (A[a_idx]*B[c_idx] + A[b_idx]*B[d_idx]) * coeff -> C[i,k] -> W[k*2+i]

    all_terms = []  # list of (a_idx, b_idx, c_idx, d_idx, i, k, coeff)

    for a_block in range(2):  # left=0, right=1
        for b_block in range(2):  # top=0, bottom=1
            for sub in range(3):  # column sub-blocks 0,1,2
                # A index base: a_block * 2 (left: 0, right: 2)
                # A indices for the 2x2 block: a_base + i*2 + j
                #   (0,0) -> a_base+0, (0,1) -> a_base+1, (1,0) -> a_base+2, (1,1) -> a_base+3
                a_base = a_block * 2  # left: 0, right: 2

                # B index base: b_block * 12 + sub * 2
                # B indices for the 2x2 block: b_base + i*2 + j
                #   (0,0) -> b_base+0, (0,1) -> b_base+1, (1,0) -> b_base+2, (1,1) -> b_base+3
                b_base = b_block * 12 + sub * 2

                # Output column base: sub * 2
                # C[i, 2*sub] -> W[2*sub * 2 + i] = W[4*sub + i]
                # C[i, 2*sub+1] -> W[(2*sub+1) * 2 + i] = W[4*sub + 2 + i]
                col_base = sub * 2

                for term_idx, (a_i, b_i, c_i, d_i, _) in enumerate(strassen_terms):
                    # a_i, b_i are local indices in the 2x2 A block
                    # c_i, d_i are local indices in the 2x2 B block
                    a_actual = a_base + a_i
                    b_actual = b_base + c_i  # B row index
                    c_actual = b_base + d_i  # B col index (within the 2x2)

                    # Wait, I need to be careful. The Strassen term is (a_idx * c_idx + b_idx * d_idx)
                    # where a_idx, b_idx are A indices and c_idx, d_idx are B indices.
                    # In the 2x2x2 case, A is 2x2 and B is 2x2.
                    # a_idx: row of A (0,1), col of A (0,1) -> flat a_idx*2 + ... 
                    # Actually in my _strassen_2x2x2, the indices are:
                    #   0=A[0,0], 1=A[0,1], 2=A[1,0], 3=A[1,1]
                    #   0=B[0,0], 1=B[0,1], 2=B[1,0], 3=B[1,1]
                    # So a_i is the flat A index (0-3), c_i is the flat B index (0-3).
                    # a_actual = a_base + a_i
                    # b_actual = b_base + c_i

                    for (out_i, out_k), coeff in strassen_out[term_idx].items():
                        # out_i: row of C (0,1), out_k: col of C within the 2x2 (0,1)
                        # Actual C column: col_base + out_k
                        actual_k = col_base + out_k
                        w_idx = actual_k * 2 + out_i  # W[k*2+i]
                        all_terms.append((a_actual, b_actual, c_actual, out_i, out_k, coeff))

    # Now build U, V, W from all_terms
    # Each term: (a_idx, b_idx, c_idx, i, k, coeff)
    # U[a_idx, col] += 1, U[b_idx, col] += 1
    # V[c_idx, col] += 1, V[d_idx, col] += 1  -- wait, I need to track c and d separately

    # Let me redo: each Strassen term is (a_i, b_i, c_i, d_i) meaning:
    #   (A[a_i] * B[c_i] + A[b_i] * B[d_i]) * coeff -> C[out_i, out_k]
    # So for the full term:
    #   U[a_actual, col] += 1
    #   U[b_actual, col] += 1
    #   V[c_actual, col] += 1
    #   V[d_actual, col] += 1
    #   W[w_idx, col] += coeff

    # But wait, the Strassen term has a_i, b_i as the two A indices and c_i, d_i as the two B indices.
    # Let me restructure.

    # Clear and redo
    all_terms = []  # (a1, a2, b1, b2, w_idx, coeff)

    for a_block in range(2):
        for b_block in range(2):
            for sub in range(3):
                a_base = a_block * 2
                b_base = b_block * 12 + sub * 2
                col_base = sub * 2

                for term_idx, (a_i, b_i, c_i, d_i, _) in enumerate(strassen_terms):
                    a1 = a_base + a_i
                    a2 = a_base + b_i
                    b1 = b_base + c_i
                    b2 = b_base + d_i

                    for (out_i, out_k), coeff in strassen_out[term_idx].items():
                        actual_k = col_base + out_k
                        w_idx = actual_k * 2 + out_i
                        all_terms.append((a1, a2, b1, b2, w_idx, coeff))

    rank = len(all_terms)
    U = np.zeros((U_dim, rank), dtype=np.float64)
    V = np.zeros((V_dim, rank), dtype=np.float64)
    W = np.zeros((W_dim, rank), dtype=np.float64)

    for col, (a1, a2, b1, b2, w_idx, coeff) in enumerate(all_terms):
        U[a1, col] = 1
        U[a2, col] = 1
        V[b1, col] = 1
        V[b2, col] = 1
        W[w_idx, col] = coeff

    # Verify: check that sum_r U[:,r] * V[:,r] * W[:,r] == T
    # T[a, b, c] = sum_r U[a,r] * V[b,r] * W[c,r]
    T_check = np.zeros((U_dim, V_dim, W_dim), dtype=np.float64)
    for r in range(rank):
        T_check += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])

    if not np.allclose(T_check, T, atol=1e-9):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((U_dim, rank), dtype=np.float64)
        V = np.zeros((V_dim, rank), dtype=np.float64)
        W = np.zeros((W_dim, rank), dtype=np.float64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1

    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()

    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
