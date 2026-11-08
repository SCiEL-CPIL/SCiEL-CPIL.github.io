import numpy as np
import time

def construct():
    start_time = time.time()
    n, m, p = 4, 4, 8
    rank = 98

    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    # Standard 4x4x4 Strassen decomposition (rank 49)
    # A is 4x4, B is 4x4, C = A*B is 4x4
    # Flattened: A[i*4+j], B[j*4+k], C[i*4+k]
    # U: (16, 49), V: (16, 49), W: (16, 49)

    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))

    # Build 4x4 Strassen from 2x2 Strassen blocks
    # 2x2 Strassen (rank 7):
    # M0 = (A00+A11)(B00+B11)
    # M1 = (A10+A11)B00
    # M2 = A00(B10-B11)
    # M3 = A11(B01-B10)
    # M4 = (A00+A01)B11
    # M5 = (A10-A00)(B00+B01)
    # M6 = (A01-A11)(B10+B11)
    # C00 = M0 + M3 - M4 - M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5

    # 2x2 Strassen matrices (indices: [00,01,10,11] for 2x2)
    S2_U = np.zeros((4, 7))
    S2_V = np.zeros((4, 7))
    S2_W = np.zeros((4, 7))

    # M0 = (A00+A11)(B00+B11)
    S2_U[0, 0] = 1; S2_U[3, 0] = 1
    S2_V[0, 0] = 1; S2_V[3, 0] = 1
    S2_W[0, 0] = 1; S2_W[3, 0] = 1

    # M1 = (A10+A11)B00
    S2_U[2, 1] = 1; S2_U[3, 1] = 1
    S2_V[0, 1] = 1
    S2_W[2, 1] = 1; S2_W[3, 1] = -1

    # M2 = A00(B10-B11)
    S2_U[0, 2] = 1
    S2_V[2, 2] = 1; S2_V[3, 2] = -1
    S2_W[1, 2] = 1; S2_W[3, 2] = 1

    # M3 = A11(B01-B10)
    S2_U[3, 3] = 1
    S2_V[1, 3] = 1; S2_V[0, 3] = -1
    S2_W[0, 3] = 1; S2_W[2, 3] = 1

    # M4 = (A00+A01)B11
    S2_U[0, 4] = 1; S2_U[1, 4] = 1
    S2_V[3, 4] = 1
    S2_W[0, 4] = -1; S2_W[1, 4] = 1

    # M5 = (A10-A00)(B00+B01)
    S2_U[2, 5] = 1; S2_U[0, 5] = -1
    S2_V[0, 5] = 1; S2_V[1, 5] = 1
    S2_W[3, 5] = 1

    # M6 = (A01-A11)(B10+B11)
    S2_U[1, 6] = 1; S2_U[3, 6] = -1
    S2_V[2, 6] = 1; S2_V[3, 6] = 1
    S2_W[0, 6] = 1

    # Now build 4x4x4 Strassen using 2x2 blocks
    # 4x4 = [[A00,A01],[A10,A11]] where each is 2x2
    # Similarly B = [[B00,B01],[B10,B11]], C = [[C00,C01],[C10,C11]]
    # C = A*B means:
    #   C00 = A00*B00 + A01*B10
    #   C01 = A00*B01 + A01*B11
    #   C10 = A10*B00 + A11*B10
    #   C11 = A10*B01 + A11*B11
    #
    # Apply Strassen at 4x4 level (treating 2x2 blocks as elements):
    # M0 = (A00+A11)(B00+B11)
    # M1 = (A10+A11)B00
    # M2 = A00(B10-B11)
    # M3 = A11(B01-B10)
    # M4 = (A00+A01)B11
    # M5 = (A10-A00)(B00+B01)
    # M6 = (A01-A11)(B10+B11)
    #
    # Each M_i is a 2x2 matrix product, computed via 2x2 Strassen (rank 7)
    # Total: 7 * 7 = 49 multiplications
    #
    # C00 = M0 + M3 - M4 - M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5

    # For each of the 7 block-level Strassen terms (k=0..6),
    # we have a 2x2 matrix product computed via 2x2 Strassen (s=0..6)
    #
    # Block-level coefficients:
    # For U (A side): which 2x2 blocks of A contribute, with what sign
    # For V (B side): which 2x2 blocks of B contribute, with what sign
    # For W (C side): which 2x2 blocks of C receive the result, with what sign

    # A block structure: A00=top-left, A01=top-right, A10=bottom-left, A11=bottom-right
    # In flattened 4x4: A00 = indices 0-3, A01 = indices 4-7, A10 = indices 8-11, A11 = indices 12-15

    # B block structure: B00=top-left, B01=top-right, B10=bottom-left, B11=bottom-right
    # In flattened 4x4: B00 = indices 0-3, B01 = indices 4-7, B10 = indices 8-11, B11 = indices 12-15

    # C block structure: C00=top-left, C01=top-right, C10=bottom-left, C11=bottom-right
    # In flattened 4x4 (row-major, C[i*4+k]): C00 = indices 0-3, C01 = indices 4-7, C10 = indices 8-11, C11 = indices 12-15

    # Block-level Strassen specs: (A_blocks, B_blocks, C_blocks)
    # Each entry: (block_index, coefficient)
    # Block indices: 0=A00/B00/C00, 1=A01/B01/C01, 2=A10/B10/C10, 3=A11/B11/C11

    block_specs = [
        # M0 = (A00+A11)(B00+B11), appears in C00(+), C11(+)
        {
            'U_blocks': [(0, 1), (3, 1)],
            'V_blocks': [(0, 1), (3, 1)],
            'W_blocks': [(0, 1), (3, 1)]
        },
        # M1 = (A10+A11)B00, appears in C10(+), C11(-)
        {
            'U_blocks': [(2, 1), (3, 1)],
            'V_blocks': [(0, 1)],
            'W_blocks': [(2, 1), (3, -1)]
        },
        # M2 = A00(B10-B11), appears in C01(+), C11(+)
        {
            'U_blocks': [(0, 1)],
            'V_blocks': [(2, 1), (3, -1)],
            'W_blocks': [(1, 1), (3, 1)]
        },
        # M3 = A11(B01-B10), appears in C00(+), C10(+)
        {
            'U_blocks': [(3, 1)],
            'V_blocks': [(1, 1), (2, -1)],
            'W_blocks': [(0, 1), (2, 1)]
        },
        # M4 = (A00+A01)B11, appears in C00(-), C01(+)
        {
            'U_blocks': [(0, 1), (1, 1)],
            'V_blocks': [(3, 1)],
            'W_blocks': [(0, -1), (1, 1)]
        },
        # M5 = (A10-A00)(B00+B01), appears in C11(+)
        {
            'U_blocks': [(2, 1), (0, -1)],
            'V_blocks': [(0, 1), (1, 1)],
            'W_blocks': [(3, 1)]
        },
        # M6 = (A01-A11)(B10+B11), appears in C00(-)
        {
            'U_blocks': [(1, 1), (3, -1)],
            'V_blocks': [(2, 1), (3, 1)],
            'W_blocks': [(0, 1)]
        },
    ]

    # For each block-level term k (0..6) and 2x2-Strassen term s (0..6):
    # Column index = k*7 + s
    #
    # The 2x2 Strassen operates on the 2x2 blocks.
    # For U: the 2x2 block at position block_idx contributes S2_U[local_idx, s] * coeff
    #   where local_idx maps the 2x2 block's internal (r,c) to global 4x4 index
    #
    # Block 0 (top-left 2x2): global indices 0-3, local (r*2+c)
    # Block 1 (top-right 2x2): global indices 4-7, local (r*2+c) -> 4 + r*2 + c
    # Block 2 (bottom-left 2x2): global indices 8-11, local (r*2+c) -> 8 + r*2 + c
    # Block 3 (bottom-right 2x2): global indices 12-15, local (r*2+c) -> 12 + r*2 + c

    for k in range(7):
        spec = block_specs[k]
        for s in range(7):
            col = k * 7 + s

            # U contributions
            for block_idx, coeff in spec['U_blocks']:
                for r in range(2):
                    for c in range(2):
                        global_idx = block_idx * 4 + r * 2 + c
                        local_idx = r * 2 + c
                        U4[global_idx, col] += coeff * S2_U[local_idx, s]

            # V contributions
            for block_idx, coeff in spec['V_blocks']:
                for r in range(2):
                    for c in range(2):
                        global_idx = block_idx * 4 + r * 2 + c
                        local_idx = r * 2 + c
                        V4[global_idx, col] += coeff * S2_V[local_idx, s]

            # W contributions
            for block_idx, coeff in spec['W_blocks']:
                for r in range(2):
                    for c in range(2):
                        global_idx = block_idx * 4 + r * 2 + c
                        local_idx = r * 2 + c
                        W4[global_idx, col] += coeff * S2_W[local_idx, s]

    # Now assemble the 4x4x8 decomposition from two 4x4x4 Strassen instances
    # First instance: computes C[:, 0:4] = A * B[:, 0:4]
    #   U: rows 0-15 of U (A has 16 elements), columns 0-48
    #   V: rows 0-15 of V (B[:, 0:4] has 16 elements at V indices 0-15), columns 0-48
    #   W: rows 0-15 of W (C[:, 0:4] at W indices 0-15), columns 0-48
    #
    # Second instance: computes C[:, 4:8] = A * B[:, 4:8]
    #   U: rows 0-15 of U (same A), columns 49-97
    #   V: rows 16-31 of V (B[:, 4:8] at V indices 16-31), columns 49-97
    #   W: rows 16-31 of W (C[:, 4:8] at W indices 16-31), columns 49-97

    U[:, 0:49] = U4
    U[:, 49:98] = U4

    V[0:16, 0:49] = V4
    V[16:32, 49:98] = V4

    W[0:16, 0:49] = W4
    W[16:32, 49:98] = W4

    # Safety check
    if time.time() - start_time > 140:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
