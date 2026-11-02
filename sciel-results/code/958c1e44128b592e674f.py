import time
import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START

def _strassen_4x4_decomposition():
    n, m, p = 4, 4, 4
    rank = 49

    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)

    # Strassen 2x2 blocks
    # P1 = (A11+A22)(B11+B22)
    # P2 = (A21+A22)B11
    # P3 = A11(B21-B22)
    # P4 = A22(B12-B11)
    # P5 = (A11+A12)B22
    # P6 = (A21-A11)(B11+B12)
    # P7 = (A12-A22)(B21+B22)
    
    # Indices:
    # A is 4x4. Blocks: A11(0,0), A12(0,1), A21(1,0), A22(1,1)
    # B is 4x4. Blocks: B11(0,0), B12(0,1), B21(1,0), B22(1,1)
    # C is 4x4. Blocks: C11(0,0), C12(0,1), C21(1,0), C22(1,1)
    
    # U, V, W are flattened.
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    
    # We define the 7 products for 2x2.
    # Each product is a bilinear form.
    # For each block product (i, j) in 2x2 grid, we have 7 scalar products.
    # Total 49 products.
    
    # Let's define the coefficients for U, V, W for each of the 49 products.
    # Product r corresponds to block (br, bc) and scalar product s.
    # r = br * 7 + bc * 7? No.
    # The block products are applied to the 4x4 matrices.
    # The 4x4 matrix is divided into 2x2 blocks of 2x2 matrices.
    # So we have 4 block positions: (0,0), (0,1), (1,0), (1,1).
    # For each block position, we apply Strassen.
    # Wait, Strassen is applied to the 2x2 block structure.
    # The 4x4 matrix A is:
    # [ A11 A12 ]
    # [ A21 A22 ]
    # where Aij are 2x2 matrices.
    # The product C = AB is:
    # [ C11 C12 ]
    # [ C21 C22 ]
    # C11 = A11 B11 + A12 B21
    # C12 = A11 B12 + A12 B22
    # C21 = A21 B11 + A22 B21
    # C22 = A21 B12 + A22 B22
    
    # Strassen computes these using 7 products:
    # P1 = (A11+A22)(B11+B22)
    # P2 = (A21+A22)B11
    # P3 = A11(B21-B22)
    # P4 = A22(B12-B11)
    # P5 = (A11+A12)B22
    # P6 = (A21-A11)(B11+B12)
    # P7 = (A12-A22)(B21+B22)
    
    # C11 = P1 + P4 - P5 + P7
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P2 + P3 + P6
    
    # Each Pk is a product of two 2x2 matrices.
    # To compute a product of two 2x2 matrices (e.g., A11 * B11), we use 7 scalar products (Strassen again).
    # So for each of the 7 block products (P1..P7), we need 7 scalar products.
    # Total 49.
    
    # Let's map the 49 products to U, V, W.
    # For each block product pi (i=0..6), and each scalar product sj (j=0..6):
    # r = i * 7 + j
    
    # The block product pi involves specific blocks of A and B.
    # e.g., P1 involves A11, A22 and B11, B22.
    # A11 is at (0,0) of the 2x2 block grid.
    # A22 is at (1,1).
    
    # We need to define the contributions to U, V, W for each (pi, sj).
    
    # Let's define the Strassen 2x2 scalar products first.
    # For 2x2 matrices X, Y:
    # S1 = (X11+X22)(Y11+Y22)
    # S2 = (X21+X22)Y11
    # S3 = X11(Y21-Y22)
    # S4 = X22(Y12-Y11)
    # S5 = (X11+X12)Y22
    # S6 = (X21-X11)(Y11+Y12)
    # S7 = (X12-X22)(Y21+Y22)
    
    # Z11 = S1 + S4 - S5 + S7
    # Z12 = S3 + S5
    # Z21 = S2 + S4
    # Z22 = S1 - S2 + S3 + S6
    
    # Now, for the 4x4 case:
    # Each block product Pi is a product of two 2x2 matrices.
    # e.g., P1 = (A11+A22)(B11+B22).
    # Let X = A11+A22, Y = B11+B22.
    # Then P1 = X Y.
    # We compute XY using 7 scalar products S1..S7.
    # The result of XY is a 2x2 matrix Z.
    # Z contributes to C11, C12, C21, C22 based on the block combination.
    # For P1, Z contributes to C11, C12, C21, C22 with coefficients from the block combination.
    # Specifically, C11 gets P1, C12 gets nothing? No.
    # C11 = P1 + P4 - P5 + P7.
    # So P1 contributes to C11 with coeff 1.
    # P1 does not contribute to C12, C21, C22 directly?
    # Wait, the block products are intermediate.
    # C11 = P1 + P4 - P5 + P7.
    # C12 = P3 + P5.
    # C21 = P2 + P4.
    # C22 = P1 - P2 + P3 + P6.
    
    # So P1 contributes to C11 (coeff 1) and C22 (coeff 1).
    # P2 contributes to C21 (coeff 1) and C22 (coeff -1).
    # etc.
    
    # Now, for each Pi, we have 7 scalar products Sj.
    # Each Sj is a scalar.
    # The 2x2 matrix Z (result of Xi Yi) has entries Z11, Z12, Z21, Z22.
    # Z11 = S1 + S4 - S5 + S7
    # Z12 = S3 + S5
    # Z21 = S2 + S4
    # Z22 = S1 - S2 + S3 + S6
    
    # So for a given Pi, the contribution to the block C_ab is:
    # coeff_Pi * Z_ab.
    # And Z_ab is a linear combination of Sj.
    
    # So the total contribution of the scalar product Sj (within Pi) to the block C_ab is:
    # coeff_Pi * (coeff of Sj in Z_ab).
    
    # And this block C_ab corresponds to 4 scalar entries in the 4x4 matrix C.
    # C_ab is a 2x2 matrix.
    # So the scalar entries are C_ab[0,0], C_ab[0,1], C_ab[1,0], C_ab[1,1].
    
    # So for each (Pi, Sj), we can determine the entries in U, V, W.
    
    # Let's precompute the coefficients.
    
    # Strassen 2x2 scalar products (Sj) definitions:
    # Sj is a bilinear form in X, Y.
    # U_Sj, V_Sj, W_Sj are 4x4 matrices (flattened to 16x1).
    # Actually, X and Y are 2x2.
    # So U_Sj is a 4-element vector (for X), V_Sj is a 4-element vector (for Y).
    # W_Sj is a 4-element vector (for Z).
    
    # Let's define U_Sj, V_Sj, W_Sj for j=0..6.
    # Indices of X: 0,1,2,3 correspond to (0,0), (0,1), (1,0), (1,1).
    
    U_S = [
        # S1: (X11+X22)(Y11+Y22)
        [1, 0, 0, 1], # X11, X12, X21, X22
        [1, 0, 0, 1], # Y
        [1, 0, 0, 1]  # Z11? No, W is for Z.
        # W_Sj should be the coefficients for Z.
        # Z11 = S1 + ...
        # So W_S1 has 1 at Z11 position.
    ]
    
    # This is getting complicated to define manually.
    # The seed code already does this correctly.
    # I will reuse the seed code's logic but optimize it.
    
    return _strassen_4x4_decomposition_optimized()

def _strassen_4x4_decomposition_optimized():
    n, m, p = 4, 4, 4
    rank = 49

    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)

    # Precomputed Strassen 2x2 data from seed
    # a_combo, b_combo for block products
    # a_local, b_local for scalar products
    # w_block, v_local for combinations
    
    # I will copy the seed's precomputation as it is correct.
    
    _STRASSEN_2X2 = [
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
        ([(1,0,1),(1,1,1)], [(0,0,1)]),
        ([(0,0,1)], [(0,1,1),(1,1,-1)]),
        ([(1,1,1)], [(1,0,1),(0,0,-1)]),
        ([(0,0,1),(0,1,1)], [(1,1,1)]),
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)]),
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)]),
    ]

    _STRASSEN_C_COMBINATION = [
        (0, 0, [(0, 1), (3, 1), (4, -1), (6, 1)]),
        (0, 1, [(2, 1), (4, 1)]),
        (1, 0, [(1, 1), (3, 1)]),
        (1, 1, [(0, 1), (1, -1), (2, 1), (5, 1)]),
    ]

    block_products = [
        ([(0, 0, 1), (1, 1, 1)], [(0, 0, 1), (1, 1, 1)]),
        ([(1, 0, 1), (1, 1, 1)], [(0, 0, 1)]),
        ([(0, 0, 1)], [(0, 1, 1), (1, 1, -1)]),
        ([(1, 1, 1)], [(1, 0, 1), (0, 0, -1)]),
        ([(0, 0, 1), (0, 1, 1)], [(1, 1, 1)]),
        ([(1, 0, 1), (0, 0, -1)], [(0, 0, 1), (0, 1, 1)]),
        ([(0, 1, 1), (1, 1, -1)], [(1, 0, 1), (1, 1, 1)]),
    ]

    c_block_combination = [
        (0, 0, [(0, 1), (3, 1), (4, -1), (6, 1)]),
        (0, 1, [(2, 1), (4, 1)]),
        (1, 0, [(1, 1), (3, 1)]),
        (1, 1, [(0, 1), (1, -1), (2, 1), (5, 1)]),
    ]

    w_block = np.zeros((7, 2, 2), dtype=np.int64)
    for (cbr, cbc, c_list) in c_block_combination:
        for (pi, wi) in c_list:
            w_block[pi, cbr, cbc] = wi

    v_local = np.zeros((7, 2, 2), dtype=np.int64)
    for (lr, lc, v_list) in _STRASSEN_C_COMBINATION:
        for (si, vs) in v_list:
            v_local[si, lr, lc] = vs

    for i, (a_combo, b_combo) in enumerate(block_products):
        for s, (a_local, b_local) in enumerate(_STRASSEN_2X2):
            r = i * 7 + s

            for (a_br, a_bc, a_coeff) in a_combo:
                for (lr, lc, l_coeff) in a_local:
                    row = 2 * a_br + lr
                    col = 2 * a_bc + lc
                    U[row * m + col, r] += a_coeff * l_coeff

            for (b_br, b_bc, b_coeff) in b_combo:
                for (lr, lc, l_coeff) in b_local:
                    row = 2 * b_br + lr
                    col = 2 * b_bc + lc
                    V[row * p + col, r] += b_coeff * l_coeff

            for cbr in range(2):
                for cbc in range(2):
                    wi = w_block[i, cbr, cbc]
                    if wi == 0:
                        continue
                    for lr in range(2):
                        for lc in range(2):
                            vs = v_local[s, lr, lc]
                            if vs == 0:
                                continue
                            c_row = 2 * cbr + lr
                            c_col = 2 * cbc + lc
                            W[c_col * n + c_row, r] += wi * vs

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _verify_decomposition(U, V, W, n, m, p, rank):
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)

    U_3d = U_arr.reshape(n, m, rank)
    V_3d = V_arr.reshape(m, p, rank)
    W_3d = W_arr.reshape(p, n, rank)

    result = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
    return bool(np.allclose(result, 1.0, atol=1e-10))

def construct():
    deadline = time.time() + 140.0
    n, m, p = DIMENSIONS

    result = _strassen_4x4_decomposition_optimized()
    U, V, W = result["U"], result["V"], result["W"]
    rank = len(U[0])

    if _verify_decomposition(U, V, W, n, m, p, rank):
        return result

    # Fallback
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
