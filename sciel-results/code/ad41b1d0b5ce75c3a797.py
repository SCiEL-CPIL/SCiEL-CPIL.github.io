"""Matrix multiplication 4x4x4 tensor decomposition using recursive Strassen (rank 49)."""
import numpy as np
import time

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START

# Strassen's 7 products for 2x2 matrices.
# For A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = A*B:
# C00 = a*e + c*h
# C01 = b*h
# C10 = a*f + d*g
# C11 = b*g + d*h
#
# Strassen products:
# P1 = (a+d)(e+h)
# P2 = (c+d)e
# P3 = a(f-h)
# P4 = d(g-e)
# P5 = (a+b)h
# P6 = (c-a)(e+f)
# P7 = (b-d)(g+h)
#
# C00 = P1 + P4 - P5 + P7
# C01 = P3 + P5
# C10 = P2 + P4
# C11 = P1 - P2 + P3 + P6

_STRASSEN_2X2 = [
    # P1 = (a+d)(e+h)
    ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
    # P2 = (c+d)e
    ([(1,0,1),(1,1,1)], [(0,0,1)]),
    # P3 = a(f-h)
    ([(0,0,1)], [(0,1,1),(1,1,-1)]),
    # P4 = d(g-e)
    ([(1,1,1)], [(1,0,1),(0,0,-1)]),
    # P5 = (a+b)h
    ([(0,0,1),(0,1,1)], [(1,1,1)]),
    # P6 = (c-a)(e+f)
    ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)]),
    # P7 = (b-d)(g+h)
    ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)]),
]

_STRASSEN_C_COMBINATION = [
    # (c_row, c_col, list of (product_idx, coeff))
    (0, 0, [(0, 1), (3, 1), (4, -1), (6, 1)]),
    (0, 1, [(2, 1), (4, 1)]),
    (1, 0, [(1, 1), (3, 1)]),
    (1, 1, [(0, 1), (1, -1), (2, 1), (5, 1)]),
]


def _strassen_4x4_decomposition():
    """
    Build rank-49 decomposition for 4x4x4 matrix multiplication using
    recursive Strassen: apply Strassen at the 4x4 level (7 block products),
    then apply Strassen within each 2x2 sub-block product (7 scalar mults each).
    Total rank = 7 * 7 = 49.
    """
    n, m, p = 4, 4, 4
    rank = 49

    U = [[0] * rank for _ in range(n * m)]  # shape (16, 49)
    V = [[0] * rank for _ in range(m * p)]  # shape (16, 49)
    W = [[0] * rank for _ in range(p * n)]  # shape (16, 49)

    # The 7 block-level Strassen products for 4x4:
    block_products = [
        ([(0, 0, 1), (1, 1, 1)], [(0, 0, 1), (1, 1, 1)]),       # M1
        ([(1, 0, 1), (1, 1, 1)], [(0, 0, 1)]),                   # M2
        ([(0, 0, 1)], [(0, 1, 1), (1, 1, -1)]),                  # M3
        ([(1, 1, 1)], [(1, 0, 1), (0, 0, -1)]),                  # M4
        ([(0, 0, 1), (0, 1, 1)], [(1, 1, 1)]),                   # M5
        ([(1, 0, 1), (0, 0, -1)], [(0, 0, 1), (0, 1, 1)]),       # M6
        ([(0, 1, 1), (1, 1, -1)], [(1, 0, 1), (1, 1, 1)]),       # M7
    ]

    # C block combination: C[cbr][cbc] = sum_i w_i * Mi
    c_block_combination = [
        (0, 0, [(0, 1), (3, 1), (4, -1), (6, 1)]),
        (0, 1, [(2, 1), (4, 1)]),
        (1, 0, [(1, 1), (3, 1)]),
        (1, 1, [(0, 1), (1, -1), (2, 1), (5, 1)]),
    ]

    # Precompute: for each block product i, w_i(cbr,cbc) for each C-block
    w_block = [[[0]*4 for _ in range(2)] for _ in range(7)]
    for (cbr, cbc, c_list) in c_block_combination:
        for (pi, wi) in c_list:
            w_block[pi][cbr][cbc] = wi

    # Precompute: for each Strassen product s, v_s(lr,lc) for each C-local entry
    v_local = [[[0]*2 for _ in range(2)] for _ in range(7)]
    for (lr, lc, v_list) in _STRASSEN_C_COMBINATION:
        for (si, vs) in v_list:
            v_local[si][lr][lc] = vs

    # Build U, V, W
    for i in range(7):
        a_combo, b_combo = block_products[i]
        for s in range(7):
            r = i * 7 + s
            a_local, b_local = _STRASSEN_2X2[s]

            # U: for each (a_br, a_bc, a_coeff) in a_combo and (lr, lc, l_coeff) in a_local
            for (a_br, a_bc, a_coeff) in a_combo:
                for (lr, lc, l_coeff) in a_local:
                    row = 2 * a_br + lr
                    col = 2 * a_bc + lc
                    U[row * m + col][r] += a_coeff * l_coeff

            # V: for each (b_br, b_bc, b_coeff) in b_combo and (lr, lc, l_coeff) in b_local
            for (b_br, b_bc, b_coeff) in b_combo:
                for (lr, lc, l_coeff) in b_local:
                    row = 2 * b_br + lr
                    col = 2 * b_bc + lc
                    V[row * p + col][r] += b_coeff * l_coeff

            # W: C[2*cbr+lr][2*cbc+lc] gets w_block[i][cbr][cbc] * v_local[s][lr][lc]
            for cbr in range(2):
                for cbc in range(2):
                    wi = w_block[i][cbr][cbc]
                    if wi == 0:
                        continue
                    for lr in range(2):
                        for lc in range(2):
                            vs = v_local[s][lr][lc]
                            if vs == 0:
                                continue
                            c_row = 2 * cbr + lr
                            c_col = 2 * cbc + lc
                            W[c_col * n + c_row][r] += wi * vs

    return {"U": U, "V": V, "W": W}


def _verify_decomposition(U, V, W, n, m, p, rank):
    """Verify that the decomposition is correct using numpy for speed."""
    U_arr = np.array(U, dtype=np.float64)  # (n*m, rank)
    V_arr = np.array(V, dtype=np.float64)  # (m*p, rank)
    W_arr = np.array(W, dtype=np.float64)  # (p*n, rank)
    
    # Reshape to 3D: U_3d[i,j,r], V_3d[j,k,r], W_3d[k,i,r]
    U_3d = U_arr.reshape(n, m, rank)
    V_3d = V_arr.reshape(m, p, rank)
    W_3d = W_arr.reshape(p, n, rank)
    
    # Contract over r: result[i,j,k] = sum_r U_3d[i,j,r] * V_3d[j,k,r] * W_3d[k,i,r]
    result = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
    
    # All entries should be 1
    return bool(np.allclose(result, 1.0, atol=1e-10))


def _schoolbook_decomposition(n, m, p):
    """Fallback schoolbook rank n*m*p decomposition."""
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


def construct():
    deadline = time.time() + 140.0
    n, m, p = DIMENSIONS

    # Try the recursive Strassen rank-49 decomposition
    t0 = time.time()
    result = _strassen_4x4_decomposition()
    U, V, W = result["U"], result["V"], result["W"]
    rank = len(U[0])

    # Verify correctness
    if _verify_decomposition(U, V, W, n, m, p, rank):
        return result

    # Fallback to schoolbook
    return _schoolbook_decomposition(n, m, p)

# EVOLVE-BLOCK-END
