"""Recursive Strassen decomposition of 4x4x4 matrix multiplication (rank 49) with rank reduction."""
import time
import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START

# Strassen's 7 products for 2x2 matrices.
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

# How each C entry is formed from the 7 Strassen products:
_STRASSEN_C_COMBINATION = [
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

    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)

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
                    U[row * m + col, r] += a_coeff * l_coeff

            # V: for each (b_br, b_bc, b_coeff) in b_combo and (lr, lc, l_coeff) in b_local
            for (b_br, b_bc, b_coeff) in b_combo:
                for (lr, lc, l_coeff) in b_local:
                    row = 2 * b_br + lr
                    col = 2 * b_bc + lc
                    V[row * p + col, r] += b_coeff * l_coeff

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
                            W[c_col * n + c_row, r] += wi * vs

    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()

    return {"U": U_list, "V": V_list, "W": W_list}


def _verify_decomposition(U, V, W, n, m, p, rank):
    """Verify that the decomposition is correct using numpy for speed."""
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    U_3d = U_arr.reshape(n, m, rank)
    V_3d = V_arr.reshape(m, p, rank)
    W_3d = W_arr.reshape(p, n, rank)
    
    result = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
    
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


def _rank_reduce(U, V, W, n, m, p, deadline):
    """
    Attempt to reduce rank by finding linearly dependent bilinear terms.
    """
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    tensor_dim = n * m * p
    
    M = np.zeros((tensor_dim, rank), dtype=np.float64)
    for r in range(rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx = i * m * p + j * p + k
                    M[idx, r] = U_arr[i * m + j, r] * V_arr[j * p + k, r] * W_arr[k * n + i, r]
    
    try:
        U_svd, s, Vt_svd = np.linalg.svd(M, full_matrices=False)
    except:
        return None
    
    tol = 1e-10
    num_rank = np.sum(s > tol)
    
    if num_rank >= rank:
        return None
    
    null_space = Vt_svd[num_rank:, :].T
    nullity = rank - num_rank
    
    kept = list(range(rank))
    removed = []
    
    norms = np.linalg.norm(M, axis=0)
    order = np.argsort(norms)
    
    for r in order:
        if time.time() > deadline:
            break
        if r not in kept:
            continue
        others = [k for k in kept if k != r]
        if not others:
            break
        try:
            x, residuals, rank_m, s_vals = np.linalg.lstsq(M[:, others], M[:, r], rcond=None)
            residual = np.linalg.norm(M[:, r] - M[:, others] @ x)
            if residual < 1e-8:
                kept.remove(r)
                removed.append(r)
        except:
            pass
    
    if not removed:
        return None
    
    new_rank = len(kept)
    U_new = U_arr[:, kept].tolist()
    V_new = V_arr[:, kept].tolist()
    W_new = W_arr[:, kept].tolist()
    
    if _verify_decomposition(U_new, V_new, W_new, n, m, p, new_rank):
        return {"U": U_new, "V": V_new, "W": W_new}
    
    return None


def construct():
    deadline = time.time() + 145.0
    n, m, p = DIMENSIONS

    result = _strassen_4x4_decomposition()
    U, V, W = result["U"], result["V"], result["W"]
    rank = len(U[0])

    if not _verify_decomposition(U, V, W, n, m, p, rank):
        return _schoolbook_decomposition(n, m, p)

    if time.time() < deadline - 5.0:
        reduced = _rank_reduce(U, V, W, n, m, p, deadline - 3.0)
        if reduced is not None:
            return reduced

    return result

# EVOLVE-BLOCK-END
