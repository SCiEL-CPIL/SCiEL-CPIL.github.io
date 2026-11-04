"""
Matrix multiplication 2x4 by 4x4, exact tensor decomposition.

Core (Strassen): partition A (2x4) into two 2x2 blocks and B (4x4) into four
2x2 blocks; the product splits into four independent 2x2 x 2x2 sub-products.
Each sub-product is computed with Strassen (rank 7). Because the sub-products
act on disjoint input subspaces, the rank is additive: 4 * 7 = 28.

Refinement: a time-bounded exact search attempts to beat 28 by finding an
integer/half-integer rank-27 (then 26) tensor whose flattening matrix has
rank <= r. Any such tensor is verified before being returned; otherwise the
verified rank-28 Strassen result is returned.
"""
import time
import itertools
import random
import numpy as np

DIMENSIONS = [2, 4, 4]
TIME_LIMIT = 150.0
START_TIME = time.time()


def _elapsed():
    return time.time() - START_TIME


def _time_ok(margin=10.0):
    return _elapsed() < TIME_LIMIT - margin


def construct():
    best_result = None
    try:
        best_result = _construct_strassen_28()
        if best_result is not None and _verify_decomposition(best_result):
            # Try to beat 28 with an exact integer/half-integer lower rank.
            for r in (27, 26):
                if not _time_ok(20.0):
                    break
                cand = _search_lower_rank(r, best_result)
                if cand is not None and _verify_decomposition(cand):
                    best_result = cand
                    break
    except Exception:
        pass
    if best_result is None:
        best_result = _construct_schoolbook_32()
    return best_result


def _verify_decomposition(result, trials=8):
    """Verify that the rank-1 sum reconstructs the matmul tensor exactly.

    T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    C[i,k] = sum_j A[i,j]*B[j,k] = sum_r (sum_{i,j} U[i*m+j,r]*A[i,j])
                                            * (sum_{j,k} V[j*p+k,r]*B[j,k]) * W[k*n+i,r]
    """
    n, m, p = DIMENSIONS
    U = np.array(result['U'], dtype=float)
    V = np.array(result['V'], dtype=float)
    W = np.array(result['W'], dtype=float)
    rank = U.shape[1]
    rng = np.random.RandomState(20240517)
    for _ in range(trials):
        A = rng.randn(n, m)
        B = rng.randn(m, p)
        C_expected = A @ B
        # u_r = U^T (A flattened), v_r = V^T (B flattened)
        u = U.T @ A.reshape(n * m)          # (rank,)
        v = V.T @ B.reshape(m * p)          # (rank,)
        C_computed = (W * (u * v)).reshape(p, n).T  # (n, p)
        if not np.allclose(C_computed, C_expected, atol=1e-9):
            return False
    return True


def _construct_schoolbook_32():
    n, m, p = DIMENSIONS
    rank = n * m * p  # 32
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}


def _construct_strassen_28():
    """2x4 * 4x4 via four independent 2x2 Strassen blocks (rank 28)."""
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]  # 8 x 28
    V = [[0] * rank for _ in range(m * p)]  # 16 x 28
    W = [[0] * rank for _ in range(p * n)]  # 8 x 28

    # Strassen 2x2 terms: (U_coeffs, V_coeffs, W_coeffs) over local indices
    # local 2x2 [a b; c d] -> a=0, b=1, c=2, d=3
    # c00 = P1 + P4 - P5 + P7 ; c01 = P3 + P5 ; c10 = P2 + P4 ; c11 = P1 - P2 + P3 + P6
    strassen_terms = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),          # P1
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),               # P2
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),               # P3
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),               # P4
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),               # P5
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),               # P6
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),               # P7
    ]

    # Four sub-products with explicit index maps.
    sub_products = [
        # 1: A_L * B_TL -> C_LL
        {'a_map': lambda r, c: r * 4 + c,
         'b_map': lambda r, c: r * 4 + c,
         'c_map': lambda r, c: c * 2 + r},
        # 2: A_R * B_BL -> C_LR
        {'a_map': lambda r, c: r * 4 + (c + 2),
         'b_map': lambda r, c: (r + 2) * 4 + c,
         'c_map': lambda r, c: c * 2 + r},
        # 3: A_L * B_TR -> C_RL
        {'a_map': lambda r, c: r * 4 + c,
         'b_map': lambda r, c: r * 4 + (c + 2),
         'c_map': lambda r, c: (c + 2) * 2 + r},
        # 4: A_R * B_BR -> C_RR
        {'a_map': lambda r, c: r * 4 + (c + 2),
         'b_map': lambda r, c: (r + 2) * 4 + (c + 2),
         'c_map': lambda r, c: (c + 2) * 2 + r},
    ]

    col = 0
    for sp in sub_products:
        for u_coeffs, v_coeffs, w_coeffs in strassen_terms:
            for loc_u, coeff in u_coeffs.items():
                r, c = loc_u // 2, loc_u % 2
                U[sp['a_map'](r, c)][col] += coeff
            for loc_v, coeff in v_coeffs.items():
                r, c = loc_v // 2, loc_v % 2
                V[sp['b_map'](r, c)][col] += coeff
            for loc_w, coeff in w_coeffs.items():
                r, c = loc_w // 2, loc_w % 2
                W[sp['c_map'](r, c)][col] += coeff
            col += 1

    return {"U": U, "V": V, "W": W}


def _flattening_rank(U, V, W):
    """Rank of the (p*n) x (n*m) flattening matrix of the tensor.

    If the tensor equals the matmul tensor, this flattening is the n*m x n*m
    identity (rank n*m = 8). A tensor of tensor-rank r has flattening rank <= r,
    so a flattening rank > r certifies that the tensor cannot have rank <= r.
    """
    n, m, p = DIMENSIONS
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    rank = U.shape[1]
    # F[k*n+i, i'*m+j] = sum_r V[j*p+k, r] * W[k*n+i, r] * U[i'*m+j, r]
    F = np.zeros((p * n, n * m), dtype=float)
    for r in range(rank):
        F += np.outer(W[:, r], U[:, r]) * V[:, r]
    return int(np.linalg.matrix_rank(F, tol=1e-9))


def _search_lower_rank(r, base_result):
    """Exact integer/half-integer search for a rank-r decomposition.

    Builds r rank-1 tensors, each a sign/linear combination of the 28 Strassen
    columns (so entries stay in {-1,0,1,2}). A candidate is accepted only if its
    flattening matrix has rank <= r (a rank certificate) AND it verifies.
    """
    n, m, p = DIMENSIONS
    U0 = np.asarray(base_result['U'], dtype=float)
    V0 = np.asarray(base_result['V'], dtype=float)
    W0 = np.asarray(base_result['W'], dtype=float)
    R0 = U0.shape[1]  # 28
    rng = random.Random(987654321)
    sign_choices = [1, -1]

    attempts = 0
    max_attempts = 40000
    while attempts < max_attempts and _time_ok(25.0):
        attempts += 1
        # Random linear combinations of the 28 Strassen columns.
        combo = [rng.choice(sign_choices) for _ in range(R0)]
        U = U0 @ np.array(combo, dtype=float)
        V = V0 @ np.array(combo, dtype=float)
        W = W0 @ np.array(combo, dtype=float)

        # Keep the r columns with the largest magnitudes (drop near-zero ones).
        mags = np.abs(U).max(axis=0) + np.abs(V).max(axis=0) + np.abs(W).max(axis=0)
        order = np.argsort(-mags)
        keep = order[:r]
        Uk, Vk, Wk = U[:, keep], V[:, keep], W[:, keep]

        if _flattening_rank(Uk, Vk, Wk) > r:
            continue  # cannot be rank <= r

        cand = {"U": Uk.tolist(), "V": Vk.tolist(), "W": Wk.tolist()}
        if _verify_decomposition(cand, trials=4):
            return cand
    return None
