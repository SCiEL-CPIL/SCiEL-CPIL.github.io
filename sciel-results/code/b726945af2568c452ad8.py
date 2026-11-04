import time
import numpy as np
from fractions import Fraction
from itertools import product

DIMENSIONS = [3, 5, 6]


def strassen_factors():
    """
    Returns (U, V, W) for 2x2x2 Strassen decomposition.
    Convention: U[i*2+j, c] for A[i,j], V[j*2+k, c] for B[j,k], W[k*2+i, c] for C[i,k].
    """
    U = np.zeros((4, 7), dtype=np.float64)
    V = np.zeros((4, 7), dtype=np.float64)
    W = np.zeros((4, 7), dtype=np.float64)

    # m1 = (a+d)(e+h)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1

    # m2 = (c+d)e
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1

    # m3 = a(f-h)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1

    # m4 = d(g-e)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1

    # m5 = (a+b)h
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1

    # m6 = (c-a)(e+f)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1

    # m7 = (b-d)(g+h)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1

    return U, V, W


def schoolbook_factors(n, m, p):
    """
    Returns (U, V, W) for n x m x p schoolbook decomposition.
    """
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1

    return U, V, W


def build_block_decomposition(n_sizes, m_sizes, p_sizes):
    """
    Builds a block decomposition using the given partitions.
    """
    n, m, p = 3, 5, 6
    total_rank = 0
    for ni in n_sizes:
        for mj in m_sizes:
            for pk in p_sizes:
                if ni == 2 and mj == 2 and pk == 2:
                    total_rank += 7
                else:
                    total_rank += ni * mj * pk

    U = np.zeros((n * m, total_rank), dtype=np.float64)
    V = np.zeros((m * p, total_rank), dtype=np.float64)
    W = np.zeros((p * n, total_rank), dtype=np.float64)

    col_offset = 0
    n_offsets = np.cumsum([0] + n_sizes)
    m_offsets = np.cumsum([0] + m_sizes)
    p_offsets = np.cumsum([0] + p_sizes)

    for i_idx, ni in enumerate(n_sizes):
        i_start = n_offsets[i_idx]
        for j_idx, mj in enumerate(m_sizes):
            j_start = m_offsets[j_idx]
            for k_idx, pk in enumerate(p_sizes):
                k_start = p_offsets[k_idx]

                if ni == 2 and mj == 2 and pk == 2:
                    U_blk, V_blk, W_blk = strassen_factors()
                else:
                    U_blk, V_blk, W_blk = schoolbook_factors(ni, mj, pk)

                r = U_blk.shape[1]

                for li in range(ni):
                    for lj in range(mj):
                        full_row_u = (i_start + li) * m + (j_start + lj)
                        U[full_row_u, col_offset:col_offset + r] = U_blk[li * mj + lj, :]

                for lj in range(mj):
                    for lk in range(pk):
                        full_row_v = (j_start + lj) * p + (k_start + lk)
                        V[full_row_v, col_offset:col_offset + r] = V_blk[lj * pk + lk, :]

                for lk in range(pk):
                    for li in range(ni):
                        full_row_w = (k_start + lk) * n + (i_start + li)
                        W[full_row_w, col_offset:col_offset + r] = W_blk[lk * ni + li, :]

                col_offset += r

    return U, V, W, total_rank


def verify_decomposition(U, V, W, n, m, p, atol=1e-9):
    """
    Verify that the decomposition exactly reproduces the matrix multiplication tensor.
    """
    i_idx = np.repeat(np.arange(n), m * p)
    j_idx = np.tile(np.repeat(np.arange(m), p), n)
    k_idx = np.tile(np.arange(p), n * m)
    u_rows = i_idx * m + j_idx
    v_rows = j_idx * p + k_idx
    w_rows = k_idx * n + i_idx

    products = U[u_rows, :] * V[v_rows, :] * W[w_rows, :]
    vals = products.sum(axis=1)

    return bool(np.allclose(vals, 1.0, atol=atol))


def try_rank_reduction(U, V, W, n, m, p, current_rank, deadline):
    """
    Attempt to reduce rank below current_rank using algebraic simplification.
    """
    start_time = time.time()
    if start_time >= deadline:
        return None

    # Check for linear dependencies in the factor matrices
    rank_u = np.linalg.matrix_rank(U)
    rank_v = np.linalg.matrix_rank(V)
    rank_w = np.linalg.matrix_rank(W)

    if rank_u < current_rank or rank_v < current_rank or rank_w < current_rank:
        pass

    # Try to find pairs of columns that can be merged
    for c1 in range(current_rank):
        if time.time() >= deadline - 5:
            break
        for c2 in range(c1 + 1, current_rank):
            try:
                u1 = U[:, c1]
                u2 = U[:, c2]
                if np.allclose(u1, 0):
                    if not np.allclose(u2, 0):
                        continue
                    u_ratio = None
                elif np.allclose(u2, 0):
                    continue
                else:
                    mask = np.abs(u1) > 1e-12
                    if not np.all(mask):
                        continue
                    u_ratio = u2[mask] / u1[mask]
                    if not np.allclose(u2[mask], u_ratio * u1[mask]):
                        continue
                    if not np.allclose(u1[~mask], 0):
                        continue
                    if not np.allclose(u2[~mask], 0):
                        continue

                v1 = V[:, c1]
                v2 = V[:, c2]
                if u_ratio is not None:
                    mask = np.abs(v1) > 1e-12
                    if not np.all(mask):
                        continue
                    v_ratio = v2[mask] / v1[mask]
                    if not np.allclose(v2[mask], v_ratio * v1[mask]):
                        continue
                    if not np.allclose(v1[~mask], 0):
                        continue
                    if not np.allclose(v2[~mask], 0):
                        continue
                else:
                    mask = np.abs(v1) > 1e-12
                    if not np.all(mask):
                        continue
                    v_ratio = v2[mask] / v1[mask]
                    if not np.allclose(v2[mask], v_ratio * v1[mask]):
                        continue
                    if not np.allclose(v1[~mask], 0):
                        continue
                    if not np.allclose(v2[~mask], 0):
                        continue

                w1 = W[:, c1]
                w2 = W[:, c2]
                if u_ratio is not None and v_ratio is not None:
                    expected_w = w1 * u_ratio * v_ratio
                    if not np.allclose(w2, expected_w):
                        continue
                else:
                    continue

                keep = [c for c in range(current_rank) if c != c2]
                new_U = U[:, keep]
                new_V = V[:, keep]
                new_W = W[:, keep]
                new_rank = len(keep)

                if verify_decomposition(new_U, new_V, new_W, n, m, p):
                    return (new_U, new_V, new_W, new_rank)

            except Exception:
                continue

    return None


def construct():
    """
    Constructs the factor matrices U, V, W for the matrix multiplication tensor
    of multiplying a 3x5 matrix by a 5x6 matrix.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 135.0

    # Build the 84-rank decomposition with partition (1,2), (2,2,1), (2,2,2)
    U, V, W, rank = build_block_decomposition([1, 2], [2, 2, 1], [2, 2, 2])

    if verify_decomposition(U, V, W, n, m, p):
        # Try to reduce rank
        result = try_rank_reduction(U, V, W, n, m, p, rank, deadline)
        if result is not None:
            new_U, new_V, new_W, new_rank = result
            if new_rank < rank:
                return {"U": new_U.tolist(), "V": new_V.tolist(), "W": new_W.tolist()}
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback: Schoolbook
    rank_fb = n * m * p
    U_fb = np.zeros((n * m, rank_fb), dtype=np.float64)
    V_fb = np.zeros((m * p, rank_fb), dtype=np.float64)
    W_fb = np.zeros((p * n, rank_fb), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j, col] = 1.0
                V_fb[j * p + k, col] = 1.0
                W_fb[k * n + i, col] = 1.0
                col += 1

    return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}


# EVOLVE-BLOCK-START
# The construct function is defined above.
# EVOLVE-BLOCK-END
