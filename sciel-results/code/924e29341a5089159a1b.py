"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]
import time
import numpy as np

# EVOLVE-BLOCK-START

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
    W[2, 1] = 1; W[3, 1] = -1

    # m3 = a(f-h)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1

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


def schoolbook_factors(ni, mj, pk):
    rank = ni * mj * pk
    U = np.zeros((ni * mj, rank), dtype=np.float64)
    V = np.zeros((mj * pk, rank), dtype=np.float64)
    W = np.zeros((pk * ni, rank), dtype=np.float64)
    col = 0
    for i in range(ni):
        for j in range(mj):
            for k in range(pk):
                U[i * mj + j, col] = 1.0
                V[j * pk + k, col] = 1.0
                W[k * ni + i, col] = 1.0
                col += 1
    return U, V, W


def block_rank(ni, mj, pk):
    if ni == 2 and mj == 2 and pk == 2:
        return 7
    return ni * mj * pk


def build_block_decomposition(n, m, p, n_sizes, m_sizes, p_sizes):
    n_offsets = []
    off = 0
    for s in n_sizes:
        n_offsets.append(off)
        off += s
    m_offsets = []
    off = 0
    for s in m_sizes:
        m_offsets.append(off)
        off += s
    p_offsets = []
    off = 0
    for s in p_sizes:
        p_offsets.append(off)
        off += s

    total_rank = 0
    for ni in n_sizes:
        for mj in m_sizes:
            for pk in p_sizes:
                total_rank += block_rank(ni, mj, pk)

    U_full = np.zeros((n * m, total_rank), dtype=np.float64)
    V_full = np.zeros((m * p, total_rank), dtype=np.float64)
    W_full = np.zeros((p * n, total_rank), dtype=np.float64)

    col_offset = 0
    for i_idx, ni in enumerate(n_sizes):
        i_start = n_offsets[i_idx]
        for j_idx, mj in enumerate(m_sizes):
            j_start = m_offsets[j_idx]
            for k_idx, pk in enumerate(p_sizes):
                k_start = p_offsets[k_idx]

                if ni == 2 and mj == 2 and pk == 2:
                    U_block, V_block, W_block = strassen_factors()
                else:
                    U_block, V_block, W_block = schoolbook_factors(ni, mj, pk)
                r = U_block.shape[1]

                for li in range(ni):
                    for lj in range(mj):
                        full_row = (i_start + li) * m + (j_start + lj)
                        U_full[full_row, col_offset:col_offset + r] = U_block[li * mj + lj, :]

                for lj in range(mj):
                    for lk in range(pk):
                        full_row = (j_start + lj) * p + (k_start + lk)
                        V_full[full_row, col_offset:col_offset + r] = V_block[lj * pk + lk, :]

                for lk in range(pk):
                    for li in range(ni):
                        full_row = (k_start + lk) * n + (i_start + li)
                        W_full[full_row, col_offset:col_offset + r] = W_block[lk * ni + li, :]

                col_offset += r

    return U_full, V_full, W_full, total_rank


def verify_decomposition(U, V, W, n, m, p, atol=1e-9):
    i_idx = np.repeat(np.arange(n), m * p)
    j_idx = np.tile(np.repeat(np.arange(m), p), n)
    k_idx = np.tile(np.arange(p), n * m)
    u_rows = i_idx * m + j_idx
    v_rows = j_idx * p + k_idx
    w_rows = k_idx * n + i_idx

    products = U[u_rows, :] * V[v_rows, :] * W[w_rows, :]
    vals = products.sum(axis=1)

    return bool(np.allclose(vals, 1.0, atol=atol))


def all_partitions(total, max_part=6):
    """Generate all integer partitions of total into parts >= 1, ordered."""
    if total == 0:
        yield []
        return
    for first in range(1, total + 1):
        for rest in all_partitions(total - first, max_part):
            yield [first] + rest


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0

    candidates = []
    # Generate all partitions
    for n_sizes in all_partitions(n):
        for m_sizes in all_partitions(m):
            for p_sizes in all_partitions(p):
                candidates.append((n_sizes, m_sizes, p_sizes))

    best_rank = None
    best_U = None
    best_V = None
    best_W = None

    for n_sizes, m_sizes, p_sizes in candidates:
        if time.time() > deadline:
            break
        U, V, W, total_rank = build_block_decomposition(n, m, p, n_sizes, m_sizes, p_sizes)
        if not verify_decomposition(U, V, W, n, m, p, atol=1e-9):
            continue
        if best_rank is None or total_rank < best_rank:
            best_rank = total_rank
            best_U = U
            best_V = V
            best_W = W

    if best_U is None:
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

# EVOLVE-BLOCK-END
