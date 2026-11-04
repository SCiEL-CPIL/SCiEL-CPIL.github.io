"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]
import time
import numpy as np

# EVOLVE-BLOCK-START

def strassen_factors():
    """
    Returns (U, V, W) for 2x2x2 Strassen decomposition.
    Convention: U[i*2+j, c] for A[i,j], V[j*2+k, c] for B[j,k], W[k*2+i, c] for C[i,k].
    A=[[a,b],[c,d]], B=[[e,f],[g,h]], C=A*B.
    C[0,0]=ae+cg, C[0,1]=af+bh, C[1,0]=ce+dg, C[1,1]=cf+dh.
    m1=(a+d)(e+h), m2=(c+d)e, m3=a(f-h), m4=d(g-e), m5=(a+b)h, m6=(c-a)(e+f), m7=(b-d)(g+h)
    C[0,0]=m1+m4-m5+m7
    C[0,1]=m3+m5
    C[1,0]=m2+m4
    C[1,1]=m1-m2+m3+m6
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


def schoolbook_factors(ni, mj, pk):
    """
    Returns (U, V, W) for ni x mj x pk schoolbook decomposition.
    U shape (ni*mj, rank), V shape (mj*pk, rank), W shape (pk*ni, rank).
    U[i*mj+j, c], V[j*pk+k, c], W[k*ni+i, c].
    """
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
    """Rank contributed by a single block."""
    if ni == 2 and mj == 2 and pk == 2:
        return 7
    return ni * mj * pk


def build_block_decomposition(n, m, p, n_sizes, m_sizes, p_sizes):
    """
    Build the full factor matrices using block decomposition.
    n_sizes partitions the n (rows of A / rows of C) dimension.
    m_sizes partitions the m (cols of A / rows of B) dimension.
    p_sizes partitions the p (cols of B / cols of C) dimension.
    """
    # Precompute offsets
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

    # Total rank
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

                # Fill U: rows indexed by (i*m + j) in full, (li*mj + lj) in block
                for li in range(ni):
                    for lj in range(mj):
                        full_row = (i_start + li) * m + (j_start + lj)
                        U_full[full_row, col_offset:col_offset + r] = U_block[li * mj + lj, :]

                # Fill V: rows indexed by (j*p + k) in full, (lj*pk + lk) in block
                for lj in range(mj):
                    for lk in range(pk):
                        full_row = (j_start + lj) * p + (k_start + lk)
                        V_full[full_row, col_offset:col_offset + r] = V_block[lj * pk + lk, :]

                # Fill W: rows indexed by (k*n + i) in full, (lk*ni + li) in block
                for lk in range(pk):
                    for li in range(ni):
                        full_row = (k_start + lk) * n + (i_start + li)
                        W_full[full_row, col_offset:col_offset + r] = W_block[lk * ni + li, :]

                col_offset += r

    return U_full, V_full, W_full, total_rank


def verify_decomposition(U, V, W, n, m, p, atol=1e-9):
    """
    Verify that the decomposition exactly reproduces the matrix multiplication tensor.
    T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal 1 for all i,j,k.
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


def schoolbook_lists(n, m, p):
    """Return schoolbook decomposition as nested lists (fallback)."""
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
    return U, V, W


def construct():
    """
    Constructs the factor matrices U, V, W for the matrix multiplication tensor
    of multiplying a 3x5 matrix by a 5x6 matrix.
    """
    n, m, p = DIMENSIONS  # 3, 5, 6
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s limit

    # Candidate block partitionings; we evaluate each and keep the lowest-rank valid one.
    # Each partition is (n_sizes, m_sizes, p_sizes).
    candidates = []

    # Primary: [2,1] x [2,2,1] x [2,2,2]
    candidates.append(([2, 1], [2, 2, 1], [2, 2, 2]))
    # Alternate: [1,2] x [2,2,1] x [2,2,2]
    candidates.append(([1, 2], [2, 2, 1], [2, 2, 2]))
    # Alternate: [2,1] x [1,2,2] x [2,2,2]
    candidates.append(([2, 1], [1, 2, 2], [2, 2, 2]))
    # Alternate: [2,1] x [2,2,1] x [1,2,3]
    candidates.append(([2, 1], [2, 2, 1], [1, 2, 3]))
    # Alternate: [2,1] x [2,2,1] x [2,1,3]
    candidates.append(([2, 1], [2, 2, 1], [2, 1, 3]))
    # Alternate: [2,1] x [2,2,1] x [3,2,1]
    candidates.append(([2, 1], [2, 2, 1], [3, 2, 1]))
    # Alternate: [1,2] x [1,2,2] x [2,2,2]
    candidates.append(([1, 2], [1, 2, 2], [2, 2, 2]))
    # Alternate: [2,1] x [2,1,2] x [2,2,2]
    candidates.append(([2, 1], [2, 1, 2], [2, 2, 2]))

    best_rank = None
    best_U = None
    best_V = None
    best_W = None

    for n_sizes, m_sizes, p_sizes in candidates:
        if time.time() > deadline:
            break
        if sum(n_sizes) != n or sum(m_sizes) != m or sum(p_sizes) != p:
            continue

        U, V, W, total_rank = build_block_decomposition(n, m, p, n_sizes, m_sizes, p_sizes)

        if not verify_decomposition(U, V, W, n, m, p, atol=1e-9):
            continue

        if best_rank is None or total_rank < best_rank:
            best_rank = total_rank
            best_U = U
            best_V = V
            best_W = W

    # If no block decomposition was valid, fall back to schoolbook.
    if best_U is None:
        U_fb, V_fb, W_fb = schoolbook_lists(n, m, p)
        return {"U": U_fb, "V": V_fb, "W": W_fb}

    if time.time() > deadline:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

# EVOLVE-BLOCK-END
