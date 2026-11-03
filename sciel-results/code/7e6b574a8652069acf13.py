import time
import numpy as np

DIMENSIONS = [3, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0

    # Partition: n=[1,2], m=[2,2,1], p=[2,2,2]
    # This gives 8 Strassen 2x2x2 blocks (rank 7 each) + schoolbook remainder
    # Total rank: 84

    n_sizes = [1, 2]
    m_sizes = [2, 2, 1]
    p_sizes = [2, 2, 2]
    n_offsets = [0, 1]
    m_offsets = [0, 2, 4]
    p_offsets = [0, 2, 4]

    # Calculate total rank
    total_rank = 0
    for ni in n_sizes:
        for mj in m_sizes:
            for pk in p_sizes:
                if ni == 2 and mj == 2 and pk == 2:
                    total_rank += 7
                else:
                    total_rank += ni * mj * pk

    assert total_rank == 84, f"Expected 84, got {total_rank}"

    # Initialize matrices
    U = np.zeros((n * m, total_rank), dtype=np.float64)
    V = np.zeros((m * p, total_rank), dtype=np.float64)
    W = np.zeros((p * n, total_rank), dtype=np.float64)

    col = 0

    # Strassen 2x2x2 factors
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)

    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    # For each block
    for i_idx, ni in enumerate(n_sizes):
        i_start = n_offsets[i_idx]
        for j_idx, mj in enumerate(m_sizes):
            j_start = m_offsets[j_idx]
            for k_idx, pk in enumerate(p_sizes):
                k_start = p_offsets[k_idx]

                if ni == 2 and mj == 2 and pk == 2:
                    # Strassen 2x2x2
                    # A indices: (i_start, j_start), (i_start, j_start+1), (i_start+1, j_start), (i_start+1, j_start+1)
                    # B indices: (j_start, k_start), (j_start, k_start+1), (j_start+1, k_start), (j_start+1, k_start+1)
                    # C indices: (i_start, k_start), (i_start, k_start+1), (i_start+1, k_start), (i_start+1, k_start+1)

                    # M1 = (A00+A11)(B00+B11)
                    U[i_start * m + j_start, col] = 1
                    U[(i_start + 1) * m + (j_start + 1), col] = 1
                    V[j_start * p + k_start, col] = 1
                    V[(j_start + 1) * p + (k_start + 1), col] = 1
                    W[k_start * n + i_start, col] = 1
                    W[(k_start + 1) * n + (i_start + 1), col] = 1
                    col += 1

                    # M2 = (A10+A11)B00
                    U[(i_start + 1) * m + j_start, col] = 1
                    U[(i_start + 1) * m + (j_start + 1), col] = 1
                    V[j_start * p + k_start, col] = 1
                    W[k_start * n + (i_start + 1), col] = 1
                    W[(k_start + 1) * n + (i_start + 1), col] = -1
                    col += 1

                    # M3 = A00(B01-B11)
                    U[i_start * m + j_start, col] = 1
                    V[j_start * p + (k_start + 1), col] = 1
                    V[(j_start + 1) * p + (k_start + 1), col] = -1
                    W[(k_start + 1) * n + i_start, col] = 1
                    W[(k_start + 1) * n + (i_start + 1), col] = 1
                    col += 1

                    # M4 = A11(B10-B00)
                    U[(i_start + 1) * m + (j_start + 1), col] = 1
                    V[(j_start + 1) * p + k_start, col] = 1
                    V[j_start * p + k_start, col] = -1
                    W[k_start * n + i_start, col] = 1
                    W[k_start * n + (i_start + 1), col] = 1
                    col += 1

                    # M5 = (A00+A01)B11
                    U[i_start * m + j_start, col] = 1
                    U[i_start * m + (j_start + 1), col] = 1
                    V[(j_start + 1) * p + (k_start + 1), col] = 1
                    W[(k_start + 1) * n + i_start, col] = -1
                    W[(k_start + 1) * n + (i_start + 1), col] = 1
                    col += 1

                    # M6 = (A10-A00)(B00+B01)
                    U[(i_start + 1) * m + j_start, col] = 1
                    U[i_start * m + j_start, col] = -1
                    V[j_start * p + k_start, col] = 1
                    V[j_start * p + (k_start + 1), col] = 1
                    W[(k_start + 1) * n + (i_start + 1), col] = 1
                    col += 1

                    # M7 = (A01-A11)(B10+B11)
                    U[i_start * m + (j_start + 1), col] = 1
                    U[(i_start + 1) * m + (j_start + 1), col] = -1
                    V[(j_start + 1) * p + k_start, col] = 1
                    V[(j_start + 1) * p + (k_start + 1), col] = 1
                    W[k_start * n + i_start, col] = 1
                    col += 1

                else:
                    # Schoolbook for this block
                    for li in range(ni):
                        for lj in range(mj):
                            for lk in range(pk):
                                U[(i_start + li) * m + (j_start + lj), col] = 1
                                V[(j_start + lj) * p + (k_start + lk), col] = 1
                                W[(k_start + lk) * n + (i_start + li), col] = 1
                                col += 1

    assert col == total_rank, f"Column mismatch: {col} vs {total_rank}"

    # Verify
    if time.time() < deadline:
        if _verify(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback: schoolbook
    rank_fb = n * m * p
    U_fb = np.zeros((n * m, rank_fb), dtype=np.float64)
    V_fb = np.zeros((m * p, rank_fb), dtype=np.float64)
    W_fb = np.zeros((p * n, rank_fb), dtype=np.float64)
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j, c] = 1.0
                V_fb[j * p + k, c] = 1.0
                W_fb[k * n + i, c] = 1.0
                c += 1

    return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}


def _verify(U, V, W, n, m, p, atol=1e-9):
    i_idx = np.repeat(np.arange(n), m * p)
    j_idx = np.tile(np.repeat(np.arange(m), p), n)
    k_idx = np.tile(np.arange(p), n * m)
    u_rows = i_idx * m + j_idx
    v_rows = j_idx * p + k_idx
    w_rows = k_idx * n + i_idx

    products = U[u_rows, :] * V[v_rows, :] * W[w_rows, :]
    vals = products.sum(axis=1)

    return bool(np.allclose(vals, 1.0, atol=atol))


# EVOLVE-BLOCK-START
# The construct function is defined above.
# EVOLVE-BLOCK-END
