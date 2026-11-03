"""Matrix multiplication 4x4x8 with Strassen-based rank-98 decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START

def _build_2x2x2_strassen():
    """Build the rank-7 Strassen decomposition for 2x2x2.
    Index conventions: U index = i*2+j, V index = j*2+k, W index = k*2+i.
    """
    terms = [
        ([1, 0, 0, -1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        ([0, 0, 1, -1], [0, 1, 0, 1], [0, 0, 0, 1]),
        ([1, 0, -1, 0], [1, 0, 1, 0], [1, 0, 0, 0]),
    ]
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    for r, (u, v, w) in enumerate(terms):
        U[:, r] = u
        V[:, r] = v
        W[:, r] = w
    return U, V, W


def _build_4x4x4_strassen2():
    """
    Build a rank-49 decomposition for 4x4x4 using Strassen on 2x2 blocks.
    U index = i*4+j, V index = j*4+k, W index = k*4+i.
    """
    U = np.zeros((16, 49))
    V = np.zeros((16, 49))
    W = np.zeros((16, 49))

    U2, V2, W2 = _build_2x2x2_strassen()

    block_U = [
        [1, 0, 0, -1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, -1],
        [1, 0, -1, 0],
    ]
    block_V = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
    ]
    c_combos = [
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0],
    ]

    for op_idx in range(7):
        lc = block_U[op_idx]
        rc = block_V[op_idx]
        cc = c_combos[op_idx]
        for t in range(7):
            col = op_idx * 7 + t
            for a_blk in range(4):
                if lc[a_blk] == 0:
                    continue
                a_row = a_blk // 2
                a_col = a_blk % 2
                for li in range(2):
                    for lj in range(2):
                        u_idx = (a_row * 2 + li) * 4 + (a_col * 2 + lj)
                        U[u_idx, col] += lc[a_blk] * U2[2 * li + lj, t]
            for b_blk in range(4):
                if rc[b_blk] == 0:
                    continue
                b_row = b_blk // 2
                b_col = b_blk % 2
                for lj in range(2):
                    for lk in range(2):
                        v_idx = (b_row * 2 + lj) * 4 + (b_col * 2 + lk)
                        V[v_idx, col] += rc[b_blk] * V2[2 * lj + lk, t]
            for c_blk in range(4):
                c_coeff = cc[c_blk]
                if c_coeff == 0:
                    continue
                c_row = c_blk // 2
                c_col = c_blk % 2
                for lk in range(2):
                    for li in range(2):
                        w_idx = (c_row * 2 + lk) * 4 + (c_col * 2 + li)
                        W[w_idx, col] += c_coeff * W2[2 * lk + li, t]

    return U, V, W


def _build_4x4x8_rank98():
    """
    Rank-98 decomposition: Strassen for both halves.
    C1 = A*B1 (Strassen 49), C2 = A*B2 (Strassen 49).
    """
    U44, V44, W44 = _build_4x4x4_strassen2()

    rank = 98
    U = np.zeros((16, rank))
    V = np.zeros((32, rank))
    W = np.zeros((32, rank))

    # C1 = A*B1: columns 0..48
    U[:, 0:49] = U44
    for j in range(4):
        for k in range(4):
            for c in range(49):
                V[j * 8 + k, c] = V44[j * 4 + k, c]
    W[0:16, 0:49] = W44

    # C2 = A*B2: columns 49..97
    U[:, 49:98] = U44
    for j in range(4):
        for k in range(4):
            for c in range(49):
                V[j * 8 + (k + 4), 49 + c] = V44[j * 4 + k, c]
    for k in range(4):
        for i in range(4):
            for c in range(49):
                W[(k + 4) * 4 + i, 49 + c] = W44[k * 4 + i, c]

    return U, V, W


def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = np.dot(U[i * m + j], V[j * p + k] * W[k * n + i])
                if abs(total - 1.0) > tol:
                    return False
    return True


def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying
    a 4x4 matrix by a 4x8 matrix.
    Returns dict with keys 'U', 'V', 'W' containing factor matrices.
    """
    start_time = time.time()
    deadline = start_time + 120.0
    n, m, p = DIMENSIONS  # 4, 4, 8

    try:
        U, V, W = _build_4x4x8_rank98()
        if _verify_decomposition(U, V, W, n, m, p):
            if time.time() < deadline - 5.0:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception:
        pass

    # Fallback: schoolbook decomposition (rank 128)
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

# EVOLVE-BLOCK-END
