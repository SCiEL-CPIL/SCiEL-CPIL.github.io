import time
import numpy as np

DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 90

    n, m, p = DIMENSIONS  # 2, 2, 16
    num_blocks = p // 2   # 8 blocks of width 2
    rank = 7 * num_blocks # 56

    U = [[0.0] * rank for _ in range(n * m)]   # 4 x 56
    V = [[0.0] * rank for _ in range(m * p)]   # 32 x 56
    W = [[0.0] * rank for _ in range(p * n)]   # 32 x 56

    # Strassen 2x2x2 decomposition: 7 rank-1 terms
    # Each term: (U_entries, V_entries, W_entries)
    # U_entries: list of (i, j, val) -> U[i*m + j][col]
    # V_entries: list of (j, k_local, val) -> V[j*p + k_base+k_local][col]
    # W_entries: list of (k_local, ip, val) -> W[(k_base+k_local)*n + ip][col]
    #
    # Standard Strassen for C = A*B where A is 2x2, B is 2x2:
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)(b00)
    # M3 = (a00)(b01-b11)
    # M4 = (a11)(b10-b00)
    # M5 = (a00+a01)(b11)
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    #
    # C00 = M1 - M4 + M5 + M7
    # C01 = M3 + M4
    # C10 = M2 + M5
    # C11 = M1 - M2 + M3 + M6

    terms = [
        # M1: (a00+a11)(b00+b11) -> contributes to C00, C11
        ([(0, 0, 1.0), (1, 1, 1.0)],
         [(0, 0, 1.0), (1, 1, 1.0)],
         [(0, 0, 1.0), (1, 1, 1.0)]),
        # M2: (a10+a11)(b00) -> contributes to C10, C11
        ([(1, 0, 1.0), (1, 1, 1.0)],
         [(0, 0, 1.0)],
         [(0, 1, 1.0), (1, 1, -1.0)]),
        # M3: (a00)(b01-b11) -> contributes to C01, C11
        ([(0, 0, 1.0)],
         [(0, 1, 1.0), (1, 1, -1.0)],
         [(1, 0, 1.0), (1, 1, 1.0)]),
        # M4: (a11)(b10-b00) -> contributes to C00, C01
        ([(1, 1, 1.0)],
         [(1, 0, 1.0), (0, 0, -1.0)],
         [(0, 0, 1.0), (0, 1, 1.0)]),
        # M5: (a00+a01)(b11) -> contributes to C00, C10
        ([(0, 0, 1.0), (0, 1, 1.0)],
         [(1, 1, 1.0)],
         [(0, 0, -1.0), (1, 0, 1.0)]),
        # M6: (a10-a00)(b00+b01) -> contributes to C11
        ([(1, 0, 1.0), (0, 0, -1.0)],
         [(0, 0, 1.0), (0, 1, 1.0)],
         [(1, 1, 1.0)]),
        # M7: (a01-a11)(b10+b11) -> contributes to C00
        ([(0, 1, 1.0), (1, 1, -1.0)],
         [(1, 0, 1.0), (1, 1, 1.0)],
         [(0, 0, 1.0)]),
    ]

    for block_idx in range(num_blocks):
        if time.time() > deadline:
            break
        k_base = 2 * block_idx
        for term_idx in range(7):
            col = block_idx * 7 + term_idx
            u_entries, v_entries, w_entries = terms[term_idx]

            for (i, j, val) in u_entries:
                U[i * m + j][col] = val

            for (j, k_local, val) in v_entries:
                k = k_base + k_local
                V[j * p + k][col] = val

            for (k_local, ip, val) in w_entries:
                k = k_base + k_local
                W[k * n + ip][col] = val

    # Verify the decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # should equal 1 for all valid (i,j,k)
    try:
        U_np = np.array(U, dtype=np.float64)
        V_np = np.array(V, dtype=np.float64)
        W_np = np.array(W, dtype=np.float64)

        # Reshape to 3D: U[i, j, r], V[j, k, r], W[k, i, r]
        U_3d = U_np.reshape(n, m, rank)
        V_3d = V_np.reshape(m, p, rank)
        W_3d = W_np.reshape(p, n, rank)

        # Compute tensor: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)

        # The matrix multiplication tensor has T[i,j,k] = 1 for all i,j,k
        if not np.allclose(T, 1.0, atol=1e-10):
            raise ValueError("Strassen decomposition verification failed")
    except Exception:
        # Fall back to schoolbook rank-64 decomposition
        rank_fb = n * m * p  # 64
        U = [[0.0] * rank_fb for _ in range(n * m)]
        V = [[0.0] * rank_fb for _ in range(m * p)]
        W = [[0.0] * rank_fb for _ in range(p * n)]

        for i in range(n):
            for j in range(m):
                for k in range(p):
                    col = i * m * p + j * p + k
                    U[i * m + j][col] = 1.0
                    V[j * p + k][col] = 1.0
                    W[k * n + i][col] = 1.0

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
