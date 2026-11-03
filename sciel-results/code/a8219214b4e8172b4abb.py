"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit

    n, m, p = DIMENSIONS
    dim_u = n * m  # 12
    dim_v = m * p  # 20
    dim_w = p * n  # 15

    def get_strassen_block(i0, j0, k0):
        """
        Returns 7 terms for a 2x2x2 Strassen block at offset (i0, j0, k0).
        Each term is (u_entries, v_entries, w_entries) where each is list of (idx, val).
        """
        terms = []

        # Term 1: M1 = (a00+a11)(b00+b11)
        u1 = [((i0+0)*m + (j0+0), 1), ((i0+1)*m + (j0+1), 1)]
        v1 = [((j0+0)*p + (k0+0), 1), ((j0+1)*p + (k0+1), 1)]
        w1 = [((k0+0)*n + (i0+0), 1), ((k0+1)*n + (i0+1), 1)]
        terms.append((u1, v1, w1))

        # Term 2: M2 = (a10+a11)b00
        u2 = [((i0+1)*m + (j0+0), 1), ((i0+1)*m + (j0+1), 1)]
        v2 = [((j0+0)*p + (k0+0), 1)]
        w2 = [((k0+0)*n + (i0+1), 1), ((k0+1)*n + (i0+1), -1)]
        terms.append((u2, v2, w2))

        # Term 3: M3 = a00(b01-b11)
        u3 = [((i0+0)*m + (j0+0), 1)]
        v3 = [((j0+0)*p + (k0+1), 1), ((j0+1)*p + (k0+1), -1)]
        w3 = [((k0+1)*n + (i0+0), 1), ((k0+1)*n + (i0+1), 1)]
        terms.append((u3, v3, w3))

        # Term 4: M4 = a11(b10-b00)
        u4 = [((i0+1)*m + (j0+1), 1)]
        v4 = [((j0+1)*p + (k0+0), 1), ((j0+0)*p + (k0+0), -1)]
        w4 = [((k0+0)*n + (i0+0), 1), ((k0+0)*n + (i0+1), 1)]
        terms.append((u4, w4))

        # Term 5: M5 = (a00+a01)b11
        u5 = [((i0+0)*m + (j0+0), 1), ((i0+0)*m + (j0+1), 1)]
        v5 = [((j0+1)*p + (k0+1), 1)]
        w5 = [((k0+0)*n + (i0+0), -1), ((k0+1)*n + (i0+0), 1)]
        terms.append((u5, v5, w5))

        # Term 6: M6 = (a10-a00)(b00+b01)
        u6 = [((i0+1)*m + (j0+0), 1), ((i0+0)*m + (j0+0), -1)]
        v6 = [((j0+0)*p + (k0+0), 1), ((j0+0)*p + (k0+1), 1)]
        w6 = [((k0+1)*n + (i0+1), 1)]
        terms.append((u6, v6, w6))

        # Term 7: M7 = (a01-a11)(b10+b11)
        u7 = [((i0+0)*m + (j0+1), 1), ((i0+1)*m + (j0+1), -1)]
        v7 = [((j0+1)*p + (k0+0), 1), ((j0+1)*p + (k0+1), 1)]
        w7 = [((k0+0)*n + (i0+0), 1)]
        terms.append((u7, v7, w7))

        return terms

    # Build the rank-56 hybrid decomposition: 4 Strassen blocks + schoolbook
    U_cols = []
    V_cols = []
    W_cols = []

    # 4 Strassen blocks: j0 in [0,2], k0 in [0,2]
    for j0 in [0, 2]:
        for k0 in [0, 2]:
            terms = get_strassen_block(0, j0, k0)
            for u, v, w in terms:
                U_col = np.zeros(dim_u)
                for idx, val in u:
                    U_col[idx] = val
                V_col = np.zeros(dim_v)
                for idx, val in v:
                    V_col[idx] = val
                W_col = np.zeros(dim_w)
                for idx, val in w:
                    W_col[idx] = val
                U_cols.append(U_col)
                V_cols.append(V_col)
                W_cols.append(W_col)

    # Schoolbook for the rest: i=2 (all j,k) and i in {0,1} with k=4
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i < 2 and k < 4:
                    continue
                U_col = np.zeros(dim_u)
                U_col[i*m + j] = 1
                V_col = np.zeros(dim_v)
                V_col[j*p + k] = 1
                W_col = np.zeros(dim_w)
                W_col[k*n + i] = 1
                U_cols.append(U_col)
                V_cols.append(V_col)
                W_cols.append(W_col)

    U = np.column_stack(U_cols)
    V = np.column_stack(V_cols)
    W = np.column_stack(W_cols)

    # Quick verification
    max_err = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i*m + j
                idx_v = j*p + k
                idx_w = k*n + i
                s = np.dot(U[idx_u, :], V[idx_v, :] * W[idx_w, :])
                max_err = max(max_err, abs(s - 1.0))

    if max_err > 1e-9:
        # Fallback to schoolbook
        U_sb = np.zeros((dim_u, n*m*p))
        V_sb = np.zeros((dim_v, n*m*p))
        W_sb = np.zeros((dim_w, n*m*p))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i*m + j, col] = 1
                    V_sb[j*p + k, col] = 1
                    W_sb[k*n + i, col] = 1
                    col += 1
        U, V, W = U_sb, V_sb, W_sb

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
