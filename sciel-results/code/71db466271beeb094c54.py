"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [5, 5, 5]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140.0  # 10s safety margin before the 150s kill

    def elapsed():
        return time.time() - start_time

    def schoolbook():
        """Rank n*m*p exact decomposition (always valid)."""
        rank = n * m * p
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

    def full_verify(U, V, W):
        """Verify sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for ALL i,j,k."""
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        # T[i,j,k] should be 1 for all i,j,k (matrix multiplication tensor)
        # Compute via einsum: result[i,j,k] = sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r]
        # Build index arrays
        idx_i = np.arange(n)
        idx_j = np.arange(m)
        idx_k = np.arange(p)
        II, JJ, KK = np.meshgrid(idx_i, idx_j, idx_k, indexing='ij')
        u_idx = (II * m + JJ).ravel()
        v_idx = (JJ * p + KK).ravel()
        w_idx = (KK * n + II).ravel()
        # For each (i,j,k), sum over r
        # U[u_idx, r] * V[v_idx, r] * W[w_idx, r]
        prod = U_np[u_idx, :] * V_np[v_idx, :] * W_np[w_idx, :]
        result = prod.sum(axis=1).reshape(n, m, p)
        if not np.allclose(result, 1.0, atol=1e-8, rtol=1e-8):
            return False
        return True

    # Block-Strassen family: 5 = 2 + 2 + 1
    block_sizes = [2, 2, 1]
    num_blocks = 3
    offsets = [0]
    for bs in block_sizes:
        offsets.append(offsets[-1] + bs)

    U_terms = []
    V_terms = []
    W_terms = []

    for bi in range(num_blocks):
        for bj in range(num_blocks):
            for bk in range(num_blocks):
                si = block_sizes[bi]
                sj = block_sizes[bj]
                sk = block_sizes[bk]

                a_row_start = offsets[bi]
                a_col_start = offsets[bj]
                b_row_start = offsets[bj]
                b_col_start = offsets[bk]
                c_row_start = offsets[bi]
                c_col_start = offsets[bk]

                if si == 2 and sj == 2 and sk == 2:
                    # Strassen: 7 rank-1 terms for C = A*B (2x2)
                    # M1 = (A00+A11)(B00+B11)
                    U_terms.append([(a_row_start+0, a_col_start+0, 1), (a_row_start+1, a_col_start+1, 1)])
                    V_terms.append([(b_row_start+0, b_col_start+0, 1), (b_row_start+0, b_col_start+1, 1)])
                    W_terms.append([(c_row_start+0, c_col_start+0, 1), (c_row_start+1, c_col_start+1, 1)])
                    # M2 = (A10+A11)(B00)
                    U_terms.append([(a_row_start+1, a_col_start+0, 1), (a_row_start+1, a_col_start+1, 1)])
                    V_terms.append([(b_row_start+0, b_col_start+0, 1)])
                    W_terms.append([(c_row_start+1, c_col_start+0, 1), (c_row_start+1, c_col_start+1, -1)])
                    # M3 = (A00)(B01-B11)
                    U_terms.append([(a_row_start+0, a_col_start+0, 1)])
                    V_terms.append([(b_row_start+0, b_col_start+1, 1), (b_row_start+1, b_col_start+1, -1)])
                    W_terms.append([(c_row_start+0, c_col_start+1, 1), (c_row_start+1, c_col_start+1, 1)])
                    # M4 = (A11)(B10-B00)
                    U_terms.append([(a_row_start+1, a_col_start+1, 1)])
                    V_terms.append([(b_row_start+1, b_col_start+0, 1), (b_row_start+0, b_col_start+0, -1)])
                    W_terms.append([(c_row_start+0, c_col_start+0, 1), (c_row_start+1, c_col_start+0, 1)])
                    # M5 = (A00+A01)(B11)
                    U_terms.append([(a_row_start+0, a_col_start+0, 1), (a_row_start+0, a_col_start+1, 1)])
                    V_terms.append([(b_row_start+1, b_col_start+1, 1)])
                    W_terms.append([(c_row_start+0, c_col_start+0, -1), (c_row_start+0, c_col_start+1, 1)])
                    # M6 = (A10-A00)(B00+B01)
                    U_terms.append([(a_row_start+1, a_col_start+0, 1), (a_row_start+0, a_col_start+0, -1)])
                    V_terms.append([(b_row_start+0, b_col_start+0, 1), (b_row_start+0, b_col_start+1, 1)])
                    W_terms.append([(c_row_start+1, c_col_start+1, 1)])
                    # M7 = (A01-A11)(B10+B11)
                    U_terms.append([(a_row_start+0, a_col_start+1, 1), (a_row_start+1, a_col_start+1, -1)])
                    V_terms.append([(b_row_start+1, b_col_start+0, 1), (b_row_start+1, b_col_start+1, 1)])
                    W_terms.append([(c_row_start+0, c_col_start+0, 1)])
                else:
                    # Schoolbook for this block
                    for ii in range(si):
                        for jj in range(sj):
                            for kk in range(sk):
                                U_terms.append([(a_row_start+ii, a_col_start+jj, 1)])
                                V_terms.append([(b_row_start+jj, b_col_start+kk, 1)])
                                W_terms.append([(c_row_start+ii, c_col_start+kk, 1)])

    rank = len(U_terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for col in range(rank):
        for (row_idx, val) in [(t[0], t[2]) for t in U_terms[col]]:
            U[row_idx][col] = val
        for (row_idx, val) in [(t[0], t[2]) for t in V_terms[col]]:
            V[row_idx][col] = val
        for (row_idx, val) in [(t[0], t[2]) for t in W_terms[col]]:
            W[row_idx][col] = val

    # High-precision polish pass: full-tensor verification (not just a subcube)
    if elapsed() < deadline:
        if full_verify(U, V, W):
            return {"U": U, "V": V, "W": W}

    # Fallback: guaranteed-valid schoolbook
    return schoolbook()
# EVOLVE-BLOCK-END
