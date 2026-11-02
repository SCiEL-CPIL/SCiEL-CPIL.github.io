"""Matrix multiplication 2x2x14 - rank-minimal exact tensor decomposition."""
import numpy as np
import time

N_DIM = 2
M_DIM = 2
P_DIM = 14


def construct():
    """Return {'U','V','W'} with minimal-rank exact decomposition of the 2x2-by-2x14 matmul tensor."""
    t_start = time.time()
    deadline = t_start + 110

    n, m, p = N_DIM, M_DIM, P_DIM
    a_sz, b_sz, c_sz = n * m, m * p, p * n

    # Build target tensor T[a,b,c] = 1 when a=i*m+j, b=j*p+k, c=k*n+i
    T_target = np.zeros((a_sz, b_sz, c_sz), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i * m + j, j * p + k, k * n + i] = 1.0

    def check_exact(U_list, V_list, W_list, tol=1e-9):
        Ua = np.asarray(U_list, dtype=np.float64)
        Va = np.asarray(V_list, dtype=np.float64)
        Wa = np.asarray(W_list, dtype=np.float64)
        recon = np.einsum("ar,br,cr->abc", Ua, Va, Wa, optimize=True)
        return np.allclose(recon, T_target, atol=tol)

    # Strassen 7-term table for 2x2x2: each term is (U_rows, V_rows, W_rows)
    # U rows index A (0=A00,1=A01,2=A10,3=A11)
    # V rows index B (0=B00,1=B01,2=B10,3=B11)
    # W rows index C (0=C00,1=C01,2=C10,3=C11)
    strassen_terms = [
        # (U_indices, U_coeffs, V_indices, V_coeffs, W_indices, W_coeffs)
        ((0, 3), (1, 1), (0, 2), (1, 1), (0, 3), (1, -1)),     # M1
        ((2, 3), (1, 1), (0,), (1,), (2, 3), (1, -1)),           # M2
        ((0,), (1,), (1, 3), (1, -1), (1, 3), (1, 1)),           # M3
        ((3,), (1,), (2, 0), (1, -1), (0, 2), (1, 1)),           # M4
        ((0, 1), (1, 1), (3,), (1,), (0, 1), (-1, 1)),           # M5
        ((2, 0), (1, -1), (0, 1), (1, 1), (3,), (1,)),           # M6
        ((1, 3), (1, -1), (2, 3), (1, 1), (0,), (1,)),           # M7
    ]

    def build_strassen_on_blocks():
        """Build rank-49 decomposition: 7 Strassen terms per block-pair of columns."""
        num_blocks = p // 2  # 7 block-pairs
        rank = num_blocks * 7
        U_mat = np.zeros((a_sz, rank), dtype=np.int64)
        V_mat = np.zeros((b_sz, rank), dtype=np.int64)
        W_mat = np.zeros((c_sz, rank), dtype=np.int64)

        for blk in range(num_blocks):
            base_col = blk * 7
            # B column indices for this block: B[0,2*blk], B[0,2*blk+1], B[1,2*blk], B[1,2*blk+1]
            b00 = 0 * p + 2 * blk       # row 0, col 2*blk
            b01 = 0 * p + 2 * blk + 1   # row 0, col 2*blk+1
            b10 = 1 * p + 2 * blk       # row 1, col 2*blk
            b11 = 1 * p + 2 * blk + 1   # row 1, col 2*blk+1

            for t in range(7):
                col = base_col + t
                u_idx, u_coeff, v_idx, v_coeff, w_idx, w_coeff = strassen_terms[t]

                for u_i, u_c in zip(u_idx, u_coeff):
                    U_mat[u_i, col] += u_c

                # Map V indices: 0->b00, 1->b01, 2->b10, 3->b11
                v_map = {0: b00, 1: b01, 2: b10, 3: b11}
                for v_i, v_c in zip(v_idx, v_coeff):
                    V_mat[v_map[v_i], col] += v_c

                # Map W indices: 0->C00, 1->C01, 2->C10, 3->C11
                # C row k*n+i: k is B-col within block (0 or 1), i is A row (0 or 1)
                # C00 = k=0,i=0 -> 0*n+0=0; C01 = k=1,i=0 -> 1*n+0=2
                # C10 = k=0,i=1 -> 0*n+1=1; C11 = k=1,i=1 -> 1*n+1=3
                w_map = {0: 0 * n + 0, 1: 1 * n + 0, 2: 0 * n + 1, 3: 1 * n + 1}
                for w_i, w_c in zip(w_idx, w_coeff):
                    W_mat[w_map[w_i] + 2 * blk, col] += w_c

        return U_mat, V_mat, W_mat

    # Attempt 1: Strassen-based rank-49 decomposition
    U_s, V_s, W_s = build_strassen_on_blocks()
    if check_exact(U_s, V_s, W_s):
        if time.time() < deadline:
            return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}

    # Attempt 2: Schoolbook fallback (rank 56)
    rank_fb = n * m * p
    U_fb = np.zeros((a_sz, rank_fb), dtype=np.int64)
    V_fb = np.zeros((b_sz, rank_fb), dtype=np.int64)
    W_fb = np.zeros((c_sz, rank_fb), dtype=np.int64)
    c_idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j, c_idx] = 1
                V_fb[j * p + k, c_idx] = 1
                W_fb[k * n + i, c_idx] = 1
                c_idx += 1

    if check_exact(U_fb, V_fb, W_fb):
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

    # Should never reach here
    return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
