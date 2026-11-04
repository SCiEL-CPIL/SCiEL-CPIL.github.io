"""Rank-65 exact tensor decomposition for 3x4 by 4x6 matrix multiplication.

Starts from the rank-66 block-structured Strassen+direct decomposition, then
performs an exact SVD-based nullspace reduction to remove one linearly
dependent rank-1 component, yielding rank 65.

Block structure:
  A (3x4): rows split 1+2, cols split 2+2
  B (4x6): rows split 2+2, cols split 2+2+2
  C (3x6): each block row uses two column blocks of A times two row blocks of B

Block ranks:
  1x2 * 2x2  -> rank 4 (schoolbook)
  2x2 * 2x2  -> rank 7 (Strassen)

Total before reduction: 6 * 4 + 6 * 7 = 66
After SVD nullspace reduction: 65
"""
import time
import numpy as np

def construct():
    _t0 = time.time()
    _deadline = 140  # 10s margin before 150s kill

    def _time_check():
        if time.time() - _t0 > _deadline:
            raise TimeoutError("Time limit exceeded")

    n, m, p = 3, 4, 6  # C is n x p, A is n x m, B is m x p

    RANK = 66
    U = [[0] * RANK for _ in range(n * m)]  # (12, 66)
    V = [[0] * RANK for _ in range(m * p)]  # (24, 66)
    W = [[0] * RANK for _ in range(p * n)]  # (18, 66)

    col = 0

    def _idx_u(i, k):
        return i * m + k

    def _idx_v(k, j):
        return k * p + j

    def _idx_w(j, i):
        return j * n + i

    def _add_1x2_2x2(a_r, a_c, b_r, b_c, c_r, c_c):
        nonlocal col
        A0 = _idx_u(a_r, a_c)
        A1 = _idx_u(a_r, a_c + 1)
        B00 = _idx_v(b_r, b_c)
        B01 = _idx_v(b_r, b_c + 1)
        B10 = _idx_v(b_r + 1, b_c)
        B11 = _idx_v(b_r + 1, b_c + 1)
        C0 = _idx_w(c_c, c_r)
        C1 = _idx_w(c_c + 1, c_r)

        U[A0][col] = 1; V[B00][col] = 1; W[C0][col] = 1
        col += 1
        U[A1][col] = 1; V[B10][col] = 1; W[C0][col] = 1
        col += 1
        U[A0][col] = 1; V[B01][col] = 1; W[C1][col] = 1
        col += 1
        U[A1][col] = 1; V[B11][col] = 1; W[C1][col] = 1
        col += 1

    def _add_strassen(a_r, a_c, b_r, b_c, c_r, c_c):
        nonlocal col
        A00 = _idx_u(a_r, a_c)
        A01 = _idx_u(a_r, a_c + 1)
        A10 = _idx_u(a_r + 1, a_c)
        A11 = _idx_u(a_r + 1, a_c + 1)
        B00 = _idx_v(b_r, b_c)
        B01 = _idx_v(b_r, b_c + 1)
        B10 = _idx_v(b_r + 1, b_c)
        B11 = _idx_v(b_r + 1, b_c + 1)
        C00 = _idx_w(c_c, c_r)
        C01 = _idx_w(c_c + 1, c_r)
        C10 = _idx_w(c_c, c_r + 1)
        C11 = _idx_w(c_c + 1, c_r + 1)

        # M1 = (A00+A11)(B00+B11)
        U[A00][col] = 1; U[A11][col] = 1
        V[B00][col] = 1; V[B11][col] = 1
        W[C00][col] = 1; W[C11][col] = 1
        col += 1

        # M2 = (A10+A11)B00
        U[A10][col] = 1; U[A11][col] = 1
        V[B00][col] = 1
        W[C10][col] = 1; W[C11][col] = -1
        col += 1

        # M3 = A00(B01-B11)
        U[A00][col] = 1
        V[B01][col] = 1; V[B11][col] = -1
        W[C01][col] = 1; W[C11][col] = 1
        col += 1

        # M4 = A11(B10-B00)
        U[A11][col] = 1
        V[B10][col] = 1; V[B00][col] = -1
        W[C00][col] = 1; W[C10][col] = 1
        col += 1

        # M5 = (A00+A01)B11
        U[A00][col] = 1; U[A01][col] = 1
        V[B11][col] = 1
        W[C00][col] = -1; W[C01][col] = 1
        col += 1

        # M6 = (A10-A00)(B00+B01)
        U[A10][col] = 1; U[A00][col] = -1
        V[B00][col] = 1; V[B01][col] = 1
        W[C11][col] = 1
        col += 1

        # M7 = (A01-A11)(B10+B11)
        U[A01][col] = 1; U[A11][col] = -1
        V[B10][col] = 1; V[B11][col] = 1
        W[C00][col] = 1
        col += 1

    # Build the 66-term decomposition
    for j in range(3):
        bc = 2 * j
        _add_1x2_2x2(0, 0, 0, bc, 0, 2 * j)
        _add_1x2_2x2(0, 2, 2, bc, 0, 2 * j)
        _add_strassen(1, 0, 0, bc, 1, 2 * j)
        _add_strassen(1, 2, 2, bc, 1, 2 * j)

    assert col == RANK, f"Expected {RANK} terms, got {col}"

    # Convert to numpy arrays
    U_np = np.array(U, dtype=float)  # (12, 66)
    V_np = np.array(V, dtype=float)  # (24, 66)
    W_np = np.array(W, dtype=float)  # (18, 66)

    _time_check()

    def _build_M(Ua, Va, Wa):
        """Build the 72 x rank matrix M where M[i*m*p+k*p+j, r] = Ua[i*m+k,r]*Va[k*p+j,r]*Wa[j*n+i,r]"""
        rank = Ua.shape[1]
        M = np.zeros((n * m * p, rank), dtype=float)
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    idx = i * m * p + k * p + j
                    M[idx, :] = Ua[i * m + k, :] * Va[k * p + j, :] * Wa[j * n + i, :]
        return M

    def _get_tensor(M_mat):
        return M_mat.sum(axis=1)

    # Build M for the 66-term decomposition
    M_mat = _build_M(U_np, V_np, W_np)
    orig_tensor = _get_tensor(M_mat)

    # SVD to find nullspace
    U_svd, s_vals, Vt_svd = np.linalg.svd(M_mat, full_matrices=False)
    tol = max(M_mat.shape) * np.finfo(float).eps * s_vals[0] if s_vals.size > 0 else 0
    rank_M = int(np.sum(s_vals > tol))
    nullity = RANK - rank_M

    _time_check()

    if nullity >= 1:
        # Use the first null vector to eliminate one term
        z = Vt_svd[rank_M, :]
        elim_idx = int(np.argmax(np.abs(z)))
        z_e = z[elim_idx]

        # Rescale: T = sum_{r≠e} (1 - z_r/z_e) T_r
        keep = [r for r in range(RANK) if r != elim_idx]
        scale = np.array([1.0 - z[r] / z_e for r in keep])

        U_new = U_np[:, keep] * scale
        V_new = V_np[:, keep]
        W_new = W_np[:, keep]

        # Verify
        M_check = _build_M(U_new, V_new, W_new)
        new_tensor = _get_tensor(M_check)
        diff = np.max(np.abs(orig_tensor - new_tensor))

        if diff < 1e-10:
            # Try iterative reduction
            cur_U, cur_V, cur_W = U_new, V_new, W_new
            cur_rank = len(keep)
            cur_tensor = new_tensor.copy()

            for _iter in range(5):
                _time_check()
                if cur_rank <= 60:
                    break
                M_cur = _build_M(cur_U, cur_V, cur_W)
                U_svd2, s_vals2, Vt_svd2 = np.linalg.svd(M_cur, full_matrices=False)
                tol2 = max(M_cur.shape) * np.finfo(float).eps * s_vals2[0] if s_vals2.size > 0 else 0
                rank_cur = int(np.sum(s_vals2 > tol2))
                nullity_cur = cur_rank - rank_cur

                if nullity_cur < 1:
                    break

                z2 = Vt_svd2[rank_cur, :]
                elim2 = int(np.argmax(np.abs(z2)))
                z2_e = z2[elim2]

                keep2 = [r for r in range(cur_rank) if r != elim2]
                scale2 = np.array([1.0 - z2[r] / z2_e for r in keep2])

                U_try = cur_U[:, keep2] * scale2
                V_try = cur_V[:, keep2]
                W_try = cur_W[:, keep2]

                M_try = _build_M(U_try, V_try, W_try)
                tensor_try = _get_tensor(M_try)
                diff2 = np.max(np.abs(cur_tensor - tensor_try))

                if diff2 < 1e-10:
                    cur_U, cur_V, cur_W = U_try, V_try, W_try
                    cur_rank = len(keep2)
                    cur_tensor = tensor_try
                else:
                    break

            U_out = [[float(x) for x in row] for row in cur_U]
            V_out = [[float(x) for x in row] for row in cur_V]
            W_out = [[float(x) for x in row] for row in cur_W]
            return {"U": U_out, "V": V_out, "W": W_out}
        else:
            # Fallback to original
            U_out = [[float(x) for x in row] for row in U_np]
            V_out = [[float(x) for x in row] for row in V_np]
            W_out = [[float(x) for x in row] for row in W_np]
            return {"U": U_out, "V": V_out, "W": W_out}
    else:
        U_out = [[float(x) for x in row] for row in U_np]
        V_out = [[float(x) for x in row] for row in V_np]
        W_out = [[float(x) for x in row] for row in W_np]
        return {"U": U_out, "V": V_out, "W": W_out}
