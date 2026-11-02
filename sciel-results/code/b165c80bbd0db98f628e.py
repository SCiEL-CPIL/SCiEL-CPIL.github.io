import time
import numpy as np
import random
from scipy.optimize import minimize

DIMENSIONS = [4, 4, 7]
WALL_DEADLINE = 145.0
SAFETY_MARGIN = 5.0


def construct():
    start_time = time.time()
    n, m, p = 4, 4, 7

    def elapsed():
        return time.time() - start_time

    def check_valid(U, V, W, tol=1e-8):
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        r = U_arr.shape[1]
        # Build the 3D tensor of products and sum over rank
        # U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # We need to check that for each (i,j,k), sum_c = 1
        # Reshape and use einsum
        # U is (n*m, r), V is (m*p, r), W is (p*n, r)
        # For each c, the product U[:,c] * V[:,c] * W[:,c] gives contributions
        # Let's compute residual directly
        res = np.zeros(n * m * p)
        for c in range(r):
            u_c = U_arr[:, c]  # (n*m,)
            v_c = V_arr[:, c]  # (m*p,)
            w_c = W_arr[:, c]  # (p*n,)
            # For each (i,j,k): u_c[i*m+j] * v_c[j*p+k] * w_c[k*n+i]
            # Reshape u_c to (n,m), v_c to (m,p), w_c to (p,n)
            u3 = u_c.reshape(n, m)
            v3 = v_c.reshape(m, p)
            w3 = w_c.reshape(p, n)
            # Product: for each (i,j,k): u3[i,j] * v3[j,k] * w3[k,i]
            # This is a 3D tensor contraction
            prod = np.einsum('ij,jk,ki->ijk', u3, v3, w3)
            res += prod.flatten()
        if np.max(np.abs(res - 1.0)) > tol:
            return False
        return True

    def build_schoolbook():
        r = n * m * p
        U = [[0] * r for _ in range(n * m)]
        V = [[0] * r for _ in range(m * p)]
        W = [[0] * r for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return U, V, W

    def build_strassen_2x2_2x2():
        U = [[0] * 7 for _ in range(4)]
        V = [[0] * 7 for _ in range(4)]
        W = [[0] * 7 for _ in range(4)]
        # p1 = (a11+a22)(b11+b22)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1
        # p2 = (a21+a22)b11
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[1][1] = 1; W[3][1] = -1
        # p3 = a11(b12-b22)
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        W[2][2] = 1; W[3][2] = 1
        # p4 = a22(b21-b11)
        U[3][3] = 1
        V[2][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[1][3] = 1
        # p5 = (a11+a12)b22
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = -1; W[2][4] = 1
        # p6 = (a21-a11)(b11+b12)
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1
        # p7 = (a12-a22)(b21+b22)
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[0][6] = 1
        return U, V, W

    def kronecker_decompose_4x4x4():
        U2, V2, W2 = build_strassen_2x2_2x2()
        U4 = [[0] * 49 for _ in range(16)]
        V4 = [[0] * 49 for _ in range(16)]
        W4 = [[0] * 49 for _ in range(16)]
        for r1 in range(7):
            for r2 in range(7):
                r = r1 * 7 + r2
                for i1 in range(2):
                    for j1 in range(2):
                        for i2 in range(2):
                            for j2 in range(2):
                                i = i1 * 2 + i2
                                j = j1 * 2 + j2
                                val = U2[i1 * 2 + j1][r1] * U2[i2 * 2 + j2][r2]
                                if val != 0:
                                    U4[i * 4 + j][r] += val
                for j1 in range(2):
                    for k1 in range(2):
                        for j2 in range(2):
                            for k2 in range(2):
                                j = j1 * 2 + j2
                                k = k1 * 2 + k2
                                val = V2[j1 * 2 + k1][r1] * V2[j2 * 2 + k2][r2]
                                if val != 0:
                                    V4[j * 4 + k][r] += val
                for k1 in range(2):
                    for i1 in range(2):
                        for k2 in range(2):
                            for i2 in range(2):
                                k = k1 * 2 + k2
                                i = i1 * 2 + i2
                                val = W2[k1 * 2 + i1][r1] * W2[k2 * 2 + i2][r2]
                                if val != 0:
                                    W4[k * 4 + i][r] += val
        return U4, V4, W4

    def build_4x4x7_decomposition():
        U4, V4, W4 = kronecker_decompose_4x4x4()
        r4 = len(U4[0])
        r_total = r4 + 48
        U = [[0] * r_total for _ in range(16)]
        V = [[0] * r_total for _ in range(28)]
        W = [[0] * r_total for _ in range(28)]
        for i in range(16):
            for r in range(r4):
                U[i][r] = U4[i][r]
        for j in range(4):
            for k in range(4):
                for r in range(r4):
                    V[j * 7 + k][r] = V4[j * 4 + k][r]
        for k in range(4):
            for i in range(4):
                for r in range(r4):
                    W[k * 4 + i][r] = W4[k * 4 + i][r]
        col_offset = r4
        for i in range(4):
            for j in range(4):
                for k in range(4, 7):
                    U[i * 4 + j][col_offset] = 1
                    V[j * 7 + k][col_offset] = 1
                    W[k * 4 + i][col_offset] = 1
                    col_offset += 1
        return U, V, W

    # ---- Refinement machinery: random linear-combination rank reduction ----
    def tensors_to_vec(U, V, W):
        """Flatten the three factor matrices into one vector (U, V, W)."""
        r = len(U[0])
        u = np.array(U, dtype=float).flatten()
        v = np.array(V, dtype=float).flatten()
        w = np.array(W, dtype=float).flatten()
        return np.concatenate([u, v, w])

    def vec_to_tensors(x, r):
        u_len = n * m * r
        v_len = m * p * r
        U = x[:u_len].reshape(n * m, r)
        V = x[u_len:u_len + v_len].reshape(m * p, r)
        W = x[u_len + v_len:].reshape(p * n, r)
        return U, V, W

    def build_jacobian(U, V, W):
        """
        Build the Jacobian of the residual vector w.r.t. the parameter vector x=(U,V,W).
        Residual for (i,j,k): sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] - 1 = 0
        d/dU[i*m+j,c] = V[j*p+k,c]*W[k*n+i,c]
        d/dV[j*p+k,c] = U[i*m+j,c]*W[k*n+i,c]
        d/dW[k*n+i,c] = U[i*m+j,c]*V[j*p+k,c]
        """
        U = np.asarray(U, dtype=float)
        V = np.asarray(V, dtype=float)
        W = np.asarray(W, dtype=float)
        r = U.shape[1]
        u_len = n * m * r
        v_len = m * p * r
        w_len = p * n * r
        J = np.zeros((n * m * p, u_len + v_len + w_len))
        for c in range(r):
            u_c = U[:, c].reshape(n, m)
            v_c = V[:, c].reshape(m, p)
            w_c = W[:, c].reshape(p, n)
            # d/dU[i*m+j,c] = V[j*p+k,c]*W[k*n+i,c] for each (i,j,k)
            # This is: for fixed (i,j), the value at row (i,j,k) is v_c[j,k]*w_c[k,i]
            # So for U: J[(i,j,k), i*m*r+j*r+c] = v_c[j,k]*w_c[k,i]
            # Reshape: u3 indices (i,j), the contribution is outer product structure
            # For each c, dU part: for each (i,j), the vector over k is v_c[j,:]*w_c[:,i]
            dU_block = np.einsum('jk,ki->ijk', v_c, w_c)  # (n,m,p)
            # dV part: for each (j,k), the vector over (i) is u_c[i,j]*w_c[k,i]
            dV_block = np.einsum('ij,ki->ijk', u_c, w_c)  # (n,m,p)
            # dW part: for each (k,i), the vector over (j) is u_c[i,j]*v_c[j,k]
            dW_block = np.einsum('ij,jk->ijk', u_c, v_c)  # (n,m,p)
            # Now fill in J
            # Row index: (i*m+j)*p + k
            # U col: i*m*r + j*r + c
            # V col: u_len + j*p*r + k*r + c
            # W col: u_len + v_len + k*n*r + i*r + c
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        row = (i * m + j) * p + k
                        J[row, i * m * r + j * r + c] = dU_block[i, j, k]
                        J[row, u_len + j * p * r + k * r + c] = dV_block[i, j, k]
                        J[row, u_len + v_len + k * n * r + i * r + c] = dW_block[i, j, k]
        return J

    def residual_vec(U, V, W):
        U = np.asarray(U, dtype=float)
        V = np.asarray(V, dtype=float)
        W = np.asarray(W, dtype=float)
        r = U.shape[1]
        res = np.zeros(n * m * p)
        for c in range(r):
            u_c = U[:, c]
            v_c = V[:, c]
            w_c = W[:, c]
            u3 = u_c.reshape(n, m)
            v3 = v_c.reshape(m, p)
            w3 = w_c.reshape(p, n)
            prod = np.einsum('ij,jk,ki->ijk', u3, v3, w3)
            res += prod.flatten()
        return res - 1.0

    def random_merge(U, V, W, n_merge, rng, tol=1e-9):
        """
        Merge n_merge columns via a random linear combination, then
        solve a least-squares correction to restore validity.
        Returns (new_U, new_V, new_W, new_rank) or None on failure.
        """
        r = len(U[0])
        U = np.array(U, dtype=float).copy()
        V = np.array(V, dtype=float).copy()
        W = np.array(W, dtype=float).copy()

        new_r = r - n_merge
        cols = list(range(r))
        rng.shuffle(cols)
        kept = cols[:new_r]
        dropped = cols[new_r:]
        
        # M: new_r x r, each new col t = kept[t] + sum_d alpha_{t,d} * dropped[d]
        M = np.zeros((new_r, r))
        for t, kc in enumerate(kept):
            M[t, kc] = 1.0
            for dc in dropped:
                M[t, dc] = rng.uniform(-0.5, 0.5)

        U_new = U @ M.T
        V_new = V @ M.T
        W_new = W @ M.T

        # Newton correction
        res = residual_vec(U_new, V_new, W_new)
        if np.linalg.norm(res) < 1e-10:
            if check_valid(U_new.tolist(), V_new.tolist(), W_new.tolist(), tol=1e-8):
                return U_new.tolist(), V_new.tolist(), W_new.tolist(), new_r
            return None
        
        J = build_jacobian(U_new, V_new, W_new)
        try:
            delta, *_ = np.linalg.lstsq(J, -res, rcond=None)
        except Exception:
            return None
        x_new = tensors_to_vec(U_new, V_new, W_new) + delta
        U2, V2, W2 = vec_to_tensors(x_new, new_r)

        # Verify with tighter check
        if not check_valid(U2, V2, W2, tol=1e-7):
            return None
        return U2, V2, W2, new_r

    def polish_decomposition(U, V, W, tol=1e-10, max_iter=100):
        """
        Polish a decomposition using Newton's method to drive residuals to machine precision.
        Returns polished (U, V, W) arrays or None on failure.
        """
        r = len(U[0])
        U = np.array(U, dtype=float).copy()
        V = np.array(V, dtype=float).copy()
        W = np.array(W, dtype=float).copy()

        for iteration in range(max_iter):
            res = residual_vec(U, V, W)
            res_norm = np.linalg.norm(res)
            if res_norm < tol:
                break

            J = build_jacobian(U, V, W)
            try:
                delta, *_ = np.linalg.lstsq(J, -res, rcond=None)
            except Exception:
                return None

            # Damping: if the full step increases residual, reduce it
            alpha = 1.0
            for _ in range(20):
                x_trial = tensors_to_vec(U, V, W) + alpha * delta
                U_t, V_t, W_t = vec_to_tensors(x_trial, r)
                res_trial = residual_vec(U_t, V_t, W_t)
                if np.linalg.norm(res_trial) < res_norm:
                    break
                alpha *= 0.5

            x_new = tensors_to_vec(U, V, W) + alpha * delta
            U, V, W = vec_to_tensors(x_new, r)

        # Final check
        if not check_valid(U.tolist(), V.tolist(), W.tolist(), tol=tol):
            return None
        return U, V, W

    # ---- Main flow ----
    U_school, V_school, W_school = build_schoolbook()
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = n * m * p

    rng = random.Random(12345)

    if elapsed() < WALL_DEADLINE - SAFETY_MARGIN:
        try:
            U_kron, V_kron, W_kron = build_4x4x7_decomposition()
            r_kron = len(U_kron[0])
            if check_valid(U_kron, V_kron, W_kron):
                if r_kron < best_rank:
                    best_result = {"U": U_kron, "V": V_kron, "W": W_kron}
                    best_rank = r_kron
        except Exception:
            pass

    # Refinement: iteratively merge columns on the best valid decomposition
    if elapsed() < WALL_DEADLINE - SAFETY_MARGIN:
        cur_U = np.array(best_result["U"], dtype=float)
        cur_V = np.array(best_result["V"], dtype=float)
        cur_W = np.array(best_result["W"], dtype=float)
        cur_rank = len(cur_U[0])

        max_attempts = 800
        attempts = 0
        target = 30
        consecutive_failures = 0
        max_consecutive_failures = 30
        
        while cur_rank > target and attempts < max_attempts and elapsed() < WALL_DEADLINE - SAFETY_MARGIN:
            if consecutive_failures >= max_consecutive_failures:
                break
            merged = False
            # Start with 1 merge, increase if successful
            for n_merge in (1, 2, 3):
                if cur_rank - n_merge < 1:
                    continue
                res = random_merge(cur_U, cur_V, cur_W, n_merge, rng, tol=1e-9)
                attempts += 1
                if res is not None:
                    cur_U, cur_V, cur_W, cur_rank = res
                    merged = True
                    consecutive_failures = 0
                    if cur_rank < best_rank:
                        best_rank = cur_rank
                        best_result = {
                            "U": cur_U.tolist(),
                            "V": cur_V.tolist(),
                            "W": cur_W.tolist(),
                        }
                    break
                consecutive_failures += 1
                if elapsed() > WALL_DEADLINE - SAFETY_MARGIN:
                    break
            if not merged and consecutive_failures >= max_consecutive_failures:
                break

        # Final polish pass: drive residuals to machine precision
        if elapsed() < WALL_DEADLINE - SAFETY_MARGIN:
            try:
                polished = polish_decomposition(cur_U, cur_V, cur_W, tol=1e-12, max_iter=30)
                if polished is not None:
                    cur_U, cur_V, cur_W = polished
                    best_result = {
                        "U": cur_U.tolist(),
                        "V": cur_V.tolist(),
                        "W": cur_W.tolist(),
                    }
            except Exception:
                pass

    return best_result
