"""Rank-minimal exact tensor decomposition for 2x5x5 matrix multiplication."""
import time
import sys
import numpy as np
from itertools import product

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill

    n, m, p = 2, 5, 5
    N = n * m * p  # 50 tensor entries

    def build_schoolbook(rank=None):
        """Schoolbook decomposition."""
        if rank is None:
            rank = N
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W

    def compute_residual(U, V, W):
        """Compute residual vector: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] - 1"""
        res = np.empty(N)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    res[idx] = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :]) - 1.0
                    idx += 1
        return res

    def max_abs_residual(U, V, W):
        return np.max(np.abs(compute_residual(U, V, W)))

    def _build_A_U(V, W, R):
        """Build linear system matrix for solving U given V, W."""
        A = np.zeros((N, n*m*R))
        for i in range(n):
            for j in range(m):
                u_idx = i*m + j
                for k in range(p):
                    row = k*n*m + i*m + j
                    A[row, u_idx*R:(u_idx+1)*R] = V[j*p+k, :] * W[k*n+i, :]
        return A

    def _build_A_V(U, W, R):
        """Build linear system matrix for solving V given U, W."""
        A = np.zeros((N, m*p*R))
        for j in range(m):
            for k in range(p):
                v_idx = j*p + k
                for i in range(n):
                    row = k*n*m + i*m + j
                    A[row, v_idx*R:(v_idx+1)*R] = U[i*m+j, :] * W[k*n+i, :]
        return A

    def _build_A_W(U, V, R):
        """Build linear system matrix for solving W given U, V."""
        A = np.zeros((N, p*n*R))
        for k in range(p):
            for i in range(n):
                w_idx = k*n + i
                for j in range(m):
                    row = k*n*m + i*m + j
                    A[row, w_idx*R:(w_idx+1)*R] = U[i*m+j, :] * V[j*p+k, :]
        return A

    def als_solve(U, V, W, max_iter=50, tol=1e-10):
        """Alternating least squares to solve the tensor equation."""
        R = U.shape[1]
        b = np.ones(N)

        for it in range(max_iter):
            if time.time() > deadline - 5:
                break

            # Solve for U
            A_U = _build_A_U(V, W, R)
            try:
                U_flat, *_ = np.linalg.lstsq(A_U, b, rcond=None)
                U = U_flat.reshape(n*m, R)
            except:
                pass

            # Solve for V
            A_V = _build_A_V(U, W, R)
            try:
                V_flat, *_ = np.linalg.lstsq(A_V, b, rcond=None)
                V = V_flat.reshape(m*p, R)
            except:
                pass

            # Solve for W
            A_W = _build_A_W(U, V, R)
            try:
                W_flat, *_ = np.linalg.lstsq(A_W, b, rcond=None)
                W = W_flat.reshape(p*n, R)
            except:
                pass

            err = max_abs_residual(U, V, W)
            if err < tol:
                break

        return U, V, W

    def gauss_newton_polish(U, V, W, max_iter=20, tol=1e-12):
        """Gauss-Newton refinement for high precision."""
        R = U.shape[1]

        for it in range(max_iter):
            if time.time() > deadline - 5:
                break

            res = compute_residual(U, V, W)
            if np.max(np.abs(res)) < tol:
                break

            # Build Jacobian analytically
            Np = n*m*R + m*p*R + p*n*R
            J = np.zeros((N, Np))
            off_u = 0
            off_v = n*m*R
            off_w = n*m*R + m*p*R

            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # d/dU[i*m+j,r] = V[j*p+k,r] * W[k*n+i,r]
                        for r in range(R):
                            J[idx, off_u + (i*m+j)*R + r] = V[j*p+k, r] * W[k*n+i, r]
                        # d/dV[j*p+k,r] = U[i*m+j,r] * W[k*n+i,r]
                        for r in range(R):
                            J[idx, off_v + (j*p+k)*R + r] = U[i*m+j, r] * W[k*n+i, r]
                        # d/dW[k*n+i,r] = U[i*m+j,r] * V[j*p+k,r]
                        for r in range(R):
                            J[idx, off_w + (k*n+i)*R + r] = U[i*m+j, r] * V[j*p+k, r]
                        idx += 1

            # Solve (J^T J) delta = -J^T res
            try:
                JtJ = J.T @ J
                JtJ += 1e-10 * np.eye(Np)
                delta = np.linalg.solve(JtJ, -J.T @ res)
            except:
                break

            # Damped step
            cur_err = np.max(np.abs(res))
            step = 1.0
            improved = False
            for _ in range(8):
                U_t = U + step * delta[:n*m*R].reshape(n*m, R)
                V_t = V + step * delta[n*m*R:n*m*R+m*p*R].reshape(m*p, R)
                W_t = W + step * delta[n*m*R+m*p*R:].reshape(p*n, R)
                new_err = max_abs_residual(U_t, V_t, W_t)
                if new_err < cur_err:
                    U, V, W = U_t, V_t, W_t
                    improved = True
                    break
                step *= 0.5

            if not improved:
                break

        return U, V, W

    def snap_to_half_int(U, V, W, tol=1e-6):
        """Try to snap values to nearest half-integer if close."""
        def snap_matrix(M):
            M_snapped = np.round(M * 2) / 2
            if np.max(np.abs(M - M_snapped)) < tol:
                return M_snapped
            return M
        U = snap_matrix(U)
        V = snap_matrix(V)
        W = snap_matrix(W)
        return U, V, W

    def validate(U, V, W, tol=1e-8):
        """Check if decomposition is exact (within tolerance)."""
        res = compute_residual(U, V, W)
        return np.max(np.abs(res)) < tol

    def try_rank(R, n_attempts=12, seed=42):
        """Try to find valid decomposition of rank R."""
        if time.time() > deadline - 15:
            return None

        np.random.seed(seed)
        best_result = None
        best_err = float('inf')

        for attempt in range(n_attempts):
            if time.time() > deadline - 5:
                break

            # Random initialization with moderate scale
            scale = 0.5
            U = np.random.randn(n*m, R) * scale
            V = np.random.randn(m*p, R) * scale
            W = np.random.randn(p*n, R) * scale

            # ALS refinement
            U, V, W = als_solve(U, V, W, max_iter=40, tol=1e-11)

            # Gauss-Newton polish
            U, V, W = gauss_newton_polish(U, V, W, max_iter=15, tol=1e-13)

            err = max_abs_residual(U, V, W)
            if err < best_err:
                best_err = err
                best_result = (U.copy(), V.copy(), W.copy())

            if err < 1e-10:
                # Try to snap to half-integers
                U_s, V_s, W_s = snap_to_half_int(U, V, W, tol=1e-5)
                if validate(U_s, V_s, W_s, tol=1e-6):
                    return U_s, V_s, W_s
                return U, V, W

        if best_result is not None and best_err < 1e-8:
            U, V, W = best_result
            U_s, V_s, W_s = snap_to_half_int(U, V, W, tol=1e-4)
            if validate(U_s, V_s, W_s, tol=1e-5):
                return U_s, V_s, W_s
            return U, V, W
        return None

    # Try ranks from lower bound upward
    # Known: matmul(2,5,5) has rank at least ~38 based on literature
    # Schoolbook is 50
    for R in range(35, 51):
        if time.time() > deadline - 20:
            break

        result = try_rank(R, n_attempts=10, seed=R * 100 + 42)
        if result is not None:
            U, V, W = result
            # Verify
            if validate(U, V, W, tol=1e-7):
                # Convert to lists
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }

    # Fallback: schoolbook
    U, V, W = build_schoolbook()
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
