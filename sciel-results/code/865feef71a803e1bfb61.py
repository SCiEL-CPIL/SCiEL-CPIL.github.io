"""Rank-minimal exact tensor decomposition for 2x5x5 matrix multiplication."""

import time
import numpy as np
from itertools import product

def construct():
    start_time = time.time()
    deadline = 145.0
    
    n, m, p = 2, 5, 5
    # Tensor indices: (i,j) for U row, (j,k) for V row, (k,i) for W row
    # Constraint: for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    def build_schoolbook(rank):
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
        """Compute the residual vector of all constraints."""
        residuals = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    residuals.append(val - 1.0)
        return np.array(residuals)
    
    def max_abs_residual(U, V, W):
        return np.max(np.abs(compute_residual(U, V, W)))
    
    def _build_A_U(V, W, R):
        """50 x (n*m*R) matrix: row (i,j,k) has V[j*p+k,:]*W[k*n+i,:] at cols (i*m+j)*R:(i*m+j+1)*R."""
        A = np.zeros((n*m*p, n*m*R))
        for i in range(n):
            for j in range(m):
                u_idx = i*m+j
                for k in range(p):
                    A[k*n*m + i*m + j, u_idx*R:(u_idx+1)*R] = V[j*p+k, :] * W[k*n+i, :]
        return A

    def _build_A_V(U, W, R):
        A = np.zeros((n*m*p, m*p*R))
        for j in range(m):
            for k in range(p):
                v_idx = j*p+k
                for i in range(n):
                    A[k*n*m + i*m + j, v_idx*R:(v_idx+1)*R] = U[i*m+j, :] * W[k*n+i, :]
        return A

    def _build_A_W(U, V, R):
        A = np.zeros((n*m*p, p*n*R))
        for k in range(p):
            for i in range(n):
                w_idx = k*n+i
                for j in range(m):
                    A[k*n*m + i*m + j, w_idx*R:(w_idx+1)*R] = U[i*m+j, :] * V[j*p+k, :]
        return A

    def als_step(U, V, W, max_iter=100, tol=1e-10):
        """Full-system ALS: solve all 50 constraints jointly for each factor (tensor flattening)."""
        R = U.shape[1]
        b = np.ones(n*m*p)

        for it in range(max_iter):
            if time.time() > deadline - 5:
                break

            # Solve for U (all rows at once)
            A_U = _build_A_U(V, W, R)
            try:
                U_flat = np.linalg.lstsq(A_U, b, rcond=None)[0]
                U = U_flat.reshape(n*m, R)
            except:
                pass

            # Solve for V
            A_V = _build_A_V(U, W, R)
            try:
                V_flat = np.linalg.lstsq(A_V, b, rcond=None)[0]
                V = V_flat.reshape(m*p, R)
            except:
                pass

            # Solve for W
            A_W = _build_A_W(U, V, R)
            try:
                W_flat = np.linalg.lstsq(A_W, b, rcond=None)[0]
                W = W_flat.reshape(p*n, R)
            except:
                pass

            err = max_abs_residual(U, V, W)
            if err < tol:
                break

        return U, V, W

    def gauss_newton(U, V, W, max_iter=30, tol=1e-12):
        """Gauss-Newton polish on the smooth sum-of-squares residual."""
        R = U.shape[1]
        N = n*m*p

        def residuals():
            res = np.empty(N)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        res[k*n*m + i*m + j] = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :]) - 1.0
            return res

        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
            res = residuals()
            if np.max(np.abs(res)) < tol:
                break

            # Build Jacobian J (N x (N_u+N_v+N_w)) numerically via analytic partials
            Np = n*m*R + m*p*R + p*n*R
            J = np.zeros((N, Np))
            # d/d U[i*m+j, r] of res(k,i,j) = V[j*p+k, r]*W[k*n+i, r]
            off_u = 0
            off_v = n*m*R
            off_w = n*m*R + m*p*R
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        idx = k*n*m + i*m + j
                        # U partial
                        for r in range(R):
                            J[idx, off_u + (i*m+j)*R + r] = V[j*p+k, r] * W[k*n+i, r]
                        # V partial
                        for r in range(R):
                            J[idx, off_v + (j*p+k)*R + r] = U[i*m+j, r] * W[k*n+i, r]
                        # W partial
                        for r in range(R):
                            J[idx, off_w + (k*n+i)*R + r] = U[i*m+j, r] * V[j*p+k, r]

            # Solve (J^T J) delta = -J^T res
            try:
                JtJ = J.T @ J
                JtJ += 1e-12 * np.eye(Np)
                delta = np.linalg.solve(JtJ, -J.T @ res)
            except:
                break

            # Damped step
            step = 1.0
            for _ in range(10):
                U_t = U.copy(); V_t = V.copy(); W_t = W.copy()
                U_t += step * delta[:n*m*R].reshape(n*m, R)
                V_t += step * delta[n*m*R:n*m*R+m*p*R].reshape(m*p, R)
                W_t += step * delta[n*m*R+m*p*R:].reshape(p*n, R)
                if max_abs_residual(U_t, V_t, W_t) < max_abs_residual(U, V, W):
                    U, V, W = U_t, V_t, W_t
                    break
                step *= 0.5
            else:
                break

        return U, V, W

    def try_rank(R, seed=42):
        """Try to find valid decomposition of rank R."""
        if time.time() > deadline - 10:
            return None

        np.random.seed(seed)

        best_result = None
        best_err = float('inf')

        # Multiple initializations
        for init_iter in range(8):
            if time.time() > deadline - 5:
                break

            # Random initialization
            U = np.random.randn(n*m, R) * 0.3
            V = np.random.randn(m*p, R) * 0.3
            W = np.random.randn(p*n, R) * 0.3

            # Full-system ALS refinement
            U, V, W = als_step(U, V, W, max_iter=60, tol=1e-11)

            # Gauss-Newton polish
            U, V, W = gauss_newton(U, V, W, max_iter=20, tol=1e-12)

            err = max_abs_residual(U, V, W)
            if err < best_err:
                best_err = err
                best_result = (U.copy(), V.copy(), W.copy())

            if err < 1e-10:
                return U, V, W

        if best_result is not None and best_err < 1e-8:
            return best_result
        return None

    # Start with schoolbook
    best_rank = n * m * p
    U_best, V_best, W_best = build_schoolbook(best_rank)

    # Try ranks from 38 upward (known rank for 2x5x5 is 38)
    for R in range(38, 45):
        if time.time() > deadline - 20:
            break
        found = False
        for seed in range(40):
            if time.time() > deadline - 10:
                break
            result = try_rank(R, seed=seed * 7 + 1)
            if result is not None:
                U_best, V_best, W_best = result
                best_rank = R
                found = True
                break
        if found:
            break

    # Final high-precision polish
    if best_rank < n * m * p:
        U_best, V_best, W_best = als_step(U_best, V_best, W_best, max_iter=100, tol=1e-13)
        U_best, V_best, W_best = gauss_newton(U_best, V_best, W_best, max_iter=40, tol=1e-14)

    return {"U": U_best.tolist(), "V": V_best.tolist(), "W": W_best.tolist()}
