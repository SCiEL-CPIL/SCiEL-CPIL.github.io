import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 4, 4]

def construct():
    start_time = time.time()
    deadline = 140  # safety margin before 150s kill

    n, m, p = DIMENSIONS

    # Build the target tensor T[i,j,k] for 2x4x4
    # i = row_a * 4 + col_a  (A is 2x4)
    # j = row_b * 4 + col_b  (B is 4x4)
    # k = col_c * 2 + row_c  (C is 2x4, W[k*n+i,r] convention: k = col*n + row)
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for row_a in range(n):
        for col_a in range(m):
            i = row_a * m + col_a
            for row_b in range(m):
                for col_b in range(p):
                    j = row_b * p + col_b
                    for row_c in range(n):
                        for col_c in range(p):
                            k = col_c * n + row_c
                            if col_a == row_b and row_a == row_c and col_b == col_c:
                                T[i, j, k] = 1.0

    # Precompute the known rank-28 Strassen decomposition as fallback
    U_s = np.array([
        [1, 0, 1, 0, 1, 0, 1],
        [0, 0, 1, -1, 0, 1, 0],
        [0, 1, 0, 0, 1, 0, -1],
        [1, 1, 0, 1, 0, -1, 0]
    ], dtype=float)

    V_s = np.array([
        [1, 1, 0, 0, 1, 0, 1],
        [0, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, 1, 0, 1, 0],
        [1, 0, 1, 0, 0, 1, 1]
    ], dtype=float)

    T_2 = np.zeros((4, 4, 4), dtype=float)
    for ra in range(2):
        for ca in range(2):
            ii = ra * 2 + ca
            for rb in range(2):
                for cb in range(2):
                    jj = rb * 2 + cb
                    for rc in range(2):
                        for cc in range(2):
                            kk = cc * 2 + rc
                            if ca == rb and ra == rc and cb == cc:
                                T_2[ii, jj, kk] = 1.0

    M_sys = np.zeros((64, 28), dtype=float)
    T_vec = np.zeros(64, dtype=float)
    for ii in range(4):
        for jj in range(4):
            for kk in range(4):
                eq = ii * 16 + jj * 4 + kk
                T_vec[eq] = T_2[ii, jj, kk]
                for s in range(7):
                    M_sys[eq, kk * 7 + s] = U_s[ii, s] * V_s[jj, s]

    W_vec, _, _, _ = np.linalg.lstsq(M_sys, T_vec, rcond=None)
    W_s = W_vec.reshape(4, 7)

    def build_rank28():
        U_full = np.zeros((8, 28), dtype=float)
        V_full = np.zeros((16, 28), dtype=float)
        W_full = np.zeros((8, 28), dtype=float)
        offset = 0
        for bi in range(2):
            for bj in range(2):
                for ra in range(2):
                    for ca in range(2):
                        local_i = ra * 2 + ca
                        global_i = ra * 4 + (2 * bi + ca)
                        U_full[global_i, offset:offset + 7] = U_s[local_i, :]
                for rb in range(2):
                    for cb in range(2):
                        local_j = rb * 2 + cb
                        global_j = (2 * bi + rb) * 4 + (2 * bj + cb)
                        V_full[global_j, offset:offset + 7] = V_s[local_j, :]
                for rc in range(2):
                    for cc in range(2):
                        local_k = cc * 2 + rc
                        global_k = (2 * bj + cc) * 2 + rc
                        W_full[global_k, offset:offset + 7] = W_s[local_k, :]
                offset += 7
        return U_full, V_full, W_full

    U28, V28, W28 = build_rank28()

    def verify_decomposition(U_full, V_full, W_full, T_target, tol=1e-9):
        max_err = 0.0
        for i in range(U_full.shape[0]):
            for j in range(V_full.shape[0]):
                for k in range(W_full.shape[0]):
                    val = np.dot(U_full[i, :] * V_full[j, :], W_full[k, :])
                    err = abs(val - T_target[i, j, k])
                    if err > max_err:
                        max_err = err
        return max_err

    # Verify rank-28
    err28 = verify_decomposition(U28, V28, W28, T)
    if err28 > 1e-9:
        # Ultimate fallback: schoolbook rank 32
        U_fb = np.zeros((8, 32))
        V_fb = np.zeros((16, 32))
        W_fb = np.zeros((8, 32))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j, col] = 1
                    V_fb[j * p + k, col] = 1
                    W_fb[k * n + i, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

    # --- NEW STRATEGY: Tensor-Train (MPS) compression + ALS CP refinement ---
    # Represent T as an MPS, compress via SVD, then use ALS to find a lower-rank CP.

    # Flatten T to a 3D array of shape (8, 16, 8)
    # MPS: T[i,j,k] = sum_{a,b} G1[i,a] * G2[a,j,b] * G3[b,k]
    # Initial MPS from SVD:
    # Reshape T to (8, 16*8) -> SVD -> (8, r1) (r1, 16)
    # Then reshape (r1, 16, 8) -> (r1*16, 8) -> SVD -> (r1*16, r2) (r2, 8)
    # But we want bond dimensions to be small.

    # Actually, let's do a proper MPS decomposition:
    # Step 1: Reshape T to matrix M1 of shape (8, 16*8) = (8, 128)
    # SVD: M1 = U1 @ S1 @ V1^T, keep top r1 singular values
    # G1 = U1[:, :r1] * sqrt(S1[:r1])  shape (8, r1)
    # R1 = sqrt(S1[:r1]) @ V1[:r1, :]  shape (r1, 128)
    # Reshape R1 to (r1, 16, 8)
    # Step 2: Reshape to M2 of shape (r1*16, 8)
    # SVD: M2 = U2 @ S2 @ V2^T, keep top r2
    # G2 = U2.reshape(r1, 16, r2) * S2[:r2]  (broadcast)
    # G3 = V2[:r2, :].T  shape (r2, 8)

    # For exact representation, we need full rank. But the MPS structure
    # reveals the tensor-train rank, which guides CP search.

    # Let's compute the MPS to understand the structure, then use ALS for CP.

    # MPS decomposition (exact, full bond dims)
    T_3d = T  # shape (8, 16, 8)
    M1 = T_3d.reshape(8, 128)
    U1, S1, V1t = np.linalg.svd(M1, full_matrices=False)
    r1 = len(S1)  # should be <= 8
    # For exact: use all singular values
    G1 = U1 * np.sqrt(S1)  # (8, r1)
    R1 = np.sqrt(S1) @ V1t  # (r1, 128)
    R1_3d = R1.reshape(r1, 16, 8)
    M2 = R1_3d.reshape(r1 * 16, 8)
    U2, S2, V2t = np.linalg.svd(M2, full_matrices=False)
    r2 = len(S2)  # should be <= 8
    G2 = (U2 * np.sqrt(S2)).reshape(r1, 16, r2)
    G3 = (np.sqrt(S2) @ V2t).T  # (r2, 8)

    # Verify MPS
    def mps_reconstruct(G1, G2, G3):
        r1 = G1.shape[1]
        r2 = G3.shape[0]
        result = np.zeros((G1.shape[0], G2.shape[1], G3.shape[1]))
        for i in range(G1.shape[0]):
            for j in range(G2.shape[1]):
                for k in range(G3.shape[1]):
                    val = 0.0
                    for a in range(r1):
                        for b in range(r2):
                            val += G1[i, a] * G2[a, j, b] * G3[b, k]
                    result[i, j, k] = val
        return result

    # The MPS structure tells us the tensor-train ranks are (r1, r2).
    # For a 2x4x4 matmul tensor, these are likely small (<= 8).
    # Now use this as initialization for ALS CP decomposition.

    # ALS CP decomposition: T ≈ sum_{r=1}^{R} u_r ⊗ v_r ⊗ w_r
    # where u_r ∈ R^8, v_r ∈ R^16, w_r ∈ R^8

    def als_cp_decompose(T_target, R, max_iter=200, tol=1e-12):
        """Alternating Least Squares for CP decomposition."""
        n_i, n_j, n_k = T_target.shape
        # Initialize with random
        np.random.seed(42)
        U = np.random.randn(n_i, R)
        V = np.random.randn(n_j, R)
        W = np.random.randn(n_k, R)

        # Normalize initial factors
        U /= np.linalg.norm(U, axis=0, keepdims=True)
        V /= np.linalg.norm(V, axis=0, keepdims=True)
        W /= np.linalg.norm(W, axis=0, keepdims=True)

        prev_err = np.inf
        for it in range(max_iter):
            # Update U: for each i, U[i,:] = (T[i,:,:] * (V ⊙ W))^T @ (V ⊙ W)
            VW = V * W  # (n_j*n_k? No, this is element-wise, not outer)
            # Actually: T[i,:,:] is (n_j, n_k). We need (V ⊙ W) which is element-wise product
            # The equation: sum_r U[i,r] * V[j,r] * W[k,r] = T[i,j,k]
            # For fixed i: U[i,:] @ (V[j,:] * W[k,:])^T = T[i,j,k]
            # Vectorize: (V ⊙ W) as matrix of shape (n_j*n_k, R) where entry (j*n_k+k, r) = V[j,r]*W[k,r]
            VW_mat = (V[:, :, None] * W[None, :, :]).reshape(n_j * n_k, R)  # (n_j*n_k, R)
            # Actually simpler: VW_mat[j*n_k+k, r] = V[j,r] * W[k,r]
            VW_mat = np.einsum('jr,kr->jkr', V, W).reshape(n_j * n_k, R)
            T_i = T_target.reshape(n_i, n_j * n_k)  # (n_i, n_j*n_k)
            # U[i,:] = (T_i[i,:] @ VW_mat^T) @ (VW_mat @ VW_mat^T)^{-1}
            A = VW_mat.T @ VW_mat  # (R, R)
            A += 1e-12 * np.eye(R)
            for i in range(n_i):
                b = T_i[i, :] @ VW_mat  # (R,)
                U[i, :] = np.linalg.solve(A, b)

            # Update V
            UW_mat = np.einsum('ir,kr->ikr', U, W).reshape(n_i * n_k, R)
            T_j = T_target.transpose(1, 0, 2).reshape(n_j, n_i * n_k)
            A = UW_mat.T @ UW_mat
            A += 1e-12 * np.eye(R)
            for j in range(n_j):
                b = T_j[j, :] @ UW_mat
                V[j, :] = np.linalg.solve(A, b)

            # Update W
            UV_mat = np.einsum('ir,jr->ijr', U, V).reshape(n_i * n_j, R)
            T_k = T_target.transpose(2, 0, 1).reshape(n_k, n_i * n_j)
            A = UV_mat.T @ UV_mat
            A += 1e-12 * np.eye(R)
            for k in range(n_k):
                b = T_k[k, :] @ UV_mat
                W[k, :] = np.linalg.solve(A, b)

            # Compute error
            err = 0.0
            for i in range(n_i):
                for j in range(n_j):
                    for k in range(n_k):
                        val = np.dot(U[i, :] * V[j, :], W[k, :])
                        err = max(err, abs(val - T_target[i, j, k]))
            if err < tol:
                break
            if abs(prev_err - err) < tol * 0.1:
                break
            prev_err = err

        return U, V, W, err

    # Try ranks from 27 down to 20
    best_result = None
    best_rank = 28  # start with known rank-28

    # First, try to find a rank-27 solution via ALS
    for R in [27, 26, 25]:
        if time.time() > deadline - 30:
            break

        # Multiple random restarts
        found_exact = False
        for restart in range(10):
            if time.time() > deadline - 30:
                break

            np.random.seed(1000 + restart)
            # Initialize using MPS structure for better starting point
            # Use the MPS factors to create a structured initialization
            U_init = np.random.randn(8, R)
            V_init = np.random.randn(16, R)
            W_init = np.random.randn(8, R)

            # Scale initialization based on MPS norms
            U_init *= np.sqrt(np.sum(G1**2) / (R * 8))
            V_init *= np.sqrt(np.sum(G2**2) / (R * 16))
            W_init *= np.sqrt(np.sum(G3**2) / (R * 8))

            U_als, V_als, W_als, err_als = als_cp_decompose(T, R, max_iter=300, tol=1e-13)

            if err_als < 1e-9:
                found_exact = True
                best_rank = R
                best_result = (U_als, V_als, W_als)
                break

        if found_exact:
            break

    # If we found a lower-rank exact solution, verify and return
    if best_result is not None and best_rank < 28:
        U_best, V_best, W_best = best_result
        final_err = verify_decomposition(U_best, V_best, W_best, T, tol=1e-9)
        if final_err < 1e-9:
            return {"U": U_best.tolist(), "V": V_best.tolist(), "W": W_best.tolist()}

    # Fallback: return the verified rank-28 Strassen decomposition
    return {"U": U28.tolist(), "V": V28.tolist(), "W": W28.tolist()}
