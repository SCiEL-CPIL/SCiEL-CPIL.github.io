"""Tensor decomposition for matrix multiplication 3x3 * 3x8.
Strassen blocks for the 2x2 core + direct terms, then aggressive
rank reduction via column-space analysis of the induced tensor map.
"""

import time
import numpy as np

DIMENSIONS = [3, 3, 8]


def construct():
    n, m, p = DIMENSIONS  # 3, 3, 8
    start_time = time.time()
    DEADLINE = 135.0  # safety margin before 150s kill

    def check_deadline():
        return time.time() - start_time > DEADLINE

    # ---- Build the base decomposition: 4 Strassen blocks + direct terms ----
    U_t = []
    V_t = []
    W_t = []

    def strassen_decomp(a0, a1, b0, b1):
        """Add 7 Strassen terms for A[a0:a0+2, a1:a1+2] * B[b0:b0+2, b1:b1+2]."""
        # m1 = (a+d)(e+h)
        U_t.append([((a0, b0), 1), ((a0+1, b1), 1)])
        V_t.append([((a0, b1), 1), ((a0+1, b1+1), 1)])
        W_t.append([((a0, b0), 1), ((a0+1, b1+1), 1)])
        # m2 = (c+d)*e
        U_t.append([((a0+1, b0), 1), ((a0+1, b1), 1)])
        V_t.append([((a0, b1), 1)])
        W_t.append([((a0+1, b0), 1), ((a0+1, b1+1), -1)])
        # m3 = a*(f-h)
        U_t.append([((a0, b0), 1)])
        V_t.append([((a0+1, b1), 1), ((a0+1, b1+1), -1)])
        W_t.append([((a0, b1), 1), ((a0+1, b1+1), 1)])
        # m4 = d*(g-e)
        U_t.append([((a0+1, b1), 1)])
        V_t.append([((a0+1, b1), 1), ((a0, b1), -1)])
        W_t.append([((a0, b0), 1), ((a0+1, b0), 1)])
        # m5 = (a+b)*h
        U_t.append([((a0, b0), 1), ((a0, b1), 1)])
        V_t.append([((a0+1, b1+1), 1)])
        W_t.append([((a0, b0), -1), ((a0, b1), 1)])
        # m6 = (c-a)*(e+f)
        U_t.append([((a0+1, b0), 1), ((a0, b0), -1)])
        V_t.append([((a0, b1), 1), ((a0+1, b1), 1)])
        W_t.append([((a0+1, b1+1), 1)])
        # m7 = (b-d)*(g+h)
        U_t.append([((a0, b1), 1), ((a0+1, b1), -1)])
        V_t.append([((a0+1, b1), 1), ((a0+1, b1+1), 1)])
        W_t.append([((a0, b0), 1)])

    def direct_term(i, j, k):
        U_t.append([((i, j), 1)])
        V_t.append([((j, k), 1)])
        W_t.append([((i, k), 1)])

    # Strassen for A[0:2, 0:2] * B[0:2, 0:8]
    for k_off in range(0, 8, 2):
        strassen_decomp(0, 0, k_off, 0)

    # Direct: A[0:2, 2] * B[2, 0:8]
    for i in range(2):
        for k in range(8):
            direct_term(i, 2, k)

    # Direct: A[2, 0:2] * B[0:2, 0:8]
    for j in range(2):
        for k in range(8):
            direct_term(2, j, k)

    # Direct: A[2,2] * B[2, 0:8]
    for k in range(8):
        direct_term(2, 2, k)

    rank = len(W_t)

    # Build dense matrices
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    for idx in range(rank):
        for (i, j), coeff in U_t[idx]:
            U[i * m + j, idx] = coeff
        for (j, k), coeff in V_t[idx]:
            V[j * p + k, idx] = coeff
        for (i, k), coeff in W_t[idx]:
            W[k * n + i, idx] = coeff

    def verify(Ua, Va, Wa):
        r = Ua.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(Ua[i*m+j], Va[j*p+k])
                    s = s * Wa[k*n+i]
                    if abs(np.sum(s) - 1.0) > 1e-9:
                        return False
        return True

    if not verify(U, V, W):
        # Fallback: schoolbook
        rank_fb = n * m * p
        U_fb = np.zeros((n * m, rank_fb), dtype=float)
        V_fb = np.zeros((m * p, rank_fb), dtype=float)
        W_fb = np.zeros((p * n, rank_fb), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j, col] = 1
                    V_fb[j * p + k, col] = 1
                    W_fb[k * n + i, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

    def to_nice_list(arr):
        result = []
        for row in arr:
            nice_row = []
            for val in row:
                val2 = val * 2
                if abs(val2 - round(val2)) < 1e-6:
                    nice_val = round(val2) / 2.0
                    if abs(nice_val - round(nice_val)) < 1e-9:
                        nice_val = int(round(nice_val))
                    nice_row.append(nice_val)
                else:
                    nice_row.append(float(val))
            result.append(nice_row)
        return result

    # ---- Rank reduction via column-space analysis ----
    # Build the tensor matrix: each column r maps to the vector
    #   (U[:,r] ⊗ V[:,r] ⊗ W[:,r]) in R^(n*m*p)
    # If column r is in the span of other columns, we can remove it.
    N = n * m * p
    T = np.zeros((N, rank), dtype=float)
    for r in range(rank):
        # T[:, r] = vec(U[:,r] * V[:,r] * W[:,r]) reshaped
        # Index: (i, j, k) -> i*m*p + j*p + k
        # U index: i*m+j, V index: j*p+k, W index: k*n+i
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i*m*p + j*p + k, r] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]

    # Use QR with column pivoting to find a minimal spanning set
    # We want the smallest set of columns that spans the column space.
    # Approach: iterative greedy - start with all columns, remove one at a time
    # if it's in the span of the rest.

    active = list(range(rank))
    T_active = T[:, active].copy()
    U_active = U[:, active].copy()
    V_active = V[:, active].copy()
    W_active = W[:, active].copy()

    # First pass: SVD to get the numerical rank
    try:
        _, s_vals, _ = np.linalg.svd(T_active, full_matrices=False)
        tol = 1e-10 * (s_vals[0] if len(s_vals) > 0 else 1.0)
        num_rank = np.sum(s_vals > tol)
    except:
        num_rank = len(active)

    # Iterative removal: try to remove columns from the end
    # Use a more efficient approach: build a basis incrementally
    # and track which original columns are needed.

    if num_rank < len(active):
        # Use QR with column pivoting to find a basis
        # We'll do this by maintaining a basis and checking residuals
        basis_cols = []
        basis_matrix = None

        # Sort columns by norm (descending) for better pivoting
        col_norms = np.linalg.norm(T_active, axis=0)
        order = np.argsort(col_norms)[::-1]

        for idx in order:
            if check_deadline():
                break
            col = T_active[:, idx]
            if basis_matrix is None:
                basis_cols.append(idx)
                basis_matrix = col.copy().reshape(-1, 1)
            else:
                # Check if col is in span of basis
                try:
                    coeffs, residuals, rk, sv = np.linalg.lstsq(
                        basis_matrix, col, rcond=1e-10)
                    recon = basis_matrix @ coeffs
                    if not np.allclose(recon, col, atol=1e-8):
                        # Not in span, add to basis
                        basis_cols.append(idx)
                        basis_matrix = np.hstack([basis_matrix, col.reshape(-1, 1)])
                except:
                    basis_cols.append(idx)
                    basis_matrix = np.hstack([basis_matrix, col.reshape(-1, 1)])

            if len(basis_cols) >= num_rank:
                break

        # basis_cols contains indices (into T_active) of columns needed
        # Map back to original indices
        keep_orig = [active[i] for i in basis_cols]
        # Sort for consistency
        keep_orig.sort()

        if len(keep_orig) < len(active):
            U = U[:, keep_orig]
            V = V[:, keep_orig]
            W = W[:, keep_orig]
            rank = len(keep_orig)

    # Second pass: try to remove more columns now that we have a smaller set
    # Rebuild T for the reduced set
    T = np.zeros((N, rank), dtype=float)
    for r in range(rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i*m*p + j*p + k, r] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]

    # Iterative removal from the end
    for r in range(rank - 1, -1, -1):
        if check_deadline():
            break
        if r == 0:
            break
        others = list(range(r)) + list(range(r + 1, rank))
        T_others = T[:, others]
        T_col = T[:, r]
        try:
            coeffs, residuals, rk, sv = np.linalg.lstsq(T_others, T_col, rcond=1e-10)
            recon = T_others @ coeffs
            if np.allclose(recon, T_col, atol=1e-8):
                U = np.delete(U, r, axis=1)
                V = np.delete(V, r, axis=1)
                W = np.delete(W, r, axis=1)
                T = np.delete(T, r, axis=1)
                rank -= 1
        except:
            pass

    # Verify after reduction
    if not verify(U, V, W):
        # Rebuild from scratch without reduction
        U_t2 = []
        V_t2 = []
        W_t2 = []
        for k_off in range(0, 8, 2):
            strassen_decomp(0, 0, k_off, 0)
        for i in range(2):
            for k in range(8):
                direct_term(i, 2, k)
        for j in range(2):
            for k in range(8):
                direct_term(2, j, k)
        for k in range(8):
            direct_term(2, 2, k)
        rank2 = len(W_t2)
        U2 = np.zeros((n * m, rank2), dtype=float)
        V2 = np.zeros((m * p, rank2), dtype=float)
        W2 = np.zeros((p * n, rank2), dtype=float)
        for idx in range(rank2):
            for (i, j), coeff in U_t2[idx]:
                U2[i * m + j, idx] = coeff
            for (j, k), coeff in V_t2[idx]:
                V2[j * p + k, idx] = coeff
            for (i, k), coeff in W_t2[idx]:
                W2[k * n + i, idx] = coeff
        return {"U": to_nice_list(U2), "V": to_nice_list(V2), "W": to_nice_list(W2)}

    return {"U": to_nice_list(U), "V": to_nice_list(V), "W": to_nice_list(W)}
