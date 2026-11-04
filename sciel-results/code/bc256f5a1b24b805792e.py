import numpy as np
import time

def construct():
    n, m, p = 3, 3, 4
    t0 = time.time()
    deadline = t0 + 110  # Safety margin

    # Target tensor: T[a, b, c] where a=i*m+j, b=j*p+k, c=k*n+i
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0

    # Precompute flattened tensors for ALS
    T_flat_W = T.reshape(n*m*m*p, p*n)
    T_flat_U = T.transpose(1, 2, 0).reshape(m*p*p*n, n*m)
    T_flat_V = T.transpose(0, 2, 1).reshape(n*m*p*n, m*p)

    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None

    def _finalize(U, V, W):
        """Polish: try to reduce rank by null-space elimination and snapping."""
        # Check current rank validity
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        if np.max(np.abs(T - T_approx)) > 1e-8:
            return U, V, W, U.shape[1]

        R = U.shape[1]

        # Try to reduce rank via SVD of the factor matrices
        # For each pair of matrices, we can try to find a low-rank approximation
        # that preserves the tensor product.

        # Strategy: for each R' < R, try random projections and check validity
        for R_test in range(max(1, R - 5), R):
            if time.time() > deadline:
                break
            for _attempt in range(3):
                if time.time() > deadline:
                    break
                # Random orthonormal projection
                P_U = np.random.randn(R, R_test)
                P_U, _ = np.linalg.qr(P_U)
                P_V = np.random.randn(R, R_test)
                P_V, _ = np.linalg.qr(P_V)

                U_new = U @ P_U
                V_new = V @ P_V
                W_new = np.zeros((p*n, R_test))

                # Need to find W_new such that sum_r u_ar v_br w_cr = T_abc
                # This is a least squares problem
                A_W = np.einsum('ar,br->abr', U_new, V_new).reshape(n*m*m*p, R_test)
                try:
                    W_solve, _, _, _ = np.linalg.lstsq(A_W, T_flat_W, rcond=None)
                    W_new = W_solve.T
                except:
                    continue

                T_check = np.einsum('ar,br,cr->abc', U_new, V_new, W_new)
                if np.max(np.abs(T - T_check)) < 1e-8:
                    # Valid! Try to snap to small rationals
                    U_s = _snap(U_new)
                    V_s = _snap(V_new)
                    W_s = _snap(W_new)
                    T_snap = np.einsum('ar,br,cr->abc', U_s, V_s, W_s)
                    if np.max(np.abs(T - T_snap)) < 1e-6:
                        return U_s, V_s, W_s, R_test
                    else:
                        return U_new, V_new, W_new, R_test

        return U, V, W, R

    def _snap(M, tol=1e-4):
        """Snap values to small fractions (multiples of 1/2, 1/3, 1/4, etc.)"""
        M_s = M.copy()
        for frac_denom in [1, 2, 3, 4, 6, 12]:
            for val in range(-24, 25):
                target = val / frac_denom
                mask = np.abs(M_s - target) < tol
                if np.any(mask):
                    M_s[mask] = target
        # Final: round very small values to 0
        M_s[np.abs(M_s) < 1e-8] = 0.0
        return M_s

    # Search ranks from high to low
    for R in range(n*m*p, 20, -1):
        if time.time() > deadline:
            break

        found = False
        n_attempts = 8 if R <= 30 else 5

        for attempt in range(n_attempts):
            if time.time() > deadline:
                break

            # Seeded random init
            seed = 1000 + R * 100 + attempt
            np.random.seed(seed)

            # Initialize with structured randomness: mix of random and sparse
            if attempt % 3 == 0:
                # Pure random
                U = np.random.randn(n*m, R)
                V = np.random.randn(m*p, R)
                W = np.random.randn(p*n, R)
            elif attempt % 3 == 1:
                # Sparse init (each row has ~1-2 nonzeros)
                U = np.zeros((n*m, R))
                V = np.zeros((m*p, R))
                W = np.zeros((p*n, R))
                for i in range(n*m):
                    idx = np.random.choice(R, size=min(2, R), replace=False)
                    U[i, idx] = np.random.choice([-1, 1], size=len(idx))
                for i in range(m*p):
                    idx = np.random.choice(R, size=min(2, R), replace=False)
                    V[i, idx] = np.random.choice([-1, 1], size=len(idx))
                for i in range(p*n):
                    idx = np.random.choice(R, size=min(2, R), replace=False)
                    W[i, idx] = np.random.choice([-1, 1], size=len(idx))
            else:
                # Hybrid: random with some zeros
                U = np.random.randn(n*m, R)
                V = np.random.randn(m*p, R)
                W = np.random.randn(p*n, R)
                U[np.abs(U) < 0.3] = 0
                V[np.abs(V) < 0.3] = 0
                W[np.abs(W) < 0.3] = 0

            best_err_this = float('inf')

            # ALS with multiple restarts inside
            for _restart in range(3):
                if time.time() > deadline:
                    break

                # ALS iterations
                for it in range(80):
                    if time.time() > deadline:
                        break

                    # Update W
                    A_W = np.einsum('ar,br->abr', U, V).reshape(n*m*m*p, R)
                    W_T, _, _, _ = np.linalg.lstsq(A_W, T_flat_W, rcond=None)
                    W = W_T.T

                    # Update U
                    A_U = np.einsum('br,cr->bcr', V, W).reshape(m*p*p*n, R)
                    U_T, _, _, _ = np.linalg.lstsq(A_U, T_flat_U, rcond=None)
                    U = U_T.T

                    # Update V
                    A_V = np.einsum('ar,cr->acr', U, W).reshape(n*m*p*n, R)
                    V_T, _, _, _ = np.linalg.lstsq(A_V, T_flat_V, rcond=None)
                    V = V_T.T

                    # Compute error
                    T_approx = np.einsum('ar,br,cr->abc', U, V, W)
                    err = np.sum((T - T_approx)**2)

                    if err < best_err_this:
                        best_err_this = err

                    if err < 1e-10:
                        break

                    # Early convergence check
                    if it > 10 and err < 1e-6:
                        # Try a few more iterations with tighter tolerance
                        pass

                if best_err_this < 1e-10:
                    break

                # Restart from perturbed solution
                if best_err_this < 1.0:
                    U += 0.1 * np.random.randn(n*m, R)
                    V += 0.1 * np.random.randn(m*p, R)
                    W += 0.1 * np.random.randn(p*n, R)

            if best_err_this < 1e-10:
                # Found a valid decomposition at rank R
                U, V, W, final_R = _finalize(U, V, W)
                if final_R < best_rank:
                    best_rank = final_R
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
                found = True
                break

        if found:
            # Found at this rank, continue searching lower
            continue
        else:
            # Didn't find at this rank; if we've been failing, we might as well stop
            # but give it a few more ranks
            if R <= 25 and not found:
                # Give up searching lower ranks, use schoolbook
                break

    # If we found something better than schoolbook, return it
    if best_U is not None and best_rank < n*m*p:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

    # Fallback: schoolbook rank 36
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
