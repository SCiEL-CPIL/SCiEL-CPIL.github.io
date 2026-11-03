"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin before 150s kill

    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 75

    # Build the matrix multiplication tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    # We try to find a decomposition of rank r < 75
    # T[a,b,c] = sum_s U[a,s] * V[b,s] * W[c,s]
    #
    # Strategy: Try ranks from 64 down to 40 (or as time allows).
    # For each rank, use a randomized tensor decomposition approach.
    # We'll use a simple iterative method: randomly initialize U, V, W and
    # try to fit the tensor via alternating least squares or gradient descent.
    #
    # Actually, for exact decomposition with small integers/half-integers,
    # we can try a structured approach. Let's first try to see if we can
    # find a decomposition using known lower bounds.
    #
    # For 3x5x5, the schoolbook rank is 75. The theoretical lower bound
    # is at least max(n*m, m*p, p*n) = 25. But practical decompositions
    # are harder.
    #
    # Let's try a randomized search for lower-rank decompositions.
    # We'll use a simple approach: randomly initialize and do ALS.

    best_rank = rank_schoolbook
    best_result = None

    # First, let's try to find a decomposition with rank < 75
    # We'll try ranks in decreasing order, but limit total time.

    def try_rank(r, time_budget):
        """Try to find a rank-r decomposition. Returns (U, V, W) or None."""
        if time.time() > time_budget:
            return None

        # Random initialization
        np.random.seed(42)  # Fixed seed for reproducibility
        U = np.random.randn(n * m, r) * 0.1
        V = np.random.randn(m * p, r) * 0.1
        W = np.random.randn(p * n, r) * 0.1

        # ALS-style iteration
        for it in range(200):
            if time.time() > time_budget:
                return None

            # Update U: for each a, U[a,:] = argmin ||T[a,:,:] - V^T * (W * U[a,:])||
            # This is a linear system: (W * V^T) U[a,:]^T = vec(T[a,:,:])
            # where (W * V^T)[c,b] = sum_s W[c,s]*V[b,s]
            # Actually, let's do it properly.
            # T[a,b,c] = sum_s U[a,s]*V[b,s]*W[c,s]
            # For fixed a: T[a,b,c] = sum_s U[a,s] * (V[b,s]*W[c,s])
            # Let M_a[b,c] = T[a,b,c], and for each s, let v_s = V[:,s], w_s = W[:,s]
            # Then M_a = sum_s U[a,s] * outer(v_s, w_s)
            # So we need to solve: sum_s U[a,s] * outer(V[:,s], W[:,s]) = T[a,:,:]
            # This is a linear system in U[a,:].

            # Precompute the Gram-like matrix
            # For each pair (b,c), the coefficient for U[a,s] is V[b,s]*W[c,s]
            # So the system is: sum_s U[a,s] * V[b,s]*W[c,s] = T[a,b,c]
            # Reshape: let Z[s, b, c] = V[b,s]*W[c,s]
            # Then sum_s U[a,s] * Z[s,b,c] = T[a,b,c]

            # Build the linear system matrix A of shape (m*p*n, r)
            # A[b*m*p + c... wait, let's index properly
            # A[(b,c), s] = V[b,s]*W[c,s]
            A = np.zeros((m * p * n, r))
            for s in range(r):
                for b in range(m * p):
                    for c in range(p * n):
                        A[b * (p * n) + c, s] = V[b, s] * W[c, s]

            # Solve for U
            # T_reshaped[a, b*p*n + c] = sum_s A[b*p*n+c, s] * U[a, s]
            # So T_reshaped[a,:] = U[a,:] @ A.T
            # U[a,:] = T_reshaped[a,:] @ (A.T @ A)^{-1} @ A.T  (least squares)
            try:
                # Use lstsq for numerical stability
                A_T = A.T  # shape (r, m*p*n)
                # For each a, solve: A @ U[a,:]^T = T[a,:,:].reshape(-1)
                # Or equivalently: A.T @ U[a,:]^T = T[a,:,:].reshape(-1) is wrong
                # Let's be careful:
                # T[a,b,c] = sum_s U[a,s] * A[(b,c), s]
                # So T[a].reshape(-1) = A @ U[a].reshape(-1)  -- NO
                # T[a,b,c] = sum_s A[(b,c),s] * U[a,s]
                # So T[a].reshape(-1) = A @ U[a].reshape(-1)
                # Therefore U[a].reshape(-1) = pinv(A) @ T[a].reshape(-1)
                pinv_A = np.linalg.pinv(A)
                for a in range(n * m):
                    U[a, :] = pinv_A @ T[a, :, :].reshape(-1)
            except:
                return None

            # Similarly update V and W
            # For V: T[a,b,c] = sum_s V[b,s] * (U[a,s]*W[c,s])
            # B[(a,c), s] = U[a,s]*W[c,s]
            B = np.zeros((n * m * p * n, r))
            for s in range(r):
                for a in range(n * m):
                    for c in range(p * n):
                        B[a * (p * n) + c, s] = U[a, s] * W[c, s]
            pinv_B = np.linalg.pinv(B)
            for b in range(m * p):
                V[b, :] = pinv_B @ T[:, b, :].reshape(-1)

            # For W: T[a,b,c] = sum_s W[c,s] * (U[a,s]*V[b,s])
            # C[(a,b), s] = U[a,s]*V[b,s]
            C = np.zeros((n * m * m * p, r))
            for s in range(r):
                for a in range(n * m):
                    for b in range(m * p):
                        C[a * (m * p) + b, s] = U[a, s] * V[b, s]
            pinv_C = np.linalg.pinv(C)
            for c in range(p * n):
                W[c, :] = pinv_C @ T[:, :, c].reshape(-1)

        # Check if the decomposition is exact
        # T[a,b,c] = sum_s U[a,s]*V[b,s]*W[c,s]
        recon = np.einsum('as,bs,cs->abc', U, V, W)
        err = np.max(np.abs(recon - T))
        if err < 1e-6:
            # Round to nearest half-integer
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2
            recon_r = np.einsum('as,bs,cs->abc', U_r, V_r, W_r)
            err_r = np.max(np.abs(recon_r - T))
            if err_r < 1e-6:
                return (U_r.tolist(), V_r.tolist(), W_r.tolist())
        return None

    # Try ranks from 74 down to 50
    for r in range(74, 49, -1):
        if time.time() > deadline:
            break
        time_budget = min(deadline, time.time() + 20)  # 20s per rank attempt
        result = try_rank(r, time_budget)
        if result is not None:
            best_rank = r
            best_result = result
            break  # Found a decomposition, use the best (lowest) rank

    if best_result is not None:
        U, V, W = best_result
        return {"U": U, "V": V, "W": W}
    else:
        # Fall back to schoolbook
        U = [[0] * rank_schoolbook for _ in range(n * m)]
        V = [[0] * rank_schoolbook for _ in range(m * p)]
        W = [[0] * rank_schoolbook for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
