import numpy as np
import time

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115  # safety margin before 150s kill

    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}, rank

    best_result, best_rank = schoolbook()

    def compute_residual(U, V, W, r):
        """Vectorized residual computation. Returns max |error|."""
        # U: (n*m, r), V: (m*p, r), W: (p*n, r)
        # For each (i,j,k): sum_t W[k*n+i,t] * U[i*m+j,t] * V[j*p+k,t] should be 1
        # Reshape to tensors: U_t[i,j,t], V_t[j,k,t], W_t[k,i,t]
        U_t = U.reshape(n, m, r)
        V_t = V.reshape(m, p, r)
        W_t = W.reshape(p, n, r)
        
        # Compute T[i,j,k] = sum_t W_t[k,i,t] * U_t[i,j,t] * V_t[j,k,t]
        # Using einsum: T[i,j,k] = W_t[k,i,t] * U_t[i,j,t] * V_t[j,k,t]
        T = np.einsum('kit,ijt,jkt->ijk', W_t, U_t, V_t)
        return np.max(np.abs(T - 1.0))

    def als_solve(r, U, V, W, max_iters=150, tol=1e-10, polish_tol=1e-14):
        """ALS with final polish pass."""
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
            
            # Update W: for each (k,i), solve A @ w = 1
            # A[j,t] = U[i*m+j,t] * V[j*p+k,t]
            for i in range(n):
                for k in range(p):
                    # A shape (m, r)
                    A = U[i * m:(i + 1) * m, :] * V[k * p:(k + 1) * p, :].T  # (m, r)
                    try:
                        w, _, _, _ = np.linalg.lstsq(A, np.ones(m), rcond=None)
                        W[k * n + i, :] = w
                    except:
                        pass

            # Update U: for each (i,j), solve A @ u = 1
            # A[k,t] = W[k*n+i,t] * V[j*p+k,t]
            for i in range(n):
                for j in range(m):
                    A = W[:, i] * V[j * p:(j + 1) * p, :].T  # (p, r)
                    try:
                        u, _, _, _ = np.linalg.lstsq(A, np.ones(p), rcond=None)
                        U[i * m + j, :] = u
                    except:
                        pass

            # Update V: for each (j,k), solve A @ v = 1
            # A[i,t] = U[i*m+j,t] * W[k*n+i,t]
            for j in range(m):
                for k in range(p):
                    A = U[:, j] * W[k * n:(k + 1) * n, :].T  # (n, r)
                    try:
                        v, _, _, _ = np.linalg.lstsq(A, np.ones(n), rcond=None)
                        V[j * p + k, :] = v
                    except:
                        pass

            err = compute_residual(U, V, W, r)
            if err < tol:
                # Polish pass: more iterations with tighter tolerance
                for pit in range(50):
                    if time.time() > deadline - 10:
                        break
                    for i in range(n):
                        for k in range(p):
                            A = U[i * m:(i + 1) * m, :] * V[k * p:(k + 1) * p, :].T
                            try:
                                w, _, _, _ = np.linalg.lstsq(A, np.ones(m), rcond=None)
                                W[k * n + i, :] = w
                            except:
                                pass
                    for i in range(n):
                        for j in range(m):
                            A = W[:, i] * V[j * p:(j + 1) * p, :].T
                            try:
                                u, _, _, _ = np.linalg.lstsq(A, np.ones(p), rcond=None)
                                U[i * m + j, :] = u
                            except:
                                pass
                    for j in range(m):
                        for k in range(p):
                            A = U[:, j] * W[k * n:(k + 1) * n, :].T
                            try:
                                v, _, _, _ = np.linalg.lstsq(A, np.ones(n), rcond=None)
                                V[j * p + k, :] = v
                            except:
                                pass
                    err = compute_residual(U, V, W, r)
                    if err < polish_tol:
                        break
                return U, V, W, err

        return U, V, W, compute_residual(U, V, W, r)

    def try_rank(r, n_attempts=8):
        """Try to find a valid decomposition at rank r with multiple strategies."""
        for attempt in range(n_attempts):
            if time.time() > deadline - 15:
                return None, None, None, None

            if attempt == 0:
                # Sparse random init: each factor has mostly zeros
                U = np.zeros((n * m, r))
                V = np.zeros((m * p, r))
                W = np.zeros((p * n, r))
                for t in range(r):
                    # Each term picks random (i,j), (j,k), (k,i)
                    i1 = np.random.randint(n)
                    j1 = np.random.randint(m)
                    k1 = np.random.randint(p)
                    U[i1 * m + j1, t] = 1.0
                    V[j1 * p + k1, t] = 1.0
                    W[k1 * n + i1, t] = 1.0
                    # Add some cross terms
                    if np.random.rand() < 0.3:
                        i2 = np.random.randint(n)
                        j2 = np.random.randint(m)
                        U[i2 * m + j2, t] = 0.5
            elif attempt == 1:
                # Rank-1 structured: use outer products of random vectors
                U = np.random.randn(n * m, r) * 0.5
                V = np.random.randn(m * p, r) * 0.5
                W = np.random.randn(p * n, r) * 0.5
            elif attempt == 2:
                # Binary init
                U = (np.random.rand(n * m, r) > 0.5).astype(float)
                V = (np.random.rand(m * p, r) > 0.5).astype(float)
                W = (np.random.rand(p * n, r) > 0.5).astype(float)
                # Ensure at least some non-zero
                for t in range(r):
                    if np.all(U[:, t] == 0):
                        U[np.random.randint(n * m), t] = 1
                    if np.all(V[:, t] == 0):
                        V[np.random.randint(m * p), t] = 1
                    if np.all(W[:, t] == 0):
                        W[np.random.randint(p * n), t] = 1
            elif attempt == 3:
                # Gaussian with larger scale
                U = np.random.randn(n * m, r)
                V = np.random.randn(m * p, r)
                W = np.random.randn(p * n, r)
            elif attempt == 4:
                # Block-structured: group indices
                U = np.zeros((n * m, r))
                V = np.zeros((m * p, r))
                W = np.zeros((p * n, r))
                for t in range(r):
                    i1 = t % n
                    j1 = t % m
                    k1 = t % p
                    U[i1 * m + j1, t] = 1.0
                    V[j1 * p + k1, t] = 1.0
                    W[k1 * n + i1, t] = 1.0
                    # Add noise
                    U[i1 * m + j1, t] += np.random.randn() * 0.1
                    V[j1 * p + k1, t] += np.random.randn() * 0.1
                    W[k1 * n + i1, t] += np.random.randn() * 0.1
            elif attempt == 5:
                # Random sparse with density 0.3
                U = (np.random.rand(n * m, r) < 0.3).astype(float) * np.random.randn(n * m, r)
                V = (np.random.rand(m * p, r) < 0.3).astype(float) * np.random.randn(m * p, r)
                W = (np.random.rand(p * n, r) < 0.3).astype(float) * np.random.randn(p * n, r)
            elif attempt == 6:
                # Scaled random
                U = np.random.randn(n * m, r) * 0.3
                V = np.random.randn(m * p, r) * 0.3
                W = np.random.randn(p * n, r) * 0.3
            else:
                # Very sparse: one non-zero per row
                U = np.zeros((n * m, r))
                V = np.zeros((m * p, r))
                W = np.zeros((p * n, r))
                for row in range(n * m):
                    col = np.random.randint(r)
                    U[row, col] = np.random.choice([-1.0, 1.0])
                for row in range(m * p):
                    col = np.random.randint(r)
                    V[row, col] = np.random.choice([-1.0, 1.0])
                for row in range(p * n):
                    col = np.random.randint(r)
                    W[row, col] = np.random.choice([-1.0, 1.0])

            U, V, W, err = als_solve(r, U, V, W, max_iters=120, tol=1e-10, polish_tol=1e-14)

            if err < 1e-8:
                return U, V, W, err

        return None, None, None, None

    # Search ranks: known bounds suggest rank <= 60 for this size
    # Try from 60 downward, with more attempts at promising ranks
    search_ranks = list(range(60, 24, -1))
    
    for r in search_ranks:
        if time.time() > deadline - 20:
            break

        # More attempts for smaller ranks (harder but more valuable)
        n_attempts = 10 if r <= 40 else 8

        U, V, W, err = try_rank(r, n_attempts)

        if U is not None and err < 1e-8:
            best_rank = r
            best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            # Continue searching for even lower rank

    return best_result
# EVOLVE-BLOCK-END
