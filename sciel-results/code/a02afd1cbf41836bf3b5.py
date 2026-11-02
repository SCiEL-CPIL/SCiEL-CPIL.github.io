"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    def check_solution(U, V, W, rank, tol=1e-8):
        """Check if the decomposition is valid."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def als_solve(rank, seed=0, max_iter=1000, tol=1e-12):
        """ALS for the tensor decomposition."""
        np.random.seed(seed)
        U = np.random.randn(n * m, rank) * 0.5
        V = np.random.randn(m * p, rank) * 0.5
        W = np.random.randn(p * n, rank) * 0.5
        
        for it in range(max_iter):
            if time.time() > deadline - 5:
                return None
            
            # Update U
            new_U = np.zeros_like(U)
            for i in range(n):
                for j in range(m):
                    # A[k, r] = V[j*p+k, r] * W[k*n+i, r] for k in range(p)
                    # Solve A @ x = ones(p)
                    A_mat = (V[j * p:(j + 1) * p, :] * W[np.arange(p) * n + i, :])
                    try:
                        new_U[i * m + j, :] = np.linalg.lstsq(A_mat, np.ones(p), rcond=None)[0]
                    except:
                        new_U[i * m + j, :] = np.linalg.pinv(A_mat) @ np.ones(p)
            U = new_U
            
            # Update V
            new_V = np.zeros_like(V)
            for j in range(m):
                for k in range(p):
                    # B[i, r] = U[i*m+j, r] * W[k*n+i, r] for i in range(n)
                    # Solve B @ x = ones(n)
                    B_mat = (U[np.arange(n) * m + j, :] * W[k * n + np.arange(n), :])
                    try:
                        new_V[j * p + k, :] = np.linalg.lstsq(B_mat, np.ones(n), rcond=None)[0]
                    except:
                        new_V[j * p + k, :] = np.linalg.pinv(B_mat) @ np.ones(n)
            V = new_V
            
            # Update W
            new_W = np.zeros_like(W)
            for k in range(p):
                for i in range(n):
                    # C[j, r] = U[i*m+j, r] * V[j*p+k, r] for j in range(m)
                    # Solve C @ x = ones(m)
                    C_mat = (U[i * m + np.arange(m), :] * V[np.arange(m) * p + k, :])
                    try:
                        new_W[k * n + i, :] = np.linalg.lstsq(C_mat, np.ones(m), rcond=None)[0]
                    except:
                        new_W[k * n + i, :] = np.linalg.pinv(C_mat) @ np.ones(m)
            W = new_W
            
            # Quick error check every 10 iterations
            if it % 10 == 0 or it == max_iter - 1:
                max_err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                            max_err = max(max_err, abs(val - 1.0))
                if max_err < tol:
                    return (U, V, W)
        
        # Final check
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                    max_err = max(max_err, abs(val - 1.0))
        if max_err < 1e-6:
            return (U, V, W)
        return None
    
    best_result = None
    
    # Try ranks from low to high
    for rank in range(16, n * m * p + 1):
        if time.time() > deadline - 10:
            break
        found = False
        for seed in range(50):
            if time.time() > deadline - 5:
                break
            result = als_solve(rank, seed=seed, max_iter=600, tol=1e-12)
            if result is not None:
                U, V, W = result
                if check_solution(U, V, W, rank, tol=1e-6):
                    best_result = (U, V, W, rank)
                    found = True
                    break
        if found:
            break
    
    if best_result is None:
        # Schoolbook fallback
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
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    U, V, W, rank = best_result
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
