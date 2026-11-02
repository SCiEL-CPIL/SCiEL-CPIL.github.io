"""Rewritten program for matrix multiplication 2x4 * 4x5 tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Schoolbook decomposition
    rank_school = n * m * p  # 2*4*5 = 40
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1

    start_time = time.time()
    deadline = 140  # leave 10s margin before 150s kill

    best_rank = rank_school
    best_U = U_school.copy()
    best_V = V_school.copy()
    best_W = W_school.copy()

    def verify_decomp_fast(U, V, W, tol=1e-6):
        """Faster verification using numpy einsum."""
        rank = U.shape[1]
        U_idx = U.reshape(n, m, rank)
        V_idx = V.reshape(m, p, rank)
        W_idx = W.reshape(p, n, rank)
        T = np.einsum('ijc,jkc,kic->ijk', U_idx, V_idx, W_idx)
        if not np.allclose(T, np.ones((n, m, p)), atol=tol):
            return False
        return True

    def als_decompose(target_rank, max_iters=200, tol=1e-7, seed=42):
        """Alternating Least Squares for tensor decomposition."""
        rng = np.random.RandomState(seed)
        
        U = rng.randn(n * m, target_rank) * 0.5
        V = rng.randn(m * p, target_rank) * 0.5
        W = rng.randn(p * n, target_rank) * 0.5
        
        for iteration in range(max_iters):
            if time.time() > deadline:
                break
                
            # Fix V and W, solve for U
            for i in range(n):
                for j in range(m):
                    M = np.zeros((p, target_rank))
                    for k in range(p):
                        M[k, :] = V[j * p + k, :] * W[k * n + i, :]
                    try:
                        U[i * m + j, :] = np.linalg.lstsq(M, np.ones(p), rcond=None)[0]
                    except:
                        U[i * m + j, :] = np.linalg.pinv(M) @ np.ones(p)
            
            # Fix U and W, solve for V
            for j in range(m):
                for k in range(p):
                    M = np.zeros((n, target_rank))
                    for i in range(n):
                        M[i, :] = U[i * m + j, :] * W[k * n + i, :]
                    try:
                        V[j * p + k, :] = np.linalg.lstsq(M, np.ones(n), rcond=None)[0]
                    except:
                        V[j * p + k, :] = np.linalg.pinv(M) @ np.ones(n)
            
            # Fix U and V, solve for W
            for k in range(p):
                for i in range(n):
                    M = np.zeros((m, target_rank))
                    for j in range(m):
                        M[j, :] = U[i * m + j, :] * V[j * p + k, :]
                    try:
                        W[k * n + i, :] = np.linalg.lstsq(M, np.ones(m), rcond=None)[0]
                    except:
                        W[k * n + i, :] = np.linalg.pinv(M) @ np.ones(m)
            
            # Check convergence
            if verify_decomp_fast(U, V, W, tol=1e-5):
                return U, V, W, True
        
        # Check if we got a valid decomposition
        if verify_decomp_fast(U, V, W, tol=1e-4):
            return U, V, W, True
        return U, V, W, False

    # Try various ranks
    target_ranks = [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]
    
    for target_rank in target_ranks:
        if time.time() > deadline:
            break
        
        if target_rank >= best_rank:
            continue
            
        # Try multiple seeds
        found = False
        for seed in range(10):
            if time.time() > deadline:
                break
            if found:
                break
                
            U, V, W, valid = als_decompose(target_rank, max_iters=300, tol=1e-8, seed=seed)
            if valid:
                best_rank = target_rank
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
                found = True

    # Convert to lists for return
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
