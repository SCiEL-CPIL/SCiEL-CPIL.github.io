import time
import numpy as np

DIMENSIONS = [3, 4, 5]
_START_TIME = time.time()

def _check_time(deadline):
    return time.time() > deadline

def _build_schoolbook(n, m, p):
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    A = np.arange(n * m, dtype=float).reshape(n, m)
    B = np.arange(m * p, dtype=float).reshape(m, p)
    C_expected = A @ B
    
    U_arr = np.array(U)
    V_arr = np.array(V)
    W_arr = np.array(W)
    rank = U_arr.shape[1]
    
    C_computed = np.zeros((n, p), dtype=float)
    for r in range(rank):
        term_U = np.zeros(n, dtype=float)
        for i in range(n):
            for j in range(m):
                term_U[i] += U_arr[i * m + j, r] * A[i, j]
        
        term_V = np.zeros(p, dtype=float)
        for k in range(p):
            for j in range(m):
                term_V[k] += V_arr[j * p + k, r] * B[j, k]
        
        for i in range(n):
            for k in range(p):
                C_computed[i, k] += W_arr[k * n + i, r] * term_U[i] * term_V[k]
    
    return np.allclose(C_computed, C_expected, atol=tol)

def _als_decomposition(n, m, p, target_rank, max_iters=500, tol=1e-12, seed=0):
    """Alternating least squares to find rank-target_rank decomposition."""
    np.random.seed(seed)
    
    U = np.random.randn(n * m, target_rank) * 0.1
    V = np.random.randn(m * p, target_rank) * 0.1
    W = np.random.randn(p * n, target_rank) * 0.1
    
    for iteration in range(max_iters):
        # Update U
        for i in range(n):
            for j in range(m):
                A_mat = np.zeros((p, target_rank))
                for k in range(p):
                    for c in range(target_rank):
                        A_mat[k, c] = V[j * p + k, c] * W[k * n + i, c]
                try:
                    ATA = A_mat.T @ A_mat
                    ATb = A_mat.T @ np.ones(p)
                    U[i * m + j, :] = np.linalg.solve(ATA + 1e-10 * np.eye(target_rank), ATb)
                except np.linalg.LinAlgError:
                    pass
        
        # Update V
        for j in range(m):
            for k in range(p):
                A_mat = np.zeros((n, target_rank))
                for i in range(n):
                    for c in range(target_rank):
                        A_mat[i, c] = U[i * m + j, c] * W[k * n + i, c]
                try:
                    ATA = A_mat.T @ A_mat
                    ATb = A_mat.T @ np.ones(n)
                    V[j * p + k, :] = np.linalg.solve(ATA + 1e-10 * np.eye(target_rank), ATb)
                except np.linalg.LinAlgError:
                    pass
        
        # Update W
        for k in range(p):
            for i in range(n):
                A_mat = np.zeros((m, target_rank))
                for j in range(m):
                    for c in range(target_rank):
                        A_mat[j, c] = U[i * m + j, c] * V[j * p + k, c]
                try:
                    ATA = A_mat.T @ A_mat
                    ATb = A_mat.T @ np.ones(m)
                    W[k * n + i, :] = np.linalg.solve(ATA + 1e-10 * np.eye(target_rank), ATb)
                except np.linalg.LinAlgError:
                    pass
        
        # Check convergence
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(target_rank):
                        val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    error += (1.0 - val) ** 2
        
        if error < tol:
            return U.tolist(), V.tolist(), W.tolist(), True
    
    return U.tolist(), V.tolist(), W.tolist(), False

def construct():
    n, m, p = DIMENSIONS
    deadline = _START_TIME + 110
    
    best = _build_schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try to find lower rank decompositions
    for target_rank in range(40, 61):
        if _check_time(deadline - 15):
            break
        
        found = False
        for seed in range(20):
            if _check_time(deadline - 10):
                break
            
            U_list, V_list, W_list, converged = _als_decomposition(
                n, m, p, target_rank, max_iters=200, tol=1e-10, seed=seed
            )
            
            if converged and _verify_decomposition(U_list, V_list, W_list, n, m, p):
                best = {"U": U_list, "V": V_list, "W": W_list}
                best_rank = target_rank
                found = True
                break
        
        if found:
            break
    
    return best
