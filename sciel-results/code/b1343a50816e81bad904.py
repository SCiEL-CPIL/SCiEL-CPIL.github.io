import numpy as np
import time

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def compute_residual(U, V, W, n, m, p):
    """Compute the residual of the tensor decomposition."""
    residual = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                val = np.dot(U[idx_u, :], V[idx_v, :] * W[idx_w, :])
                residual += (val - 1.0) ** 2
    return residual

def try_round_and_verify(U, V, W, n, m, p):
    """Try rounding to half-integers and verify exactness."""
    U_r = np.round(U * 2) / 2
    V_r = np.round(V * 2) / 2
    W_r = np.round(W * 2) / 2
    
    res_r = compute_residual(U_r, V_r, W_r, n, m, p)
    if res_r < 1e-9:
        return {
            "U": U_r.tolist(),
            "V": V_r.tolist(),
            "W": W_r.tolist()
        }
    return None

def gradient_polish(U, V, W, n, m, p, max_iter=50, lr=0.01):
    """Perform gradient descent polish to minimize residual."""
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    R = U.shape[1]
    
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    for it in range(max_iter):
        # Compute gradients
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    
                    # val = dot(U[idx_u], V[idx_v] * W[idx_w])
                    # d/dU[idx_u] = V[idx_v] * W[idx_w] * (val - 1)
                    # d/dV[idx_v] = U[idx_u] * W[idx_w] * (val - 1)
                    # d/dW[idx_w] = U[idx_u] * V[idx_v] * (val - 1)
                    
                    val = np.dot(U[idx_u, :], V[idx_v, :] * W[idx_w, :])
                    err = val - 1.0
                    
                    grad_U[idx_u, :] += V[idx_v, :] * W[idx_w, :] * err
                    grad_V[idx_v, :] += U[idx_u, :] * W[idx_w, :] * err
                    grad_W[idx_w, :] += U[idx_u, :] * V[idx_v, :] * err
        
        # Update
        U -= lr * grad_U
        V -= lr * grad_V
        W -= lr * grad_W
        
        # Check convergence
        residual = compute_residual(U, V, W, n, m, p)
        if residual < 1e-10:
            break
        
        # Adaptive learning rate
        if it % 10 == 9 and it > 0:
            lr *= 0.9
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    
    start_time = time.time()
    deadline = start_time + 110
    
    best_result = None
    best_rank = n * m * p  # Schoolbook rank
    
    # Precompute index arrays for efficiency
    indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                indices.append((idx_u, idx_v, idx_w))
    
    # Try ranks from 44 down to 30
    for R in range(44, 29, -1):
        if time.time() > deadline:
            break
        
        # Try multiple restarts for this rank
        num_restarts = 3 if R > 40 else 5
        
        for restart in range(num_restarts):
            if time.time() > deadline:
                break
            
            # Initialize with different strategies
            np.random.seed(42 + restart * 1000 + R)
            
            # Strategy 1: Random
            if restart % 3 == 0:
                U = np.random.rand(dim_U, R)
                V = np.random.rand(dim_V, R)
                W = np.random.rand(dim_W, R)
            # Strategy 2: Sparse random
            elif restart % 3 == 1:
                U = np.random.rand(dim_U, R)
                V = np.random.rand(dim_V, R)
                W = np.random.rand(dim_W, R)
                # Make some entries zero
                for mat in [U, V, W]:
                    mask = np.random.rand(*mat.shape) < 0.3
                    mat[mask] = 0
            # Strategy 3: Small values
            else:
                U = np.random.rand(dim_U, R) * 0.5
                V = np.random.rand(dim_V, R) * 0.5
                W = np.random.rand(dim_W, R) * 0.5
            
            # ALS iterations
            max_iter = 80
            for it in range(max_iter):
                if time.time() > deadline:
                    break
                
                # Update U
                for i in range(n):
                    for j in range(m):
                        idx_u = i * m + j
                        A = np.zeros((p, R))
                        for k in range(p):
                            idx_v = j * p + k
                            idx_w = k * n + i
                            A[k, :] = V[idx_v, :] * W[idx_w, :]
                        u, _, _, _ = np.linalg.lstsq(A, np.ones(p), rcond=None)
                        U[idx_u, :] = u
                        
                # Update V
                for j in range(m):
                    for k in range(p):
                        idx_v = j * p + k
                        B = np.zeros((n, R))
                        for i in range(n):
                            idx_u = i * m + j
                            idx_w = k * n + i
                            B[i, :] = U[idx_u, :] * W[idx_w, :]
                        v, _, _, _ = np.linalg.lstsq(B, np.ones(n), rcond=None)
                        V[idx_v, :] = v
                        
                # Update W
                for k in range(p):
                    for i in range(n):
                        idx_w = k * n + i
                        C = np.zeros((m, R))
                        for j in range(m):
                            idx_u = i * m + j
                            idx_v = j * p + k
                            C[j, :] = U[idx_u, :] * V[idx_v, :]
                        w, _, _, _ = np.linalg.lstsq(C, np.ones(m), rcond=None)
                        W[idx_w, :] = w
                
                # Check residual
                residual = compute_residual(U, V, W, n, m, p)
                
                if residual < 1e-8:
                    # Try to round
                    result = try_round_and_verify(U, V, W, n, m, p)
                    if result is not None:
                        return result
                    
                    # If rounding failed, try gradient polish
                    U_p, V_p, W_p = gradient_polish(U, V, W, n, m, p, max_iter=30, lr=0.005)
                    result = try_round_and_verify(U_p, V_p, W_p, n, m, p)
                    if result is not None:
                        return result
                    
                    # If still not working, break
                    break
                    
            # End ALS iterations
            
        # End restarts
        
    # If no better rank found, return schoolbook
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
# EVOLVE-BLOCK-END
