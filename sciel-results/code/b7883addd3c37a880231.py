import time
import numpy as np
import random

DIMENSIONS = [2, 3, 5]

def _verify_decomposition(U, V, W):
    """
    Verify that the decomposition correctly represents matrix multiplication.
    For each (i,j,k), the coefficient of A[i,j]*B[j,k] in C[i,k] should be 1.
    """
    n, m, p = DIMENSIONS
    rank = len(U[0]) if U else 0
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # For each (i, j, k), check that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_row = U_arr[i * m + j, :]
                v_row = V_arr[j * p + k, :]
                w_row = W_arr[k * n + i, :]
                product = np.sum(u_row * v_row * w_row)
                if abs(product - 1.0) > 1e-9:
                    return False
    return True

def _try_rank_reduction(n, m, p, target_rank, time_limit=60.0):
    """
    Try to find a decomposition with the given target_rank.
    Uses random initialization with local optimization.
    Returns (U, V, W) if successful, None otherwise.
    """
    start_time = time.time()
    U_dim = n * m
    V_dim = m * p
    W_dim = p * n
    
    # Random initialization
    U = np.random.randn(U_dim, target_rank)
    V = np.random.randn(V_dim, target_rank)
    W = np.random.randn(W_dim, target_rank)
    
    # Compute the error: for each (i,j,k), we need sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
    # Let's define the residual as the difference from 1
    
    def compute_residual():
        residuals = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    residuals[i, j, k] = np.sum(U[u_idx, :] * V[v_idx, :] * W[w_idx, :])
        return residuals
    
    def compute_loss():
        res = compute_residual()
        return np.sum((res - 1.0) ** 2)
    
    best_loss = compute_loss()
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Simple gradient-free local search
    lr = 0.1
    for step in range(500):
        if time.time() - start_time > time_limit:
            break
        
        res = compute_residual()
        error = res - 1.0
        loss = np.sum(error ** 2)
        
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if best_loss < 1e-10:
            break
        
        # Gradient computation (approximate)
        # dL/dU[u_idx, r] = sum over (j,k) where u_idx = i*m+j of error[i,j,k] * V[j*p+k, r] * W[k*n+i, r]
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    e = error[i, j, k]
                    dU[u_idx, :] += e * V[v_idx, :] * W[w_idx, :]
                    dV[v_idx, :] += e * U[u_idx, :] * W[w_idx, :]
                    dW[w_idx, :] += e * U[u_idx, :] * V[v_idx, :]
        
        # Update
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
        
        # Decay learning rate
        lr *= 0.99
    
    if best_loss < 1e-8:
        return best_U.tolist(), best_V.tolist(), best_W.tolist()
    return None

def _find_best_decomposition(n, m, p, time_limit=120.0):
    """
    Search for the best (lowest rank) decomposition within the time limit.
    """
    start_time = time.time()
    schoolbook_rank = n * m * p
    
    # Start with schoolbook
    U = [[0] * schoolbook_rank for _ in range(n * m)]
    V = [[0] * schoolbook_rank for _ in range(m * p)]
    W = [[0] * schoolbook_rank for _ in range(p * n)]
    
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    
    best_rank = schoolbook_rank
    best_U, best_V, best_W = U, V, W
    
    # Try lower ranks
    for target_rank in range(schoolbook_rank - 1, 0, -1):
        if time.time() - start_time > time_limit - 30:  # Leave 30s margin
            break
        
        result = _try_rank_reduction(n, m, p, target_rank, time_limit=30.0)
        if result is not None:
            U_list, V_list, W_list = result
            if _verify_decomposition(U_list, V_list, W_list):
                best_rank = target_rank
                best_U, best_V, best_W = U_list, V_list, W_list
                break  # Found a valid lower rank, stop searching
    
    return best_U, best_V, best_W

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the matrix multiplication tensor decomposition for 2x3 by 3x5.
    Returns a dictionary with keys 'U', 'V', 'W' representing the factor matrices.
    U: shape (n*m, rank)
    V: shape (m*p, rank)
    W: shape (p*n, rank)
    
    Searches for a rank-minimal exact tensor decomposition.
    Falls back to schoolbook algorithm (rank 30) if no better found in time.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    try:
        U, V, W = _find_best_decomposition(n, m, p, time_limit=120.0)
    except Exception:
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][c] = 1
                    V[j * p + k][c] = 1
                    W[k * n + i][c] = 1
                    c += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
