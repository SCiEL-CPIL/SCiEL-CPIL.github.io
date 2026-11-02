"""Matrix multiplication 3x4x4 tensor decomposition - rank minimization."""
import time
import numpy as np

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START

def _build_tensor():
    """Build the 3x4x4 matrix multiplication tensor T[i,j,k] = 1 if i<3, j<4, k<4."""
    n, m, p = DIMENSIONS
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
    return T

def _decompose_schoolbook():
    """Return schoolbook decomposition of rank n*m*p."""
    n, m, p = DIMENSIONS
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
    return U, V, W

def _verify(U, V, W):
    """Verify that the decomposition is exact."""
    n, m, p = DIMENSIONS
    T = _build_tensor()
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(val - T[i,j,k]) > 1e-6:
                    return False
    return True

def _try_rank_decomposition(rank, max_iters=200, seed=42):
    """Try to find a rank-r decomposition using gradient-based optimization."""
    n, m, p = DIMENSIONS
    T = _build_tensor()
    rng = np.random.RandomState(seed)
    
    # Initialize with random factors
    U = rng.randn(n * m, rank) * 0.1
    V = rng.randn(m * p, rank) * 0.1
    W = rng.randn(p * n, rank) * 0.1
    
    best_loss = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    lr = 0.05
    for it in range(max_iters):
        if time.time() > 10:  # time guard
            break
        
        # Compute tensor from current factors
        T_hat = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_hat[i, j, k] = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
        
        loss = np.sum((T_hat - T)**2)
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if loss < 1e-8:
            break
        
        # Compute gradients
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    err = T_hat[i, j, k] - T[i, j, k]
                    dU[i*m+j, :] += 2 * err * V[j*p+k, :] * W[k*n+i, :]
                    dV[j*p+k, :] += 2 * err * U[i*m+j, :] * W[k*n+i, :]
                    dW[k*n+i, :] += 2 * err * U[i*m+j, :] * V[j*p+k, :]
        
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
        lr = max(lr * 0.995, 0.001)
    
    # Verify best solution
    T_hat = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_hat[i, j, k] = np.sum(best_U[i*m+j, :] * best_V[j*p+k, :] * best_W[k*n+i, :])
    if np.max(np.abs(T_hat - T)) < 1e-6:
        return best_U, best_V, best_W
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 120  # safety margin
    
    # First, get schoolbook as fallback
    U_school, V_school, W_school = _decompose_schoolbook()
    
    # Try to find lower rank decomposition
    # For 3x4x4, known bounds: rank is between 19 and 48
    # Try ranks from 19 upward
    for rank in range(19, 49):
        if time.time() > deadline - 10:
            break
        
        result = _try_rank_decomposition(rank)
        if result is not None:
            U, V, W = result
            if _verify(U, V, W):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}

# EVOLVE-BLOCK-END
