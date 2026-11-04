"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def _verify_decomposition_fast(U, V, W, n, m, p):
    """Faster verification using matrix operations."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition with rank = n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}

def _try_random_rank(n, m, p, rank, max_iter=500, seed=None):
    """Try to find a rank-`rank` decomposition via random initialization and gradient descent."""
    if seed is not None:
        np.random.seed(seed)
    
    U = np.random.randn(n * m, rank)
    V = np.random.randn(m * p, rank)
    W = np.random.randn(p * n, rank)
    
    best_err = float('inf')
    lr = 0.01
    
    for it in range(max_iter):
        # Compute gradients
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # dT/dU[i*m+j, r] = V[j*p+k, r] * W[k*n+i, r]
                    # dT/dV[j*p+k, r] = U[i*m+j, r] * W[k*n+i, r]
                    # dT/dW[k*n+i, r] = U[i*m+j, r] * V[j*p+k, r]
                    diff = 1.0 - np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    err += diff * diff
                    dU[i*m+j, :] += diff * V[j*p+k, :] * W[k*n+i, :]
                    dV[j*p+k, :] += diff * U[i*m+j, :] * W[k*n+i, :]
                    dW[k*n+i, :] += diff * U[i*m+j, :] * V[j*p+k, :]
        
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if err < 1e-10:
            break
    
    if best_err < 1e-9:
        return {"U": best_U, "V": best_V, "W": best_W}
    return None

def _polish_decomposition(U, V, W, n, m, p, max_iter=1000, tol=1e-12):
    """Polish a near-exact decomposition using Newton's method on the tensor equations."""
    rank = U.shape[1]
    U = U.astype(float).copy()
    V = V.astype(float).copy()
    W = W.astype(float).copy()
    
    for it in range(max_iter):
        # Compute current error
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    err += (1.0 - s) ** 2
        if err < tol:
            break
        
        # Newton step: solve linear system for corrections
        # This is expensive, so we use a simplified Gauss-Seidel-like approach
        # Or use a smaller learning rate gradient step with line search
        
        lr = 0.05
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    diff = 1.0 - np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    dU[i*m+j, :] += diff * V[j*p+k, :] * W[k*n+i, :]
                    dV[j*p+k, :] += diff * U[i*m+j, :] * W[k*n+i, :]
                    dW[k*n+i, :] += diff * U[i*m+j, :] * V[j*p+k, :]
        
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
    
    # Final check
    err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                err += (1.0 - s) ** 2
    
    if err < 1e-9:
        return {"U": U, "V": V, "W": W}
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Start with schoolbook (rank = 90)
    best_result = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try to find lower rank decompositions
    lower_bound = max(n * m, m * p, p * n)
    
    # Try ranks from lower_bound to best_rank - 1
    # For each rank, try multiple random initializations
    for rank in range(lower_bound, best_rank):
        if time.time() > deadline - 10:
            break
        
        found = False
        # Try up to 5 random initializations per rank
        for attempt in range(5):
            if time.time() > deadline - 5:
                break
            
            seed = rank * 100 + attempt
            result = _try_random_rank(n, m, p, rank, max_iter=300, seed=seed)
            if result is not None:
                if _verify_decomposition_fast(result["U"], result["V"], result["W"], n, m, p):
                    best_result = result
                    best_rank = rank
                    found = True
                    break
        
        if found:
            break
    
    # If we have time, try to polish the best result to ensure exactness
    if time.time() < deadline - 2:
        polished = _polish_decomposition(best_result["U"], best_result["V"], best_result["W"], n, m, p)
        if polished is not None:
            best_result = polished
    
    # Convert to lists
    return {"U": best_result["U"].tolist(), "V": best_result["V"].tolist(), "W": best_result["W"].tolist()}
# EVOLVE-BLOCK-END
