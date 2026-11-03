"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, tol=1e-6):
    """Check if the decomposition is exact (within tolerance)."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    # Reconstruct tensor T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # Compare with the matrix multiplication tensor: T[i,j,k] = 1 if (i,j,k) is a valid index
    # Actually, the tensor for matrix multiplication C = A * B where A is n x m, B is m x p, C is n x p
    # The tensor is T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # Wait, that's not right. The matrix multiplication tensor has T[i,j,k] = 1 for all valid (i,j,k).
    # Let me re-read the problem.
    # "exactly decompose the matrix multiplication tensor"
    # The matrix multiplication tensor T has T[i,j,k] = 1 for all i,j,k in their respective ranges.
    # So we need sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k.
    
    # Let's build the tensor from U, V, W
    T = np.zeros((n, m, p))
    for c in range(rank):
        uc = U[:, c]  # shape (n*m,)
        vc = V[:, c]  # shape (m*p,)
        wc = W[:, c]  # shape (p*n,)
        
        # Reshape to (n,m), (m,p), (p,n)
        uc_reshaped = uc.reshape(n, m)
        vc_reshaped = vc.reshape(m, p)
        wc_reshaped = wc.reshape(p, n)
        
        # The contribution of term c to T[i,j,k] is uc_reshaped[i,j] * vc_reshaped[j,k] * wc_reshaped[k,i]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i,j,k] += uc_reshaped[i,j] * vc_reshaped[j,k] * wc_reshaped[k,i]
    
    # The target tensor is all 1s
    target = np.ones((n, m, p))
    return np.allclose(T, target, atol=tol)

def _try_rank(rank, n, m, p, max_iters=2000, seed=42):
    """Try to find a decomposition of the given rank."""
    rng = np.random.RandomState(seed)
    # Initialize with random values
    U = rng.randn(n*m, rank)
    V = rng.randn(m*p, rank)
    W = rng.randn(p*n, rank)
    
    best_error = np.inf
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iters):
        # Compute gradient for each factor
        # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        # dT/dU[i*m+j,c] = V[j*p+k,c] * W[k*n+i,c]
        # dT/dV[j*p+k,c] = U[i*m+j,c] * W[k*n+i,c]
        # dT/dW[k*n+i,c] = U[i*m+j,c] * V[j*p+k,c]
        
        # Build the tensor from current U, V, W
        T = np.zeros((n, m, p, rank))
        for c in range(rank):
            uc = U[:, c].reshape(n, m)
            vc = V[:, c].reshape(m, p)
            wc = W[:, c].reshape(p, n)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i,j,k,c] = uc[i,j] * vc[j,k] * wc[k,i]
        
        # Compute gradients
        grad_U = np.zeros((n*m, rank))
        grad_V = np.zeros((m*p, rank))
        grad_W = np.zeros((p*n, rank))
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for c in range(rank):
                        grad_U[i*m+j, c] = V[j*p+k, c] * W[k*n+i, c]
                        grad_V[j*p+k, c] = U[i*m+j, c] * W[k*n+i, c]
                        grad_W[k*n+i, c] = U[i*m+j, c] * V[j*p+k, c]
        
        # The error is (T - 1)^2, so the gradient of the error is 2*(T-1)*dT/dparam
        # We want to minimize sum_{i,j,k} (T[i,j,k] - 1)^2
        # dError/dU[i*m+j,c] = 2 * sum_{i,j,k} (T[i,j,k] - 1) * dT/dU[i*m+j,c]
        # But dT/dU[i*m+j,c] is nonzero only for the specific (i,j,k,c)
        # Actually, T[i,j,k] = sum_c T[i,j,k,c], so dT[i,j,k]/dU[i*m+j,c] = V[j*p+k,c] * W[k*n+i,c]
        
        # Let's compute the gradient properly
        # Error = sum_{i,j,k} (sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] - 1)^2
        # dError/dU[i*m+j,c] = 2 * (sum_c' U[i*m+j,c']*V[j*p+k,c']*W[k*n+i,c'] - 1) * V[j*p+k,c] * W[k*n+i,c]
        
        # This is getting complex. Let's use a simpler approach: just do gradient descent on the error.
        
        # Compute T
        T_current = np.zeros((n, m, p))
        for c in range(rank):
            uc = U[:, c].reshape(n, m)
            vc = V[:, c].reshape(m, p)
            wc = W[:, c].reshape(p, n)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_current[i,j,k] += uc[i,j] * vc[j,k] * wc[k,i]
        
        error = np.sum((T_current - 1.0)**2)
        if error < best_error:
            best_error = error
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if error < 1e-8:
            break
        
        # Compute gradients
        grad_U = np.zeros((n*m, rank))
        grad_V = np.zeros((m*p, rank))
        grad_W = np.zeros((p*n, rank))
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    diff = T_current[i,j,k] - 1.0
                    for c in range(rank):
                        uc_val = U[i*m+j, c]
                        vc_val = V[j*p+k, c]
                        wc_val = W[k*n+i, c]
                        grad_U[i*m+j, c] += 2 * diff * vc_val * wc_val
                        grad_V[j*p+k, c] += 2 * diff * uc_val * wc_val
                        grad_W[k*n+i, c] += 2 * diff * uc_val * vc_val
        
        # Learning rate
        lr = 0.01
        U -= lr * grad_U
        V -= lr * grad_V
        W -= lr * grad_W
    
    if best_error < 1e-6:
        return best_U, best_V, best_W
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 120.0  # seconds
    
    # Try ranks from low to high
    for rank in range(1, n*m*p + 1):
        if time.time() - start_time > deadline - 10:  # Safety margin
            break
        
        result = _try_rank(rank, n, m, p, max_iters=1000, seed=42 + rank)
        if result is not None:
            U, V, W = result
            # Convert to lists
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback to schoolbook
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
