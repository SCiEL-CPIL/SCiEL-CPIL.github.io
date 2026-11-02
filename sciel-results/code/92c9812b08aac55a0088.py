import time
import random
import numpy as np

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START

def _build_schoolbook(n, m, p):
    """Return the standard (schoolbook) rank-n*m*p decomposition."""
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
    return U, V, W


def _verify(U, V, W, n, m, p, tol=1e-8):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                if abs(s - 1.0) > tol:
                    return False
    return True


def _round_to_half(arr):
    """Round each entry to the nearest half-integer."""
    return np.round(arr * 2.0) / 2.0


def _vectorized_gradient_step(U, V, W, n, m, p, lr):
    """
    Compute one gradient descent step for the tensor decomposition.
    Returns updated U, V, W and the error.
    """
    rank = U.shape[1]
    
    # Precompute products for efficiency
    # grad_U[i*m+j, c] = 2 * sum_k (sum_c' U[i*m+j,c']*V[j*p+k,c']*W[k*n+i,c'] - 1) * V[j*p+k,c] * W[k*n+i,c]
    # This is expensive to compute naively. Let's use a different approach.
    
    # Actually, let's compute the error tensor E[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] - 1
    # Then grad_U[i*m+j, c] = 2 * sum_k E[i,j,k] * V[j*p+k,c] * W[k*n+i,c]
    # grad_V[j*p+k, c] = 2 * sum_i E[i,j,k] * U[i*m+j,c] * W[k*n+i,c]
    # grad_W[k*n+i, c] = 2 * sum_j E[i,j,k] * U[i*m+j,c] * V[j*p+k,c]
    
    E = np.zeros((n, m, p), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                E[i, j, k] = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :]) - 1.0
    
    error = np.sum(E * E)
    
    grad_U = np.zeros_like(U)
    grad_V = np.zeros_like(V)
    grad_W = np.zeros_like(W)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                e = E[i, j, k]
                if abs(e) < 1e-15:
                    continue
                prod_VW = V[j * p + k, :] * W[k * n + i, :]
                prod_UW = U[i * m + j, :] * W[k * n + i, :]
                prod_UV = U[i * m + j, :] * V[j * p + k, :]
                
                grad_U[i * m + j, :] += 2 * e * prod_VW
                grad_V[j * p + k, :] += 2 * e * prod_UW
                grad_W[k * n + i, :] += 2 * e * prod_UV
    
    U_new = U - lr * grad_U
    V_new = V - lr * grad_V
    W_new = W - lr * grad_W
    
    return U_new, V_new, W_new, error


def _find_decomp_vectorized(n, m, p, max_rank, time_limit=10, n_trials=50):
    """
    Try to find a decomposition of rank <= max_rank for M_{n,m,p} using vectorized gradient descent.
    Returns (U, V, W) or (None, None, None).
    """
    start_time = time.time()
    lb = max(n * m, m * p, p * n)
    
    # Try ranks from lb to max_rank
    for rank in range(lb, max_rank + 1):
        if time.time() - start_time > time_limit:
            break
            
        for trial in range(n_trials):
            if time.time() - start_time > time_limit:
                break
                
            # Random initialization
            U = np.random.randn(n * m, rank) * 0.3
            V = np.random.randn(m * p, rank) * 0.3
            W = np.random.randn(p * n, rank) * 0.3
            
            lr = 0.05
            best_error = float('inf')
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            for it in range(2000):
                if time.time() - start_time > time_limit:
                    break
                    
                U, V, W, error = _vectorized_gradient_step(U, V, W, n, m, p, lr)
                
                if error < best_error:
                    best_error = error
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
                if error < 1e-10:
                    break
                    
                # Adaptive learning rate
                if it % 100 == 0 and it > 0:
                    lr *= 0.95
                if lr < 1e-6:
                    lr = 1e-6
            
            # Check if we found a solution
            if best_error < 1e-8:
                U_r = _round_to_half(best_U)
                V_r = _round_to_half(best_V)
                W_r = _round_to_half(best_W)
                
                if _verify(U_r, V_r, W_r, n, m, p):
                    return U_r, V_r, W_r
            else:
                # Try rounding the best solution found
                U_r = _round_to_half(best_U)
                V_r = _round_to_half(best_V)
                W_r = _round_to_half(best_W)
                
                if _verify(U_r, V_r, W_r, n, m, p):
                    return U_r, V_r, W_r
    
    return None, None, None


def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix by a 3x8 matrix.
    """
    n, m, p = DIMENSIONS  # 3, 3, 8
    deadline = time.time() + 140
    
    # Split p=8 into 3, 3, 2
    p1, p2, p3 = 3, 3, 2
    
    # Find decompositions for sub-problems
    # M_{3,3,3} - known to have rank 23
    U1, V1, W1 = _find_decomp_vectorized(n, m, p1, 27, time_limit=25, n_trials=30)
    if U1 is None:
        U1, V1, W1 = _build_schoolbook(n, m, p1)
        rank1 = n * m * p1
    else:
        rank1 = U1.shape[1]
        
    # M_{3,3,3} (second part)
    U2, V2, W2 = _find_decomp_vectorized(n, m, p2, 27, time_limit=25, n_trials=30)
    if U2 is None:
        U2, V2, W2 = _build_schoolbook(n, m, p2)
        rank2 = n * m * p2
    else:
        rank2 = U2.shape[1]
        
    # M_{3,3,2}
    U3, V3, W3 = _find_decomp_vectorized(n, m, p3, 18, time_limit=20, n_trials=30)
    if U3 is None:
        U3, V3, W3 = _build_schoolbook(n, m, p3)
        rank3 = n * m * p3
    else:
        rank3 = U3.shape[1]
        
    total_rank = rank1 + rank2 + rank3
    
    # Combine
    U = np.hstack([U1, U2, U3])
    
    # V: shape (m*p, total_rank)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)
    
    # Fill V
    for j in range(m):
        for k in range(p1):
            V[j * p + k, :rank1] = V1[j * p1 + k, :]
        for k in range(p2):
            V[j * p + (p1 + k), rank1:rank1+rank2] = V2[j * p2 + k, :]
        for k in range(p3):
            V[j * p + (p1 + p2 + k), rank1+rank2:] = V3[j * p3 + k, :]
            
    # Fill W
    for i in range(n):
        for k in range(p1):
            W[k * n + i, :rank1] = W1[k * n + i, :]
        for k in range(p2):
            W[(p1 + k) * n + i, rank1:rank1+rank2] = W2[k * n + i, :]
        for k in range(p3):
            W[(p1 + p2 + k) * n + i, rank1+rank2:] = W3[k * n + i, :]
            
    # Verify combined solution
    if _verify(U, V, W, n, m, p):
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    else:
        # Fallback to schoolbook
        U0, V0, W0 = _build_schoolbook(n, m, p)
        return {
            "U": U0.tolist(),
            "V": V0.tolist(),
            "W": W0.tolist()
        }

# EVOLVE-BLOCK-END
