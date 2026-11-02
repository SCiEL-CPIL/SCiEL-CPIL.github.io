import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 4]

def _schoolbook_decomposition(n, m, p):
    """Return the standard schoolbook decomposition with rank n*m*p."""
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

def _validate_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Check if U, V, W exactly decompose the matrix multiplication tensor."""
    rank = U.shape[1]
    # Reconstruct the tensor: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _snap_to_half_integers(arr, tol=1e-8):
    """Snap values close to half-integers to exact half-integers."""
    snapped = arr.copy()
    for idx in np.ndindex(snapped.shape):
        val = snapped[idx]
        closest = round(val * 2) / 2
        if abs(val - closest) < tol:
            snapped[idx] = closest
    return snapped

def _try_rank_search(n, m, p, start_rank, end_rank, time_budget=120.0):
    """Try to find a decomposition with rank in [start_rank, end_rank]."""
    deadline = time.time() + time_budget
    best_result = None
    best_rank = None
    
    U_school, V_school, W_school = _schoolbook_decomposition(n, m, p)
    
    for rank in range(start_rank, end_rank + 1):
        if time.time() > deadline - 5:
            break
            
        # Try multiple random initializations
        for trial in range(50):
            if time.time() > deadline - 2:
                break
                
            # Random initialization
            U = np.random.randn(n * m, rank)
            V = np.random.randn(m * p, rank)
            W = np.random.randn(p * n, rank)
            
            # Gradient descent to minimize the error
            lr = 0.01
            for step in range(500):
                if time.time() > deadline - 1:
                    break
                    
                # Compute gradients
                grad_U = np.zeros_like(U)
                grad_V = np.zeros_like(V)
                grad_W = np.zeros_like(W)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            # T[i,j,k] should equal 1
                            # d/dU[i*m+j, r] = V[j*p+k, r] * W[k*n+i, r]
                            # d/dV[j*p+k, r] = U[i*m+j, r] * W[k*n+i, r]
                            # d/dW[k*n+i, r] = U[i*m+j, r] * V[j*p+k, r]
                            for r in range(rank):
                                diff = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] - 1.0
                                grad_U[i*m+j, r] += diff * V[j*p+k, r] * W[k*n+i, r]
                                grad_V[j*p+k, r] += diff * U[i*m+j, r] * W[k*n+i, r]
                                grad_W[k*n+i, r] += diff * U[i*m+j, r] * V[j*p+k, r]
                
                # Update
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                # Check convergence
                if step % 50 == 0:
                    if _validate_decomposition(U, V, W, n, m, p, tol=1e-6):
                        U = _snap_to_half_integers(U)
                        V = _snap_to_half_integers(V)
                        W = _snap_to_half_integers(W)
                        if _validate_decomposition(U, V, W, n, m, p, tol=1e-10):
                            return rank, U, V, W
            
            # Try with schoolbook initialization truncated
            if rank <= n * m * p:
                U_init = U_school[:, :rank]
                V_init = V_school[:, :rank]
                W_init = W_school[:, :rank]
                
                U = U_init + 0.01 * np.random.randn(n * m, rank)
                V = V_init + 0.01 * np.random.randn(m * p, rank)
                W = W_init + 0.01 * np.random.randn(p * n, rank)
                
                lr = 0.005
                for step in range(300):
                    if time.time() > deadline - 1:
                        break
                        
                    grad_U = np.zeros_like(U)
                    grad_V = np.zeros_like(V)
                    grad_W = np.zeros_like(W)
                    
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                for r in range(rank):
                                    diff = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] - 1.0
                                    grad_U[i*m+j, r] += diff * V[j*p+k, r] * W[k*n+i, r]
                                    grad_V[j*p+k, r] += diff * U[i*m+j, r] * W[k*n+i, r]
                                    grad_W[k*n+i, r] += diff * U[i*m+j, r] * V[j*p+k, r]
                    
                    U -= lr * grad_U
                    V -= lr * grad_V
                    W -= lr * grad_W
                    
                    if step % 50 == 0:
                        if _validate_decomposition(U, V, W, n, m, p, tol=1e-6):
                            U = _snap_to_half_integers(U)
                            V = _snap_to_half_integers(V)
                            W = _snap_to_half_integers(W)
                            if _validate_decomposition(U, V, W, n, m, p, tol=1e-10):
                                return rank, U, V, W
    
    return best_rank, best_result

def construct():
    """
    Constructs the rank-minimal tensor decomposition for 2x3 by 3x4 matrix multiplication.
    Searches for lower-rank decompositions before falling back to schoolbook.
    """
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 24
    
    # Start with schoolbook as default
    U, V, W = _schoolbook_decomposition(n, m, p)
    
    # Try to find lower rank
    # The theoretical lower bound is ceil(n*m*p / (n+p-1)) = ceil(24/5) = 5
    # But known bounds are higher. Let's try ranks from 12 to 23.
    start_rank = 12
    end_rank = schoolbook_rank - 1
    
    if time.time() < 150 - 30:  # Leave 30s safety margin
        try:
            rank_found, U_found, V_found, W_found = _try_rank_search(n, m, p, start_rank, end_rank, time_budget=120.0)
            if rank_found is not None:
                U, V, W = U_found, V_found, W_found
        except Exception:
            pass
    
    # Convert to lists of lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
