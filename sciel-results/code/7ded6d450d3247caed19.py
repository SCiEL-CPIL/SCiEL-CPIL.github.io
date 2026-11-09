"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check if U, V, W form a valid rank-r decomposition."""
    r = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # For each (i,j,k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal 1
    # And for any (i,j,k) where we don't want a 1, it should be 0
    # Actually, the tensor T[i,j,k] = delta_{i,i'}delta_{j,j'}delta_{k,k'} for matmul
    # T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # This should equal 1 if this is a valid matmul tensor entry
    
    # The matmul tensor: C[i,k] = sum_j A[i,j]*B[j,k]
    # So T[i,j,k] = 1 for all i,j,k (it's the identity in the "multiplication" sense)
    # Wait, let me reconsider.
    
    # Matrix multiplication: C = A * B where A is n x m, B is m x p, C is n x p
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T has indices (i, j, k) and T[i,j,k] = 1 for all valid i,j,k
    # because each term A[i,j]*B[j,k] contributes to C[i,k]
    
    # So we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(r):
                    val += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(val - 1.0) > tol:
                    return False
    return True

def _try_rank(r, n, m, p, max_iters=200, seed=42, deadline=None):
    """Try to find a rank-r decomposition using random search + local optimization."""
    np.random.seed(seed)
    random.seed(seed)
    
    best_error = float('inf')
    best_U = best_V = best_W = None
    
    for attempt in range(10):
        if deadline and time.time() > deadline:
            break
            
        # Random initialization with small values
        U = np.random.randn(n * m, r) * 0.5
        V = np.random.randn(m * p, r) * 0.5
        W = np.random.randn(p * n, r) * 0.5
        
        # Local optimization: gradient descent on the error
        lr = 0.01
        for it in range(max_iters):
            if deadline and time.time() > deadline:
                break
                
            # Compute gradient
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            error = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for rr in range(r):
                            val += U[i*m+j, rr] * V[j*p+k, rr] * W[k*n+i, rr]
                        err = val - 1.0
                        error += err * err
                        for rr in range(r):
                            grad_U[i*m+j, rr] += 2 * err * V[j*p+k, rr] * W[k*n+i, rr]
                            grad_V[j*p+k, rr] += 2 * err * U[i*m+j, rr] * W[k*n+i, rr]
                            grad_W[k*n+i, rr] += 2 * err * U[i*m+j, rr] * V[j*p+k, rr]
            
            if error < best_error:
                best_error = error
                if error < 1e-9:
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
                    return True, U.tolist(), V.tolist(), W.tolist()
                
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            # Adaptive learning rate
            if error > best_error * 1.5:
                lr *= 0.5
        
        if best_error < 1e-9:
            return True, best_U.tolist(), best_V.tolist(), best_W.tolist()
    
    return best_error < 1e-9, None, None, None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Leave 10s margin
    
    # Start with schoolbook as fallback
    schoolbook_rank = n * m * p
    U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    # Try to find lower rank decompositions
    # Known lower bounds: max(n*m, m*p, p*n) = max(10, 25, 10) = 25
    # Try ranks from 25 upward
    for r in range(25, schoolbook_rank + 1):
        if time.time() > deadline - 30:
            break
            
        found, U, V, W = _try_rank(r, n, m, p, max_iters=50, seed=42+r, deadline=deadline - 10)
        if found:
            # Verify with exact arithmetic
            if _verify(U, V, W, n, m, p, tol=1e-9):
                return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    return {"U": U_sb, "V": V_sb, "W": W_sb}
# EVOLVE-BLOCK-END
