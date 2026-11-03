import numpy as np
import time
import random

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly decomposes the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(val - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the standard schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _sgd_search(n, m, p, target_rank, max_iters=2000, lr=0.01, lambda_l1=0.01, seed=42):
    """
    Attempt to find a decomposition of target_rank using SGD with L1 regularization.
    Returns (U, V, W) if a valid integer/half-integer solution is found, else None.
    """
    np.random.seed(seed)
    
    # Target tensor T[i*m+j, j*p+k, k*n+i] = 1
    # We minimize || sum_r U_r V_r W_r - T ||^2 + lambda * (||U||_1 + ||V||_1 + ||W||_1)
    
    # Initialize with random values, scaled to be small
    U = np.random.randn(n * m, target_rank) * 0.1
    V = np.random.randn(m * p, target_rank) * 0.1
    W = np.random.randn(p * n, target_rank) * 0.1
    
    # Precompute indices for faster tensor computation
    # T is sparse, only 1s at (i,j,k)
    # We can compute the error by iterating over non-zero entries of T
    
    indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                indices.append((i * m + j, j * p + k, k * n + i))
                
    def compute_error(U, V, W):
        err = 0.0
        # Sum of squared errors
        # We only need to check the positions where T=1
        # And we need to ensure other positions are 0?
        # The tensor T is 1 at (i,j,k) and 0 elsewhere?
        # No, the tensor for matrix multiplication is such that
        # C[i,k] = sum_j A[i,j] B[j,k].
        # The tensor T has T[i,j,k] = 1 for all i,j,k.
        # Wait, if T[i,j,k]=1 for all i,j,k, then the sum over r must be 1 for all i,j,k.
        # And for other indices? The tensor is defined on the domain of the bilinear map.
        # The domain is A (n x m) and B (m x p).
        # The tensor is in A* ⊗ B* ⊗ C*.
        # The coefficients are 1 for the basis elements.
        # So yes, T[i,j,k] = 1 for all valid i,j,k.
        
        for idx in indices:
            u_idx, v_idx, w_idx = idx
            val = 0.0
            for r in range(target_rank):
                val += U[u_idx, r] * V[v_idx, r] * W[w_idx, r]
            err += (val - 1.0) ** 2
        return err

    # SGD Loop
    for it in range(max_iters):
        if it % 100 == 0:
            # Check time
            if time.time() > time.time() + 100: # Dummy check, handled outside
                break
            
        # Compute gradients numerically or analytically?
        # Analytical is better.
        # Error E = sum_{i,j,k} (sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] - 1)^2
        # dE/dU[i*m+j, r] = 2 * (sum_k (sum_r ... - 1) * V[j*p+k, r] * W[k*n+i, r])
        
        # This is expensive to compute fully.
        # Let's use a simple finite difference or just random updates?
        # No, let's try to be smart.
        
        # For each (i,j,k), let Res[i,j,k] = sum_r ... - 1
        # Grad_U[i*m+j, r] = 2 * sum_k Res[i,j,k] * V[j*p+k, r] * W[k*n+i, r]
        
        Res = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(target_rank):
                        val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    Res[i, j, k] = val - 1.0
        
        Grad_U = np.zeros_like(U)
        Grad_V = np.zeros_like(V)
        Grad_W = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    res = Res[i, j, k]
                    if abs(res) < 1e-6: continue
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    for r in range(target_rank):
                        # dE/dU
                        Grad_U[u_idx, r] += 2 * res * V[v_idx, r] * W[w_idx, r]
                        # dE/dV
                        Grad_V[v_idx, r] += 2 * res * U[u_idx, r] * W[w_idx, r]
                        # dE/dW
                        Grad_W[w_idx, r] += 2 * res * U[u_idx, r] * V[v_idx, r]
        
        # L1 gradient (sign)
        Grad_U += lambda_l1 * np.sign(U)
        Grad_V += lambda_l1 * np.sign(V)
        Grad_W += lambda_l1 * np.sign(W)
        
        # Update
        U -= lr * Grad_U
        V -= lr * Grad_V
        W -= lr * Grad_W
        
        # Periodic rounding and check
        if it % 500 == 0:
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            
            if _verify(U_rounded, V_rounded, W_rounded, n, m, p):
                # Check rank (number of non-zero columns)
                # We want to ensure we are using the full target_rank or less
                # If we found a solution, we can return it
                return U_rounded, V_rounded, W_rounded
                
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = n * m * p
    
    # Try SGD search for lower ranks
    # We try to find a solution with rank < best_rank
    # Start from best_rank - 1 down to some limit
    # Since 100 is the schoolbook, try 99, 98, 97
    
    for target_rank in [best_rank - 1, best_rank - 2, best_rank - 3]:
        if time.time() > deadline - 20:
            break
            
        result = _sgd_search(n, m, p, target_rank, max_iters=1000, lr=0.05, lambda_l1=0.05, seed=target_rank)
        if result is not None:
            U_found, V_found, W_found = result
            # Verify again to be safe
            if _verify(U_found, V_found, W_found, n, m, p):
                best_U, best_V, best_W = U_found, V_found, W_found
                best_rank = target_rank
                break # Found a better one, stop searching lower ranks? 
                # Actually, we might find an even lower one, but it's rare.
                # We can continue to search for lower ranks if time permits.
                
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
