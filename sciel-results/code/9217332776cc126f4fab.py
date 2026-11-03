"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [4, 4, 7]

def _schoolbook():
    n, m, p = DIMENSIONS
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

def _verify(U, V, W, n, m, p, rank, tol=1e-9):
    """Check if U, V, W form a valid tensor decomposition."""
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    
    # Build the multiplication tensor T[i,j,k] = 1 if the term a[i,j]*b[j,k] appears
    # T is indexed as (i, j, k) with shape (n, m, p)
    # The decomposition requires: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Reshape for batch computation
    # U: (n*m, rank) -> U[i*m+j, c]
    # V: (m*p, rank) -> V[j*p+k, c]  
    # W: (p*n, rank) -> W[k*n+i, c]
    
    # For each (i,j,k), compute sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # This equals element-wise product summed over c
    
    # Create index arrays
    i_idx = np.repeat(np.arange(n), m * p)
    j_idx = np.tile(np.repeat(np.arange(m), p), n)
    k_idx = np.tile(np.arange(p), n * m)
    
    u_idx = i_idx * m + j_idx  # (n*m*p,)
    v_idx = j_idx * p + k_idx  # (n*m*p,)
    w_idx = k_idx * n + i_idx  # (p*n) indexing
    
    u_vals = Ua[u_idx, :]  # (n*m*p, rank)
    v_vals = Va[v_idx, :]  # (n*m*p, rank)
    w_vals = Wa[w_idx, :]  # (n*m*p, rank)
    
    # Element-wise product summed over rank
    prod = u_vals * v_vals * w_vals  # (n*m*p, rank)
    result = prod.sum(axis=1)  # (n*m*p,)
    
    # The tensor T[i,j,k] should be 1 for all valid (i,j,k)
    expected = np.ones(n * m * p)
    
    return np.allclose(result, expected, atol=tol)

def _refine(U, V, W, n, m, p, rank, max_iter=100, tol=1e-9):
    """Iteratively refine the decomposition using gradient descent on the error."""
    Ua = np.array(U, dtype=float).copy()
    Va = np.array(V, dtype=float).copy()
    Wa = np.array(W, dtype=float).copy()
    
    # Precompute indices
    i_idx = np.repeat(np.arange(n), m * p)
    j_idx = np.tile(np.repeat(np.arange(m), p), n)
    k_idx = np.tile(np.arange(p), n * m)
    
    u_idx = i_idx * m + j_idx
    v_idx = j_idx * p + k_idx
    w_idx = k_idx * n + i_idx
    
    for it in range(max_iter):
        # Compute current values
        u_vals = Ua[u_idx, :]  # (N, rank)
        v_vals = Va[v_idx, :]
        w_vals = Wa[w_idx, :]
        
        prod = u_vals * v_vals * w_vals
        result = prod.sum(axis=1)
        
        error = result - 1.0  # Should be zero
        err_norm = np.linalg.norm(error)
        
        if err_norm < tol:
            break
        
        # Gradient descent: minimize sum_c (U*V*W - 1)^2
        # d/dU[i*m+j,c] = sum over (j,k) where i*m+j is fixed: 2*(prod_sum - 1) * V[j*p+k,c] * W[k*n+i,c]
        # This is complex; use simple coordinate updates or just check convergence
        
        # Simple approach: scale factors to reduce error
        # If error is positive, scale down; if negative, scale up
        # This is heuristic but can help
        
        # Compute mean error
        mean_err = np.mean(error)
        
        # Apply small updates
        lr = 0.01 / (1 + it * 0.1)
        
        # For each rank component c, compute gradient
        # dL/dc for U[a,c] where a = i*m+j:
        #   sum over (j,k) with fixed i: 2 * error[i,j,k] * V[j*p+k,c] * W[k*n+i,c]
        
        # Build gradient for U
        grad_U = np.zeros_like(Ua)
        grad_V = np.zeros_like(Va)
        grad_W = np.zeros_like(Wa)
        
        # Vectorized gradient computation
        # For U: for each (i,j,k) and c, contribution to U[i*m+j, c] is 2*error[i,j,k]*V[j*p+k,c]*W[k*n+i,c]
        # We need to sum over all k for fixed (i,j)
        
        # Reshape for easier summation
        # error: (n*m*p,) -> (n, m, p)
        err_3d = error.reshape(n, m, p)
        
        # u_vals: (n*m*p, rank) -> (n, m, p, rank)
        u_4d = u_vals.reshape(n, m, p, rank)
        v_4d = v_vals.reshape(n, m, p, rank)
        w_4d = w_vals.reshape(n, m, p, rank)
        
        # grad_U[i*m+j, c] = sum_k 2 * err_3d[i,j,k] * v_4d[i,j,k,c] * w_4d[i,j,k,c]
        grad_U_4d = 2 * err_4d if False else 2 * np.expand_dims(err_3d, axis=-1) * v_4d * w_4d
        grad_U = np.sum(grad_U_4d, axis=2).reshape(n*m, rank)
        
        # grad_V[j*p+k, c] = sum_i 2 * err_3d[i,j,k] * u_4d[i,j,k,c] * w_4d[i,j,k,c]
        grad_V_4d = 2 * np.expand_dims(err_3d, axis=-1) * u_4d * w_4d
        # Need to sum over i, but V is indexed by (j,k), so we need to gather
        # grad_V[j*p+k, c] = sum_i grad_V_4d[i,j,k,c]
        grad_V_flat = np.zeros((m*p, rank))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    grad_V_flat[j*p+k, :] += grad_V_4d[i,j,k,:]
        grad_V = grad_V_flat
        
        # grad_W[k*n+i, c] = sum_j 2 * err_3d[i,j,k] * u_4d[i,j,k,c] * v_4d[i,j,k,c]
        grad_W_4d = 2 * np.expand_dims(err_3d, axis=-1) * u_4d * v_4d
        grad_W_flat = np.zeros((p*n, rank))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    grad_W_flat[k*n+i, :] += grad_W_4d[i,j,k,:]
        grad_W = grad_W_flat
        
        # Update
        Ua -= lr * grad_U
        Va -= lr * grad_V
        Wa -= lr * grad_W
    
    # Convert back to lists
    return Ua.tolist(), Va.tolist(), Wa.tolist()

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Start with schoolbook
    best = _schoolbook()
    best_rank = n * m * p
    
    # Try to find lower rank decompositions
    for rank in range(n*m, best_rank + 1):
        if time.time() > deadline:
            break
        
        # Try multiple random initializations
        max_attempts = 50
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            random.seed(rank * 1000 + attempt)
            np.random.seed(rank * 1000 + attempt)
            
            # Random initialization with small values
            U = np.random.randn(n*m, rank) * 0.1
            V = np.random.randn(m*p, rank) * 0.1
            W = np.random.randn(p*n, rank) * 0.1
            
            # Add some structure: initialize with schoolbook-like patterns
            # For each (i,j,k), set one rank component
            if rank >= n*m*p:
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            c = i*m*p + j*p + k
                            U[i*m+j, c] = 1.0
                            V[j*p+k, c] = 1.0
                            W[k*n+i, c] = 1.0
            
            # Refine
            U_ref, V_ref, W_ref = _refine(U.tolist(), V.tolist(), W.tolist(), n, m, p, rank, max_iter=50)
            
            # Verify
            if _verify(U_ref, V_ref, W_ref, n, m, p, rank, tol=1e-6):
                best = {"U": U_ref, "V": V_ref, "W": W_ref}
                best_rank = rank
                break  # Found valid decomposition at this rank, try lower
        
        # If we found a valid one at this rank, continue to try lower ranks
        # (the loop naturally goes upward, so we need to check if we should stop)
    
    return best
# EVOLVE-BLOCK-END
