"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    
    # We want a rank-minimal exact tensor decomposition for matrix multiplication:
    # C = A * B, where A is (n x m), B is (m x p), C is (n x p).
    # The multiplication tensor T has T[i,j,k] = 1 for all valid (i,j,k).
    # We seek U (n*m, r), V (m*p, r), W (p*n, r) such that:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k.
    #
    # The interface specifies W[k*n+i, r], which means W is indexed by (k, i)
    # where k is the output column and i is the output row.
    # So W[k*n + i, c] is the coefficient for C[i,k] in the c-th term.
    #
    # Let's verify the schoolbook decomposition first:
    # For each (i,j,k), we have a term with U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1.
    # This gives rank = n*m*p = 3*3*4 = 36.
    #
    # Can we do better? For 3x3x3, the optimal rank is 23 (Landsberg).
    # For 3x3x4, the optimal rank is not known to be significantly lower than 36 in simple constructions.
    # However, we can try to use block structure or known decompositions.
    #
    # A simple improvement: if we can find a decomposition of rank < 36, we use it.
    # Otherwise, we return the schoolbook.
    #
    # Let's try to construct a rank 23 or lower decomposition using a heuristic search.
    # Given the 120s limit, a full search is risky.
    #
    # Alternative: Use the known rank 23 decomposition for 3x3x3 and extend it.
    # But extending to 3x3x4 is non-trivial.
    #
    # Let's try a random search with local optimization for a few ranks below 36.
    # If we find a valid decomposition, we use it. Otherwise, we fall back to schoolbook.
    
    import random
    import time
    
    start_time = time.time()
    time_limit = 100  # Leave some margin for the 120s limit
    
    # Precompute indices
    indices = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    num_eqs = len(indices)
    
    def check_decomposition(U, V, W, tol=1e-6):
        """Check if the decomposition is exact within tolerance."""
        max_err = 0.0
        for (i, j, k) in indices:
            u_idx = i * m + j
            v_idx = j * p + k
            w_idx = k * n + i
            val = np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])
            err = abs(val - 1.0)
            if err > max_err:
                max_err = err
        return max_err < tol
    
    def loss_and_grad(U, V, W):
        """Compute loss and gradients for the decomposition."""
        loss = 0.0
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        for (i, j, k) in indices:
            u_idx = i * m + j
            v_idx = j * p + k
            w_idx = k * n + i
            
            # val = sum_c U[u_idx, c] * V[v_idx, c] * W[w_idx, c]
            prod = U[u_idx, :] * V[v_idx, :] * W[w_idx, :]
            val = np.sum(prod)
            diff = val - 1.0
            loss += diff * diff
            
            # Gradient w.r.t U[u_idx, c] = 2 * diff * V[v_idx, c] * W[w_idx, c]
            dU[u_idx, :] += 2.0 * diff * (V[v_idx, :] * W[w_idx, :])
            # Gradient w.r.t V[v_idx, c] = 2 * diff * U[u_idx, c] * W[w_idx, c]
            dV[v_idx, :] += 2.0 * diff * (U[u_idx, :] * W[w_idx, :])
            # Gradient w.r.t W[w_idx, c] = 2 * diff * U[u_idx, c] * V[v_idx, c]
            dW[w_idx, :] += 2.0 * diff * (U[u_idx, :] * V[v_idx, :])
        
        return loss, dU, dV, dW
    
    # Try to find a decomposition of rank < 36
    # Start with rank 35 and go down to 23
    # Use a simple gradient descent with random restarts
    
    best_U, best_V, best_W, best_rank = None, None, None, None
    
    for rank in range(35, 22, -1):
        if time.time() - start_time > time_limit:
            break
        
        for seed in range(5):
            if time.time() - start_time > time_limit:
                break
            
            np.random.seed(seed)
            U = np.random.randn(n * m, rank) * 0.1
            V = np.random.randn(m * p, rank) * 0.1
            W = np.random.randn(p * n, rank) * 0.1
            
            lr = 0.01
            prev_loss = float('inf')
            converged = False
            
            for it in range(500):
                if time.time() - start_time > time_limit:
                    break
                
                loss, dU, dV, dW = loss_and_grad(U, V, W)
                
                if loss < 1e-8:
                    converged = True
                    break
                
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
                
                if loss > prev_loss * 1.01:
                    lr *= 0.5
                prev_loss = loss
            
            if converged:
                best_U, best_V, best_W, best_rank = U, V, W, rank
                break
        
        if best_rank is not None:
            break
    
    if best_rank is not None:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Fall back to schoolbook decomposition
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
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
