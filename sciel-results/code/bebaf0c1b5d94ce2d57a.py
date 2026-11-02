"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin before 150s kill
    
    # Build the target tensor: M[i,j,k] = 1 for all valid (i,j,k)
    # Decomposition: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # Try to find a low-rank decomposition
    # Start with schoolbook rank and try to reduce
    
    def schoolbook():
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
        return U, V, W, rank
    
    def verify(U, V, W, tol=1e-8):
        """Check if the decomposition is valid."""
        n, m, p = DIMENSIONS
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1) > tol:
                        return False
        return True
    
    def try_rank(r, max_iters=2000, seed=42):
        """Try to find a rank-r decomposition using gradient descent."""
        if time.time() > deadline - 10:
            return None
        
        rng = np.random.RandomState(seed)
        
        # Initialize with small random values
        U = rng.randn(n * m, r) * 0.1
        V = rng.randn(m * p, r) * 0.1
        W = rng.randn(p * n, r) * 0.1
        
        # Compute residuals and gradients
        def residual(U, V, W):
            res = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        res[i,j,k] = np.sum(U[i*m+j,:] * V[j*p+k,:] * W[k*n+i,:]) - 1
            return res
        
        # Use a simple gradient descent / Newton-like approach
        # This is a system of polynomial equations, so we use a least-squares approach
        
        # Flatten the problem
        def flatten_residual(U, V, W):
            res = np.zeros(n*m*p)
            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        res[idx] = np.sum(U[i*m+j,:] * V[j*p+k,:] * W[k*n+i,:]) - 1
                        idx += 1
            return res
        
        # Try iterative refinement
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        best_err = np.inf
        
        lr = 0.01
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
            
            res = flatten_residual(U, V, W)
            err = np.sum(res**2)
            
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if err < 1e-12:
                break
            
            # Compute gradients
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        r_val = res[idx]
                        dU[i*m+j, :] += 2 * r_val * V[j*p+k,:] * W[k*n+i,:]
                        dV[j*p+k, :] += 2 * r_val * U[i*m+j,:] * W[k*n+i,:]
                        dW[k*n+i, :] += 2 * r_val * U[i*m+j,:] * V[j*p+k,:]
                        idx += 1
            
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
            # Adaptive learning rate
            new_res = flatten_residual(U, V, W)
            new_err = np.sum(new_res**2)
            if new_err > err:
                U += lr * dU
                V += lr * dV
                W += lr * dW
                lr *= 0.5
            elif new_err > err * 0.99:
                lr *= 0.8
            else:
                lr = min(lr * 1.1, 0.1)
        
        if best_err < 1e-8:
            return best_U, best_V, best_W, r
        return None
    
    # First, get the schoolbook solution as a fallback
    U_sb, V_sb, W_sb, rank_sb = schoolbook()
    
    # Try to find a lower rank
    # Theoretical lower bound: at least max(n*m, m*p, n*p) = max(6, 21, 14) = 21
    # But likely higher. Let's try from 21 upward.
    
    best_solution = (U_sb, V_sb, W_sb, rank_sb)
    
    # Try various ranks
    for r in range(21, rank_sb):
        if time.time() > deadline - 20:
            break
        
        # Try multiple seeds
        for seed in [42, 123, 456, 789, 1000]:
            if time.time() > deadline - 20:
                break
            
            result = try_rank(r, max_iters=500, seed=seed)
            if result is not None:
                U, V, W, rank = result
                if rank < best_solution[3]:
                    best_solution = (U, V, W, rank)
                    break  # Found a valid solution for this rank
    
    # Return the best solution
    U, V, W, rank = best_solution
    
    # Convert to list format
    U_list = [[float(U[i,j]) for j in range(rank)] for i in range(n*m)]
    V_list = [[float(V[i,j]) for j in range(rank)] for i in range(m*p)]
    W_list = [[float(W[i,j]) for j in range(rank)] for i in range(p*n)]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
