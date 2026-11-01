"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140  # Safety margin before 150s limit
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition of rank n*m*p
    rank_schoolbook = n * m * p
    U = [[0] * rank_schoolbook for _ in range(n * m)]
    V = [[0] * rank_schoolbook for _ in range(m * p)]
    W = [[0] * rank_schoolbook for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    best_rank = rank_schoolbook
    best_U = U
    best_V = V
    best_W = W
    
    def verify(U_np, V_np, W_np, r):
        """Check if the decomposition is exact: M[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1"""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_np[i*m+j], V_np[j*p+k] * W_np[k*n+i])
                    if abs(val - 1.0) > 1e-8:
                        return False
        return True
    
    def als_step(U_np, V_np, W_np, r, max_iter=200):
        """ALS optimization for tensor decomposition"""
        for it in range(max_iter):
            if time.time() > deadline:
                break
            
            # Update U: for each column c, solve linear system
            # The constraint is: for all (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
            # This is a 3D tensor constraint, so we use gradient-based approach
            
            # Compute gradient for U
            grad_U = np.zeros_like(U_np)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        a = i*m + j
                        b = j*p + k
                        c_idx = k*n + i
                        # d/dU[a,c] of sum_c' U[a,c']*V[b,c']*W[c_idx,c'] 
                        # = V[b,:] * W[c_idx,:]
                        grad_U[a, :] += V_np[b, :] * W_np[c_idx, :]
            
            # The target is 1, current value is sum
            # We want to minimize (sum_c U[a,c]*V[b,c]*W[c,c] - 1)^2
            # Gradient descent step
            lr = 0.01
            U_np -= lr * grad_U
            
            # Update V
            grad_V = np.zeros_like(V_np)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        a = i*m + j
                        b = j*p + k
                        c_idx = k*n + i
                        grad_V[b, :] += U_np[a, :] * W_np[c_idx, :]
            V_np -= lr * grad_V
            
            # Update W
            grad_W = np.zeros_like(W_np)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        a = i*m + j
                        b = j*p + k
                        c_idx = k*n + i
                        grad_W[c_idx, :] += U_np[a, :] * V_np[b, :]
            W_np -= lr * grad_W
            
            # Check convergence
            if it % 20 == 0:
                err = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U_np[i*m+j], V_np[j*p+k] * W_np[k*n+i])
                            err += (val - 1.0)**2
                if err < 1e-12:
                    break
        
        return U_np, V_np, W_np
    
    # Try random search for lower rank
    np.random.seed(42)
    
    for target_rank in range(30, rank_schoolbook):  # Try ranks from 30 to 49
        if time.time() > deadline:
            break
        
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            # Initialize with small random values
            U_r = np.random.uniform(-0.5, 0.5, size=(n*m, target_rank))
            V_r = np.random.uniform(-0.5, 0.5, size=(m*p, target_rank))
            W_r = np.random.uniform(-0.5, 0.5, size=(p*n, target_rank))
            
            # Run ALS optimization
            U_r, V_r, W_r = als_step(U_r, V_r, W_r, target_rank)
            
            # Verify
            if verify(U_r, V_r, W_r, target_rank):
                best_rank = target_rank
                best_U = U_r.tolist()
                best_V = V_r.tolist()
                best_W = W_r.tolist()
                break  # Found a valid decomposition at this rank, try lower
        
        if time.time() > deadline:
            break
    
    # Return the best result found
    if isinstance(best_U, np.ndarray):
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
