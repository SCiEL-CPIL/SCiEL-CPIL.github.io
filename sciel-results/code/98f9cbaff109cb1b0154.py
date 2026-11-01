"""Matrix multiplication 2x5x5 with rank-minimal exact tensor decomposition."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115.0  # 15s safety margin
    
    # The tensor T[i,j,k] for 2x5x5 multiplication
    # We want T = sum_t U[:,t] * V[:,t] * W[:,t]
    # where U is n*m x rank, V is m*p x rank, W is p*n x rank
    
    # Strategy: Use a randomized search to find a lower rank decomposition
    # We'll try to find a rank-r decomposition using iterative methods
    
    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None
    
    # Initialize with schoolbook decomposition
    rank0 = n * m * p
    U0 = np.zeros((n*m, rank0))
    V0 = np.zeros((m*p, rank0))
    W0 = np.zeros((p*n, rank0))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i*m + j, col] = 1.0
                V0[j*p + k, col] = 1.0
                W0[k*n + i, col] = 1.0
                col += 1
    best_U = U0.copy()
    best_V = V0.copy()
    best_W = W0.copy()
    
    # Try to find lower rank decompositions
    # For 2x5x5, we'll try ranks from 25 down to 1
    for target_rank in range(24, 0, -1):
        if time.time() > deadline:
            break
        
        # Multiple random restarts
        max_restarts = 50 if target_rank > 20 else 100
        max_iter = 2000 if target_rank > 20 else 5000
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            
            np.random.seed(42 + restart * 1000)
            
            # Initialize U, V, W with random values
            U = np.random.randn(n*m, target_rank)
            V = np.random.randn(m*p, target_rank)
            W = np.random.randn(p*n, target_rank)
            
            success = False
            for it in range(max_iter):
                if time.time() > deadline:
                    break
                
                # Compute tensor approximation
                # T[i,j,k] = sum_t U[i,j,t] * V[j,k,t] * W[k,i,t]
                # For each t: outer product of reshaped U[:,t], V[:,t], W[:,t]
                
                # Efficient check: For each (i,k) pair, compute the vector
                # h_i,k[j] = sum_t U[i,j,t] * V[j,k,t]
                # Then T[i,j,k] = h_i,k[j] * W[k,i,t] -- this isn't right
                
                # Let's think differently. The tensor T has entries T[i,j,k].
                # We want T[i,j,k] = sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
                
                # For a fixed i,k, define g_{i,k}[j,t] = U[i*m+j, t] * W[k*n+i, t]
                # Then T[i,j,k] = sum_t g_{i,k}[j,t] * V[j*p+k, t]
                # For each i,k, we have n*m*p = 2*5*5 = 50 such constraints
                # Actually for fixed i,k, we have 5 j values, so 5 equations per (i,k) pair
                # Total: n*p = 2*5 = 10 pairs, each giving m=5 equations = 50 total
                
                # The residual for entry (i,j,k) is:
                # r[i,j,k] = 1 - sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
                
                # We can update each matrix in turn. For fixed i,j:
                # The contribution to V[j*p+k, t] should make
                # sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] = 1 for all i,k
                # But this is coupled across i.
                
                # Simpler: use gradient descent on the full objective
                # F = 0.5 * sum_{i,j,k} (sum_t U[i*m+j,t]*V[j*p+k,t]*W[k*n+i,t] - 1)^2
                
                # Compute gradient with respect to U
                # dF/dU[a,t] where a = i*m+j
                # = sum_k (pred[i,j,k] - 1) * V[j*p+k, t] * W[k*n+i, t]
                # where pred[i,j,k] = sum_t' U[i*m+j,t']*V[j*p+k,t']*W[k*n+i,t']
                
                # This is getting complex. Let's use a direct approach:
                # Build the full tensor and compute Frobenius norm difference.
                
                # Build current tensor approximation
                # T_pred[i,j,k] = sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
                T_pred = np.zeros((n, m, p))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            T_pred[i, j, k] = np.dot(
                                U[i*m + j, :] * W[k*n + i, :],
                                V[j*p + k, :]
                            )
                
                # Compute residual
                residual = T_pred - 1.0
                err = np.linalg.norm(residual)
                
                if err < 1e-10:
                    success = True
                    best_rank = target_rank
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
                    break
                
                # Simple gradient step
                # We'll do coordinate descent: for each (a, t), U, etc.
                # Actually let's just do a small step via finite differences or analytic grad
                
                # Analytic gradient for U
                grad_U = np.zeros_like(U)
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            coeff = residual[i, j, k] * V[j*p + k, :] * W[k*n + i, :]
                            grad_U[i*m + j, :] += coeff
                
                # Analytic gradient for V  
                grad_V = np.zeros_like(V)
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            coeff = residual[i, j, k] * U[i*m + j, :] * W[k*n + i, :]
                            grad_V[j*p + k, :] += coeff
                
                # Analytic gradient for W
                grad_W = np.zeros_like(W)
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            coeff = residual[i, j, k] * U[i*m + j, :] * V[j*p + k, :]
                            grad_W[k*n + i, :] += coeff
                
                # Take a small gradient step
                lr = 0.01
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
            if success:
                break
        
        if best_rank == target_rank:
            break
    
    # Convert to lists of numbers
    U_list = [[int(x) if abs(x - round(x)) < 1e-9 else float(x) for x in row] for row in best_U]
    V_list = [[int(x) if abs(x - round(x)) < 1e-9 else float(x) for x in row] for row in best_V]
    W_list = [[int(x) if abs(x - round(x)) < 1e-9 else float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
