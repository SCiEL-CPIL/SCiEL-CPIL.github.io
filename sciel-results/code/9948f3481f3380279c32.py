import numpy as np
import time
import random
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # 10 seconds margin
    
    # Target rank to try
    target_rank = 12
    
    # Initialize random matrices
    U = np.random.randn(n * m, target_rank)
    V = np.random.randn(m * p, target_rank)
    W = np.random.randn(p * n, target_rank)
    
    def loss(params):
        if time.time() > deadline:
            return float('inf')
        U_flat = params[:n*m*target_rank].reshape(n*m, target_rank)
        V_flat = params[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
        W_flat = params[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
        
        # Compute tensor T from U, V, W
        # T[i, j, k] = sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
        # We need T to match the multiplication tensor.
        # Multiplication tensor M[i, j, k] is 1 if i,j,k form a valid term?
        # Actually, the tensor M is defined such that M[(i,j),(j,k),(k,i)] = 1.
        # In the flattened indices:
        # Input A index: i*m + j
        # Input B index: j*p + k
        # Output C index: k*n + i
        
        # We want sum_t U[i*m+j, t] V[j*p+k, t] W[k*n+i, t] = 1 for all i,j,k
        # Wait, the output is C[i,k] = sum_j A[i,j] B[j,k].
        # So the term is A[i,j] * B[j,k].
        # The tensor T has non-zero entries at (i*m+j, j*p+k, k*n+i).
        # We want T_decomp[i*m+j, j*p+k, k*n+i] = 1.
        
        error = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for t in range(target_rank):
                        val += U_flat[i*m+j, t] * V_flat[j*p+k, t] * W_flat[k*n+i, t]
                    # The target value is 1
                    error += (val - 1)**2
        return error

    params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    
    # Try to optimize
    best_params = params
    best_loss = loss(params)
    
    # Simple iterative refinement
    for _ in range(50):
        if time.time() > deadline:
            break
        result = minimize(loss, params, method='BFGS', options={'maxiter': 10})
        if result.fun < best_loss:
            best_loss = result.fun
            best_params = result.x
    
    # Check if solution is good enough
    if best_loss < 1e-6:
        U_sol = best_params[:n*m*target_rank].reshape(n*m, target_rank)
        V_sol = best_params[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
        W_sol = best_params[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
        
        # Convert to list of lists of floats/ints
        # The task requires integer/half-integer entries.
        # We can try to round to nearest 0.5
        U_rounded = np.round(U_sol * 2) / 2
        V_rounded = np.round(V_sol * 2) / 2
        W_rounded = np.round(W_sol * 2) / 2
        
        # Verify rounded solution
        error = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for t in range(target_rank):
                        val += U_rounded[i*m+j, t] * V_rounded[j*p+k, t] * W_rounded[k*n+i, t]
                    error += (val - 1)**2
        
        if error < 1e-6:
            return {
                "U": U_rounded.tolist(),
                "V": V_rounded.tolist(),
                "W": W_rounded.tolist()
            }
            
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
