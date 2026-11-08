"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    # Schoolbook decomposition
    rank_school = n * m * p
    U_school = np.zeros((n*m, rank_school))
    V_school = np.zeros((m*p, rank_school))
    W_school = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    
    best_rank = rank_school
    best_U = U_school.copy()
    best_V = V_school.copy()
    best_W = W_school.copy()
    
    def check_decomposition(U, V, W, tol=1e-6):
        """Check if the decomposition satisfies the tensor condition."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(U.shape[1]))
                    if abs(val - 1) > tol:
                        return False
        return True
    
    def tensor_residual(U, V, W):
        """Compute sum of squared residuals for all (i,j,k)."""
        residual = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(U.shape[1]))
                    residual += (val - 1) ** 2
        return residual
    
    def gradient_step(U, V, W, lr=0.01, max_iter=500):
        """Perform gradient descent to minimize tensor residual."""
        r = U.shape[1]
        for it in range(max_iter):
            if time.time() > deadline - 10:
                break
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(r))
                        err = val - 1
                        for c in range(r):
                            dU[i*m+j, c] += 2 * err * V[j*p+k, c] * W[k*n+i, c]
                            dV[j*p+k, c] += 2 * err * U[i*m+j, c] * W[k*n+i, c]
                            dW[k*n+i, c] += 2 * err * U[i*m+j, c] * V[j*p+k, c]
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            if tensor_residual(U, V, W) < 1e-10:
                break
        return U, V, W
    
    # Try ranks from schoolbook-1 down
    for target_rank in range(rank_school - 1, max(15, rank_school - 20), -1):
        if time.time() > deadline - 40:
            break
        
        found = False
        for attempt in range(30):
            if time.time() > deadline - 40:
                break
            
            # Initialize by removing random columns from schoolbook and adding noise
            U = np.zeros((n*m, target_rank))
            V = np.zeros((m*p, target_rank))
            W = np.zeros((p*n, target_rank))
            
            # Start with a subset of schoolbook columns
            cols_to_use = random.sample(range(rank_school), target_rank)
            for c, sc in enumerate(cols_to_use):
                U[:, c] = U_school[:, sc]
                V[:, c] = V_school[:, sc]
                W[:, c] = W_school[:, sc]
            
            # Add small random perturbation
            noise_scale = 0.1
            U += noise_scale * np.random.randn(n*m, target_rank)
            V += noise_scale * np.random.randn(m*p, target_rank)
            W += noise_scale * np.random.randn(p*n, target_rank)
            
            # Run gradient descent
            U, V, W = gradient_step(U, V, W, lr=0.005, max_iter=300)
            
            if check_decomposition(U, V, W):
                best_rank = target_rank
                best_U = U
                best_V = V
                best_W = W
                found = True
                break
        
        if found:
            break
    
    # Round to nearest half-integer for cleaner output
    def round_half(x):
        return np.round(x * 2) / 2
    
    best_U = round_half(best_U)
    best_V = round_half(best_V)
    best_W = round_half(best_W)
    
    # Final verification
    if not check_decomposition(best_U, best_V, best_W, tol=1e-4):
        # Fall back to schoolbook
        best_U = U_school
        best_V = V_school
        best_W = W_school
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
