"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import nnls

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    """Return U, V, W as lists for the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W

def _try_find_decomposition(n, m, p, target_rank, max_iter=1000, tol=1e-6):
    """
    Try to find a rank decomposition using iterative optimization.
    Returns (U, V, W) as lists if found, else None.
    """
    nm, mp, pn = n * m, m * p, p * n
    
    # Initialize with schoolbook and try to reduce rank
    # Use random initialization for the rank-r decomposition
    np.random.seed(42)
    
    for attempt in range(10):
        # Random initialization
        U = np.random.randn(nm, target_rank) * 0.5
        V = np.random.randn(mp, target_rank) * 0.5
        W = np.random.randn(pn, target_rank) * 0.5
        
        # Iterative refinement: fix two matrices, solve for the third
        for iteration in range(max_iter):
            # Fix U and V, solve for W
            # For each (k, i): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            # This is linear in W if U and V are fixed
            
            # Build the linear system for W
            # Unknowns: W[k*n+i, r] for all k, i, r
            # Equations: for each (i, j, k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            
            # Actually, let's fix W and solve for U and V jointly
            # Or use a different approach: gradient descent on the loss
            
            # Compute current tensor
            current = np.zeros((n, m, p))
            for r in range(target_rank):
                u_r = U[:, r].reshape(n, m)
                v_r = V[:, r].reshape(m, p)
                w_r = W[:, r].reshape(p, n)
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            current[i, j, k] += u_r[i, j] * v_r[j, k] * w_r[k, i]
            
            # Target tensor: all ones
            target = np.ones((n, m, p))
            loss = np.sum((current - target) ** 2)
            
            if loss < tol:
                # Check if entries are close to integer or half-integer
                all_entries = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
                # Round to nearest half-integer
                rounded = np.round(all_entries * 2) / 2
                if np.max(np.abs(all_entries - rounded)) < 1e-4:
                    # Convert to lists
                    U_list = [[rounded[i*target_rank + r] for r in range(target_rank)] for i in range(nm)]
                    V_list = [[rounded[(nm + i)*target_rank + r] for r in range(target_rank)] for i in range(mp)]
                    W_list = [[rounded[(nm + mp + i)*target_rank + r] for r in range(target_rank)] for i in range(pn)]
                    return U_list, V_list, W_list
            
            # Gradient descent step
            # dU[i*m+j, r] = -2 * sum_k (current[i,j,k] - 1) * V[j*p+k, r] * W[k*n+i, r]
            # dV[j*p+k, r] = -2 * sum_i (current[i,j,k] - 1) * U[i*m+j, r] * W[k*n+i, r]
            # dW[k*n+i, r] = -2 * sum_j (current[i,j,k] - 1) * U[i*m+j, r] * V[j*p+k, r]
            
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        err = current[i, j, k] - 1
                        for r in range(target_rank):
                            dU[i*m+j, r] -= 2 * err * V[j*p+k, r] * W[k*n+i, r]
                            dV[j*p+k, r] -= 2 * err * U[i*m+j, r] * W[k*n+i, r]
                            dW[k*n+i, r] -= 2 * err * U[i*m+j, r] * V[j*p+k, r]
            
            # Update with small learning rate
            lr = 0.01
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
        
        # Check final loss
        current = np.zeros((n, m, p))
        for r in range(target_rank):
            u_r = U[:, r].reshape(n, m)
            v_r = V[:, r].reshape(m, p)
            w_r = W[:, r].reshape(p, n)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        current[i, j, k] += u_r[i, j] * v_r[j, k] * w_r[k, i]
        
        target = np.ones((n, m, p))
        loss = np.sum((current - target) ** 2)
        
        if loss < 1e-4:
            all_entries = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            rounded = np.round(all_entries * 2) / 2
            if np.max(np.abs(all_entries - rounded)) < 1e-3:
                U_list = [[rounded[i*target_rank + r] for r in range(target_rank)] for i in range(nm)]
                V_list = [[rounded[(nm + i)*target_rank + r] for r in range(target_rank)] for i in range(mp)]
                W_list = [[rounded[(nm + mp + i)*target_rank + r] for r in range(target_rank)] for i in range(pn)]
                return U_list, V_list, W_list
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # safety margin before 120s limit
    
    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try to find better decompositions from rank 59 down to 30
    for r in range(best_rank - 1, 29, -1):
        if time.time() > deadline - 15:
            break
        result = _try_find_decomposition(n, m, p, r, max_iter=200, tol=1e-5)
        if result is not None:
            U, V, W = result
            best_rank = r
            break
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
