import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 4, 7]

def construct():
    start_time = time.time()
    deadline = 140.0  # 10s margin
    
    n, m, p = DIMENSIONS
    # Target: for all i,j,k, sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # Schoolbook solution
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_rank = rank_school
    best_U, best_V, best_W = U_school.copy(), V_school.copy(), W_school.copy()
    
    # Try to find lower rank using randomized search with iterative refinement
    # We'll try various ranks and use gradient-based optimization
    
    def check_decomposition(U, V, W, tol=1e-6):
        """Check if the decomposition exactly reconstructs the multiplication tensor."""
        n, m, p = DIMENSIONS
        rank = U.shape[1]
        errors = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    errors.append(abs(val - 1.0))
        return max(errors) if errors else 0.0
    
    def try_rank(rank, max_iter=1000, lr=0.01, seed=42):
        """Try to find a decomposition of given rank using randomized initialization and refinement."""
        if rank >= best_rank:
            return None
        
        np.random.seed(seed)
        
        # Initialize with small random values
        U = np.random.randn(n * m, rank) * 0.1
        V = np.random.randn(m * p, rank) * 0.1
        W = np.random.randn(p * n, rank) * 0.1
        
        # Scale to help convergence
        U_scale = np.sqrt(rank_school / rank)
        U *= U_scale
        V *= U_scale
        W *= U_scale
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            
            # Compute gradients
            U_grad = np.zeros_like(U)
            V_grad = np.zeros_like(V)
            W_grad = np.zeros_like(W)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # Current value
                        val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                        error = val - 1.0
                        
                        # Gradient contributions
                        U_grad[i*m+j, :] += error * V[j*p+k, :] * W[k*n+i, :]
                        V_grad[j*p+k, :] += error * U[i*m+j, :] * W[k*n+i, :]
                        W_grad[k*n+i, :] += error * U[i*m+j, :] * V[j*p+k, :]
            
            # Update
            U -= lr * U_grad
            V -= lr * V_grad
            W -= lr * W_grad
            
            # Check convergence
            if it % 50 == 0:
                err = check_decomposition(U, V, W)
                if err < 1e-8:
                    return (U, V, W)
        
        # Final check
        err = check_decomposition(U, V, W)
        if err < 1e-6:
            return (U, V, W)
        return None
    
    # Try ranks from 83 down to 60
    for rank in range(min(83, best_rank - 1), 59, -1):
        if time.time() > deadline:
            break
        
        # Try multiple seeds for this rank
        for seed in range(10):
            if time.time() > deadline:
                break
            
            result = try_rank(rank, max_iter=500, lr=0.005, seed=seed)
            if result is not None:
                U, V, W = result
                best_rank = rank
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                break  # Found a valid decomposition for this rank
        
        if best_rank <= rank:
            # Found something at this rank or better, continue to lower ranks
            continue
    
    # Round to half-integers if close
    def round_to_halfint(arr):
        rounded = np.round(arr * 2) / 2
        # Check if rounding is close
        if np.max(np.abs(arr - rounded)) < 1e-4:
            return rounded
        return arr
    
    best_U = round_to_halfint(best_U)
    best_V = round_to_halfint(best_V)
    best_W = round_to_halfint(best_W)
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
