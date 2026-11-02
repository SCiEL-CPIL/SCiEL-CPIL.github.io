import time
import random
import numpy as np

DIMENSIONS = [4, 5, 6]
START_TIME = time.time()
TIME_LIMIT = 120.0  # seconds

def construct():
    """
    Returns a decomposition for 4x5 by 5x6 matrix multiplication.
    Attempts to find a rank-minimal decomposition using random search.
    Falls back to schoolbook decomposition if no better rank is found.
    """
    n, m, p = DIMENSIONS
    nm = n * m  # 20
    mp = m * p  # 30
    pn = p * n  # 24
    
    # Schoolbook decomposition
    rank_school = nm * mp * pn  # 120
    best_rank = rank_school
    best_U = [[0] * rank_school for _ in range(nm)]
    best_V = [[0] * rank_school for _ in range(mp)]
    best_W = [[0] * rank_school for _ in range(pn)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i * m + j][col] = 1
                best_V[j * p + k][col] = 1
                best_W[k * n + i][col] = 1
                col += 1
    
    # Try random search for lower rank decompositions
    # The tensor equation: T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # T[i,j,k] = 1 for all i,j,k (it's the matrix multiplication tensor)
    
    # We need to find rank r such that there exist U, V, W with integer/half-integer entries
    # where for all i, j, k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # Start trying ranks from 100 down to 100 (limited search due to time constraints)
    # Actually, let's try a few random ranks between 80 and 120
    
    # For a valid decomposition with rank r, we need:
    # For each (i,j,k): sum_{r=1}^{R} U[i*m+j, r-1] * V[j*p+k, r-1] * W[k*n+i, r-1] = 1
    
    # This is a system of nonlinear equations. Let's try random initialization and Newton's method
    # or gradient descent.
    
    # Due to time constraints, we'll do a limited random search
    # Try ranks: 100, 105, 110, 115
    target_ranks = [110, 105, 100, 95, 90]
    
    for rank in target_ranks:
        if time.time() - START_TIME > TIME_LIMIT - 10:
            break
            
        # Try multiple random initializations
        for attempt in range(50):
            if time.time() - START_TIME > TIME_LIMIT - 10:
                break
            
            # Initialize with random small values
            U = np.random.randn(nm, rank) * 0.1
            V = np.random.randn(mp, rank) * 0.1
            W = np.random.randn(pn, rank) * 0.1
            
            # Gradient descent to minimize the error
            # Error = sum_{i,j,k} (1 - sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r])^2
            
            lr = 0.01
            for step in range(100):
                if time.time() - START_TIME > TIME_LIMIT - 10:
                    break
                    
                # Compute error and gradients
                err = 0
                U_grad = np.zeros_like(U)
                V_grad = np.zeros_like(V)
                W_grad = np.zeros_like(W)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            idx_u = i * m + j
                            idx_v = j * p + k
                            idx_w = k * n + i
                            val = np.sum(U[idx_u] * V[idx_v] * W[idx_w])
                            diff = 1.0 - val
                            err += diff * diff
                            for r in range(rank):
                                U_grad[idx_u][r] += -2 * diff * V[idx_v][r] * W[idx_w][r]
                                V_grad[idx_v][r] += -2 * diff * U[idx_u][r] * W[idx_w][r]
                                W_grad[idx_w][r] += -2 * diff * U[idx_u][r] * V[idx_v][r]
                
                # Update
                U -= lr * U_grad
                V -= lr * V_grad
                W -= lr * W_grad
                
                if err < 1e-6:
                    break
            
            # Check if we found a valid solution
            if err < 1e-6:
                # Convert to integer/half-integer if possible
                U_int = np.round(U * 2) / 2
                V_int = np.round(V * 2) / 2
                W_int = np.round(W * 2) / 2
                
                # Verify
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            idx_u = i * m + j
                            idx_v = j * p + k
                            idx_w = k * n + i
                            val = np.sum(U_int[idx_u] * V_int[idx_v] * W_int[idx_w])
                            if abs(val - 1.0) > 1e-9:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                
                if valid:
                    # Store as list of lists with integer/half-integer entries
                    best_rank = rank
                    best_U = U_int.tolist()
                    best_V = V_int.tolist()
                    best_W = W_int.tolist()
                    break
        
        if best_rank < rank_school:
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}
