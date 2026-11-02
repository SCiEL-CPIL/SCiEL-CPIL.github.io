import numpy as np
import time
import random

DIMENSIONS = [3, 5, 7]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    deadline = time.time() + 140
    
    # Schoolbook
    rank = n * m * p
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    def verify(U, V, W, tol=1e-9):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(U.shape[1]))
                    if abs(val - 1) > tol:
                        return False
        return True
    
    # Local search to reduce rank
    best_rank = rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    attempts = 0
    while time.time() < deadline - 10 and attempts < 500:
        attempts += 1
        # Try to eliminate a random column
        tc = random.randint(0, best_rank - 1)
        
        # Save current state
        U_try = best_U.copy()
        V_try = best_V.copy()
        W_try = best_W.copy()
        
        # Zero out column tc
        U_try[:, tc] = 0
        V_try[:, tc] = 0
        W_try[:, tc] = 0
        
        # Compute the error
        # The error at (i,j,k) is: U_orig[i*m+j,tc] * V_orig[j*p+k,tc] * W_orig[k*n+i,tc]
        # We need to compensate this using other columns
        
        # Try to find small adjustments to other columns to fix the error
        # Use gradient descent on the adjustments
        
        # For simplicity, try random small perturbations to non-zero columns
        improved = False
        for _ in range(50):
            # Pick a random column to adjust
            adj_col = random.randint(0, best_rank - 1)
            if adj_col == tc:
                continue
            
            # Try small random adjustments
            for scale in [0.01, 0.1, 0.5]:
                U_adj = U_try.copy()
                V_adj = V_try.copy()
                W_adj = W_try.copy()
                
                # Add small random values to column adj_col
                U_adj[:, adj_col] += random.uniform(-scale, scale)
                V_adj[:, adj_col] += random.uniform(-scale, scale)
                W_adj[:, adj_col] += random.uniform(-scale, scale)
                
                if verify(U_adj, V_adj, W_adj, tol=1e-6):
                    # Found a valid decomposition with one fewer term
                    best_U, best_V, best_W = U_adj, V_adj, W_adj
                    best_rank -= 1
                    improved = True
                    break
            
            if improved:
                break
        
        if not improved and time.time() > deadline - 20:
            break
    
    # Round to nearest half-integer
    best_U = np.round(best_U * 2) / 2
    best_V = np.round(best_V * 2) / 2
    best_W = np.round(best_W * 2) / 2
    
    # Final verification
    if not verify(best_U, best_V, best_W, tol=1e-9):
        # Fallback to schoolbook
        best_U, best_V, best_W = U, V, W
        best_rank = rank
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
