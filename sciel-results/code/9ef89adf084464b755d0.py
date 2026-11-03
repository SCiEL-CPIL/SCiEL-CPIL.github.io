import numpy as np
import time

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    # Start with schoolbook
    rank_schoolbook = n * m * p
    U = np.zeros((n * m, rank_schoolbook))
    V = np.zeros((m * p, rank_schoolbook))
    W = np.zeros((p * n, rank_schoolbook))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                col = i * m * p + j * p + k
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
    
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U, V, W
    
    # Try lower ranks
    # Lower bound is roughly 12, but let's start higher
    for r in [24, 20, 18, 16, 14, 12]:
        if time.time() > 100: # Safety margin
            break
        success = False
        for restart in range(10):
            if time.time() > 110:
                break
            # Random initialization
            U_r = np.random.randn(n * m, r) * 0.1
            V_r = np.random.randn(m * p, r) * 0.1
            W_r = np.random.randn(p * n, r) * 0.1
            
            # Optimize
            # We want to minimize || U * V * W - T ||_F^2
            # where T is the tensor with 1s at (i,j,k)
            # This is a 3-way tensor.
            # We can flatten it.
            
            # Let's use a simple gradient descent on the flattened parameters
            # Or use scipy minimize on the concatenated vector
            
            # For speed, let's try a few iterations of alternating minimization
            # Fix U, V, solve for W?
            # The objective is sum_{i,j,k} (sum_t U_{i,j,t} V_{j,k,t} W_{k,i,t} - 1)^2
            # This is quadratic in W if U, V are fixed.
            # Let's build the linear system for W.
            
            # For each t, W_{k,i,t} is a variable.
            # The term is sum_t W_{k,i,t} (U_{i,j,t} V_{j,k,t})
            # So for fixed i,j,k, the equation is:
            # sum_t W_{k,i,t} A_{i,j,k,t} = 1
            # where A_{i,j,k,t} = U_{i,j,t} V_{j,k,t}
            
            # This is a linear system in W.
            # Number of equations: n*m*p = 48
            # Number of variables: p*n*r = 12r
            # If r >= 4, we have more variables than equations.
            # We can use least squares.
            
            # Let's reformulate.
            # We want to find W such that:
            # sum_t W_{k,i,t} U_{i,j,t} V_{j,k,t} = 1 for all i,j,k.
            
            # Let's create the matrix M_W of shape (n*m*p, p*n*r)
            # Row index: (i,j,k)
            # Col index: (k,i,t)
            # Entry: U_{i,j,t} V_{j,k,t}
            
            # This matrix is sparse? No, dense.
            # Size: 48 x (12r)
            # For r=24, 48 x 288.
            
            # We can solve for W using least squares.
            # Then fix W, solve for V, etc.
            
            # Let's implement alternating least squares.
            
            U_curr = U_r
            V_curr = V_r
            W_curr = W_r
            
            for iter in range(100):
                # Solve for W
                # Build M_W
                M_W = np.zeros((n * m * p, p * n * r))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            row = i * m * p + j * p + k
                            for t in range(r):
                                col = k * n * r + i * r + t
                                M_W[row, col] = U_curr[i * m + j, t] * V_curr[j * p + k, t]
                
                # Target vector is 1s
                target = np.ones(n * m * p)
                W_flat, residuals, rank, s = np.linalg.lstsq(M_W, target, rcond=None)
                W_curr = W_flat.reshape(p * n, r)
                
                # Solve for V
                M_V = np.zeros((n * m * p, m * p * r))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            row = i * m * p + j * p + k
                            for t in range(r):
                                col = j * p * r + k * r + t
                                M_V[row, col] = U_curr[i * m + j, t] * W_curr[k * n + i, t]
                
                V_flat, residuals, rank, s = np.linalg.lstsq(M_V, target, rcond=None)
                V_curr = V_flat.reshape(m * p, r)
                
                # Solve for U
                M_U = np.zeros((n * m * p, n * m * r))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            row = i * m * p + j * p + k
                            for t in range(r):
                                col = i * m * r + j * r + t
                                M_U[row, col] = V_curr[j * p + k, t] * W_curr[k * n + i, t]
                
                U_flat, residuals, rank, s = np.linalg.lstsq(M_U, target, rcond=None)
                U_curr = U_flat.reshape(n * m, r)
                
                # Check error
                error = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.sum(U_curr[i * m + j] * V_curr[j * p + k] * W_curr[k * n + i])
                            error += (val - 1)**2
                
                if error < 1e-6:
                    success = True
                    break
            
            if success:
                best_rank = r
                best_U, best_V, best_W = U_curr, V_curr, W_curr
                break # Try next lower rank
        
        if success:
            # Try to prune?
            # For now, just record
            pass
    
    # Convert to lists
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
