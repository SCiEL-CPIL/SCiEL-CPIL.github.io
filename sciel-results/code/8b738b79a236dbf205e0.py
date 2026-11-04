"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p  # 63
    
    # Build the target tensor T[i,j,k] = 1 for all i,j,k
    # T is n*m*p = 63 dimensional
    T = np.ones((n, m, p))
    
    # We'll try to find a decomposition with rank < 63
    # Start with schoolbook and try to reduce rank via random initialization and optimization
    
    best_rank = rank
    best_U = None
    best_V = None
    best_W = None
    
    # Schoolbook decomposition as fallback
    U_sb = np.zeros((n*m, rank))
    V_sb = np.zeros((m*p, rank))
    W_sb = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # Try to find lower rank decompositions
    # The tensor equation is: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # for all i, j, k
    
    deadline = time.time() + 130  # Leave 20s safety margin
    
    # Try various ranks
    for target_rank in [62, 61, 60, 59, 58, 57, 56, 55]:
        if time.time() > deadline:
            break
        
        # Initialize with random values
        np.random.seed(42)
        U_init = np.random.randn(n*m, target_rank) * 0.1
        V_init = np.random.randn(m*p, target_rank) * 0.1
        W_init = np.random.randn(p*n, target_rank) * 0.1
        
        def objective(params):
            U_flat, V_flat, W_flat = params
            U_r = U_flat.reshape(n*m, target_rank)
            V_r = V_flat.reshape(m*p, target_rank)
            W_r = W_flat.reshape(p*n, target_rank)
            
            # Compute sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
            # This is a tensor contraction
            result = np.zeros((n, m, p))
            for c in range(target_rank):
                uc = U_r[:, c].reshape(n, m)
                vc = V_r[:, c].reshape(m, p)
                wc = W_r[:, c].reshape(p, n)
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            result[i, j, k] += uc[i, j] * vc[j, k] * wc[k, i]
            
            error = np.sum((result - 1.0)**2)
            return error
        
        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        # Use L-BFGS-B
        res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-10})
        
        if res.fun < 1e-8:
            U_r = res.x[:n*m*target_rank].reshape(n*m, target_rank)
            V_r = res.x[n*m*target_rank:(n*m+m*p)*target_rank].reshape(m*p, target_rank)
            W_r = res.x[(n*m+m*p)*target_rank:].reshape(p*n, target_rank)
            
            # Round to nearest half-integer or integer if close
            U_r = np.round(U_r * 2) / 2
            V_r = np.round(V_r * 2) / 2
            W_r = np.round(W_r * 2) / 2
            
            # Verify
            result = np.zeros((n, m, p))
            for c in range(target_rank):
                uc = U_r[:, c].reshape(n, m)
                vc = V_r[:, c].reshape(m, p)
                wc = W_r[:, c].reshape(p, n)
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            result[i, j, k] += uc[i, j] * vc[j, k] * wc[k, i]
            
            if np.max(np.abs(result - 1.0)) < 1e-10:
                best_rank = target_rank
                best_U, best_V, best_W = U_r, V_r, W_r
                break
    
    # Convert to list format
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
