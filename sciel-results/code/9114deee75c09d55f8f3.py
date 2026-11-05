import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 4, 7]

def construct():
    start_time = time.time()
    DEADLINE = 140  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Build the tensor T[i,j,k] = 1 for all i,j,k
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    # Start with schoolbook rank 56
    # We will try to find a lower rank decomposition.
    
    def verify(U, V, W, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    lhs = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(rank))
                    if abs(lhs - 1.0) > 1e-6:
                        return False
        return True

    best_rank = 56
    best_U = np.zeros((nm, 56))
    best_V = np.zeros((mp, 56))
    best_W = np.zeros((pn, 56))
    
    # Initialize with schoolbook
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i*m+j, col] = 1
                best_V[j*p+k, col] = 1
                best_W[k*n+i, col] = 1
                col += 1
    
    # Try to reduce rank
    # We will try ranks from 50 down to 40
    for target_rank in range(50, 39, -1):
        if time.time() - start_time > DEADLINE - 20:
            break
        
        # Try multiple random initializations
        for attempt in range(5):
            if time.time() - start_time > DEADLINE - 10:
                break
            
            np.random.seed(42 + attempt + target_rank)
            
            # Initialize with random sparse matrices
            U_init = np.random.randn(nm, target_rank) * 0.1
            V_init = np.random.randn(mp, target_rank) * 0.1
            W_init = np.random.randn(pn, target_rank) * 0.1
            
            # Add some schoolbook terms to help convergence
            for r in range(min(target_rank, 56)):
                if r < nm:
                    U_init[r % nm, r] += 1
                    V_init[(r // nm) % mp, r] += 1
                    W_init[(r // (nm*mp)) % pn, r] += 1
            
            def objective(params):
                U_p = params[:nm*target_rank].reshape(nm, target_rank)
                V_p = params[nm*target_rank:nm*target_rank+mp*target_rank].reshape(mp, target_rank)
                W_p = params[nm*target_rank+mp*target_rank:].reshape(pn, target_rank)
                
                error = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            lhs = sum(U_p[i*m+j, r] * V_p[j*p+k, r] * W_p[k*n+i, r] for r in range(target_rank))
                            error += (lhs - 1.0) ** 2
                
                # L2 regularization to keep values small
                reg = 0.01 * (np.sum(U_p**2) + np.sum(V_p**2) + np.sum(W_p**2))
                
                return error + reg
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                result = minimize(objective, x0, method='L-BFGS-B', 
                                 options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50})
                
                if result.fun < 1e-6:
                    U_opt = result.x[:nm*target_rank].reshape(nm, target_rank)
                    V_opt = result.x[nm*target_rank:nm*target_rank+mp*target_rank].reshape(mp, target_rank)
                    W_opt = result.x[nm*target_rank+mp*target_rank:].reshape(pn, target_rank)
                    
                    # Round to nearest half-integer
                    U_round = np.round(U_opt * 2) / 2
                    V_round = np.round(V_opt * 2) / 2
                    W_round = np.round(W_opt * 2) / 2
                    
                    if verify(U_round, V_round, W_round, target_rank):
                        best_rank = target_rank
                        best_U, best_V, best_W = U_round, V_round, W_round
                        break  # Found a valid lower-rank decomposition
            except Exception:
                pass
        
        if best_rank < target_rank + 1:
            break  # Found a better rank, stop
    
    # Convert to list format
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
