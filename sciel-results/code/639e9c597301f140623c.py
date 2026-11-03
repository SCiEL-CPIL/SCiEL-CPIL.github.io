"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Build the multiplication tensor
    # T[u_idx, v_idx, w_idx] where u_idx = i*m+j, v_idx = j*p+k, w_idx = k*n+i
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0
    
    # Try to find a low-rank decomposition
    # We'll try ranks from 16 up to 18
    best_result = None
    best_rank = n * m * p  # schoolbook
    
    deadline = time.time() + 120  # safety margin
    
    for rank in range(16, best_rank + 1):
        if time.time() > deadline - 30:
            break
        
        # Try multiple random initializations
        found = False
        for trial in range(50):
            if time.time() > deadline - 30:
                break
            
            # Random initialization
            rng = np.random.default_rng(trial * 1000 + rank)
            U = rng.standard_normal((n*m, rank)) * 0.5
            V = rng.standard_normal((m*p, rank)) * 0.5
            W = rng.standard_normal((p*n, rank)) * 0.5
            
            # Flatten and optimize
            def loss(x):
                U_flat = x[:n*m*rank].reshape(n*m, rank)
                V_flat = x[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
                W_flat = x[n*m*rank + m*p*rank:].reshape(p*n, rank)
                
                # Compute the reconstructed tensor
                # T_recon[u,v,w] = sum_c U[u,c] * V[v,c] * W[w,c]
                # This is a 3-tensor contraction
                # Efficient computation: for each c, outer product of U[:,c], V[:,c], W[:,c]
                recon = np.zeros((n*m, m*p, p*n))
                for c in range(rank):
                    recon += np.einsum('i,j,k->ijk', U_flat[:,c], V_flat[:,c], W_flat[:,c])
                
                # Loss: sum of squared errors
                return np.sum((recon - T) ** 2)
            
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            # Use L-BFGS
            try:
                result = minimize(loss, x0, method='L-BFGS-B', 
                                  options={'maxiter': 1000, 'maxfun': 10000})
                
                if result.fun < 1e-10:
                    # Check if we can round to half-integers
                    U_opt = result.x[:n*m*rank].reshape(n*m, rank)
                    V_opt = result.x[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
                    W_opt = result.x[n*m*rank + m*p*rank:].reshape(p*n, rank)
                    
                    # Round to nearest half-integer
                    U_round = np.round(U_opt * 2) / 2
                    V_round = np.round(V_opt * 2) / 2
                    W_round = np.round(W_opt * 2) / 2
                    
                    # Verify
                    recon = np.zeros((n*m, m*p, p*n))
                    for c in range(rank):
                        recon += np.einsum('i,j,k->ijk', U_round[:,c], V_round[:,c], W_round[:,c])
                    
                    if np.max(np.abs(recon - T)) < 1e-9:
                        # Found exact solution!
                        best_result = {
                            "U": U_round.tolist(),
                            "V": V_round.tolist(),
                            "W": W_round.tolist()
                        }
                        best_rank = rank
                        found = True
                        break
                    elif result.fun < 1e-10:
                        # Use the unrounded solution if it's exact in float
                        best_result = {
                            "U": U_opt.tolist(),
                            "V": V_opt.tolist(),
                            "W": W_opt.tolist()
                        }
                        best_rank = rank
                        found = True
                        break
            except Exception:
                continue
        
        if found:
            break
    
    if best_result is None:
        # Fall back to schoolbook
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
        best_result = {"U": U, "V": V, "W": W}
    
    return best_result
# EVOLVE-BLOCK-END
