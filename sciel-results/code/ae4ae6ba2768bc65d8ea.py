"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import scipy
import time
import numpy as np
from scipy.optimize import nnls

DIMENSIONS = [2, 2, 3]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x2 * 2x3 matrix multiplication.
    """
    n, m, p = DIMENSIONS  # n=2, m=2, p=3
    
    # The tensor T[i,j,k] = 1 for all i in range(n), j in range(m), k in range(p)
    # We want: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all (i,j,k)
    
    # First, create the schoolbook solution as fallback (Rank 12)
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_sb

    deadline = time.time() + 140  # Leave 10s margin
    
    # Try to find lower rank decompositions
    # Known lower bound is 6, known rank is 8.
    # We try 8 first as it is the known minimal rank.
    for r in [8, 7, 6]:
        if time.time() > deadline:
            break
        
        result = _find_decomposition(n, m, p, r, deadline)
        if result is not None:
            best_result = result
            best_rank = r
            break  # We found the minimum rank, stop
    
    return best_result


def _find_decomposition(n, m, p, r, deadline):
    """
    Try to find a rank-r decomposition.
    Uses random initialization and L-BFGS-B optimization.
    """
    np.random.seed(42)
    
    def loss_and_grad(params):
        U_flat = params[:n*m*r]
        V_flat = params[n*m*r : n*m*r + m*p*r]
        W_flat = params[n*m*r + m*p*r :]
        
        U = U_flat.reshape(n*m, r)
        V = V_flat.reshape(m*p, r)
        W = W_flat.reshape(p*n, r)
        
        vals = np.zeros((n, m, p))
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    vals[i, j, k] = val
                    err = val - 1.0
                    # Gradients
                    grad_U[i*m+j, :] += 2 * err * (V[j*p+k, :] * W[k*n+i, :])
                    grad_V[j*p+k, :] += 2 * err * (U[i*m+j, :] * W[k*n+i, :])
                    grad_W[k*n+i, :] += 2 * err * (U[i*m+j, :] * V[j*p+k, :])
        
        loss = np.sum(err**2)
        
        # Flatten gradients
        grad_flat = np.concatenate([
            grad_U.flatten(), 
            grad_V.flatten(), 
            grad_W.flatten()
        ])
        
        return loss, grad_flat

    for attempt in range(100):
        if time.time() > deadline:
            return None
        
        # Initialize with small random values to help convergence
        U_init = np.random.randn(n * m, r) * 0.1
        V_init = np.random.randn(m * p, r) * 0.1
        W_init = np.random.randn(p * n, r) * 0.1
        
        x0 = np.concatenate([
            U_init.flatten(), 
            V_init.flatten(), 
            W_init.flatten()
        ])
        
        try:
            # Use L-BFGS-B for robust optimization
            result = scipy.optimize.minimize(
                loss_and_grad, 
                x0, 
                method='L-BFGS-B', 
                jac=True, 
                options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if result.success or result.fun < 1e-6:
                # Extract optimized parameters
                U_opt = result.x[:n*m*r].reshape(n*m, r)
                V_opt = result.x[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
                W_opt = result.x[n*m*r + m*p*r :].reshape(p*n, r)
                
                # Try to round to half-integers
                # We check a few rounding strategies
                for scale in [1, 2]:
                    U_rounded = np.round(U_opt * scale) / scale
                    V_rounded = np.round(V_opt * scale) / scale
                    W_rounded = np.round(W_opt * scale) / scale
                    
                    # Verify exactness
                    loss_rounded = 0.0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = np.dot(U_rounded[i*m+j, :], V_rounded[j*p+k, :] * W_rounded[k*n+i, :])
                                loss_rounded += (val - 1.0)**2
                                
                    if loss_rounded < 1e-6:
                        U_list = [[float(U_rounded[i, c]) for c in range(r)] for i in range(n*m)]
                        V_list = [[float(V_rounded[i, c]) for c in range(r)] for i in range(m*p)]
                        W_list = [[float(W_rounded[i, c]) for c in range(r)] for i in range(p*n)]
                        return {"U": U_list, "V": V_list, "W": W_list}
                        
        except Exception:
            continue
            
    return None
# EVOLVE-BLOCK-END
