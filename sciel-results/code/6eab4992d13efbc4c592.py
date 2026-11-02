"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import least_squares

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 2, 3, 4
    schoolbook_rank = n * m * p  # 24
    
    # Schoolbook solution as fallback
    U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = schoolbook_rank
    
    # Try ranks from schoolbook-1 down to some minimum
    for rank in range(schoolbook_rank - 1, 17, -1):
        if time.time() > deadline:
            break
        
        try:
            result = _try_rank(n, m, p, rank, deadline)
            if result is not None:
                best_result = result
                best_rank = rank
                continue
        except Exception:
            continue
    
    return best_result

def _try_rank(n, m, p, rank, deadline):
    """Try to find a rank decomposition with the given rank using least_squares."""
    num_params = (n * m + m * p + p * n) * rank
    
    # Residual function for least_squares
    def residual_function(params):
        U = params[:n*m*rank].reshape(n*m, rank)
        V = params[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
        W = params[n*m*rank + m*p*rank:].reshape(p*n, rank)
        
        # Efficient tensor contraction
        X = np.einsum('ir,jr,kr->ijk', U, V, W)
        
        # Target tensor: 1 at (i,j,k) for all i,j,k
        T = np.ones((n, m, p))
        
        residuals = (X - T).flatten()
        return residuals

    # Multiple restarts to escape local minima
    np.random.seed(int(time.time() * 1000) % 10000)
    
    for restart in range(10):
        if time.time() > deadline:
            break
            
        # Random initialization
        U0 = np.random.randn(n * m, rank)
        V0 = np.random.randn(m * p, rank)
        W0 = np.random.randn(p * n, rank)
        
        # Scale initialization to be closer to expected magnitude
        scale = 1.0 / np.sqrt(rank)
        U0 *= scale
        V0 *= scale
        W0 *= scale
        
        params0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
        
        try:
            result = least_squares(residual_function, params0, method='trf', max_nfev=200)
        except Exception:
            continue
            
        if result.cost < 1e-10:
            U_opt = result.x[:n*m*rank].reshape(n*m, rank)
            V_opt = result.x[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
            W_opt = result.x[n*m*rank + m*p*rank:].reshape(p*n, rank)
            
            # Try snapping to half-integers
            for denom in [1, 2, 3, 4, 5, 6, 8]:
                U_snap = np.round(U_opt * denom) / denom
                V_snap = np.round(V_opt * denom) / denom
                W_snap = np.round(W_opt * denom) / denom
                
                # Verify exactness
                X = np.einsum('ir,jr,kr->ijk', U_snap, V_snap, W_snap)
                max_err = np.max(np.abs(X - 1.0))
                
                if max_err < 1e-9:
                    return {
                        "U": U_snap.tolist(),
                        "V": V_snap.tolist(),
                        "W": W_snap.tolist()
                    }
    
    return None

# EVOLVE-BLOCK-END
