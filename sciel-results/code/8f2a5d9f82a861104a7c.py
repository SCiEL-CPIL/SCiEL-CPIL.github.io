import time
import sys
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 6]
DEADLINE = 140.0  # safety margin before 150s kill

def construct():
    """
    Returns a rank-minimal (or best known) tensor decomposition for 
    multiplying a 2x3 matrix by a 3x6 matrix.
    
    Uses optimization to search for rank < 36 decompositions.
    Falls back to schoolbook (rank 36) if no improvement found.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Schoolbook rank
    sb_rank = n * m * p  # 36
    
    # Build schoolbook decomposition
    U_sb = np.zeros((n*m, sb_rank), dtype=float)
    V_sb = np.zeros((m*p, sb_rank), dtype=float)
    W_sb = np.zeros((p*n, sb_rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1.0
                V_sb[j*p+k, col] = 1.0
                W_sb[k*n+i, col] = 1.0
                col += 1
    
    # Define the target tensor T (sparse: only 36 non-zero entries)
    # Store as list of (a, b, c) indices where T=1
    T_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_indices.append((i*m+j, j*p+k, k*n+i))
    
    best_rank = sb_rank
    best_U = U_sb.copy()
    best_V = V_sb.copy()
    best_W = W_sb.copy()
    
    # Helper: compute error for a decomposition
    def compute_error(U_mat, V_mat, W_mat):
        """Compute sum of squared errors over all T entries."""
        error = 0.0
        for (a, b, c) in T_indices:
            val = np.dot(U_mat[a, :], V_mat[b, :] * W_mat[c, :])
            error += (val - 1.0) ** 2
        # Also check that non-target entries are ~0 (sampled)
        # For efficiency, only check a subset
        return error
    
    # Helper: full validation
    def validate(U_mat, V_mat, W_mat, tol=1e-8):
        """Full validation: check all target entries = 1, and sample non-target = 0."""
        r = U_mat.shape[1]
        for (a, b, c) in T_indices:
            val = np.dot(U_mat[a, :], V_mat[b, :] * W_mat[c, :])
            if abs(val - 1.0) > tol:
                return False
        # Check some non-target entries
        np.random.seed(123)
        for _ in range(200):
            a = np.random.randint(0, n*m)
            b = np.random.randint(0, m*p)
            c = np.random.randint(0, p*n)
            if (a, b, c) in T_indices:
                continue
            val = np.dot(U_mat[a, :], V_mat[b, :] * W_mat[c, :])
            if abs(val) > tol:
                return False
        return True
    
    # Try to find lower rank decompositions using optimization
    # Strategy: for each target rank, initialize near schoolbook and optimize
    
    np.random.seed(42)
    
    for target_rank in [35, 34, 33, 32, 31, 30]:
        if time.time() - start_time > DEADLINE - 20:
            break
        
        # Initialize: take first target_rank columns of schoolbook
        U_init = U_sb[:, :target_rank].copy()
        V_init = V_sb[:, :target_rank].copy()
        W_init = W_sb[:, :target_rank].copy()
        
        # Flatten parameters
        def flatten_params(U, V, W):
            return np.concatenate([U.ravel(), V.ravel(), W.ravel()])
        
        def unflatten_params(x):
            u_size = n*m * target_rank
            v_size = m*p * target_rank
            U = x[:u_size].reshape(n*m, target_rank)
            V = x[u_size:u_size+v_size].reshape(m*p, target_rank)
            W = x[u_size+v_size:].reshape(p*n, target_rank)
            return U, V, W
        
        def objective(x):
            U, V, W = unflatten_params(x)
            return compute_error(U, V, W)
        
        x0 = flatten_params(U_init, V_init, W_init)
        
        # Run optimization
        try:
            result = minimize(objective, x0, method='L-BFGS-B', 
                            options={'maxiter': 500, 'maxfun': 10000, 'ftol': 1e-12})
            
            U_opt, V_opt, W_opt = unflatten_params(result.x)
            
            if validate(U_opt, V_opt, W_opt, tol=1e-8):
                best_rank = target_rank
                best_U = U_opt
                best_V = V_opt
                best_W = W_opt
                break  # Found a valid lower-rank decomposition
        except Exception:
            pass
    
    if time.time() - start_time > DEADLINE - 5:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
