"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition is exact."""
    if not U or not V or not W:
        return False
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    if U_arr.shape != (n*m, rank):
        return False
    if V_arr.shape != (m*p, rank):
        return False
    if W_arr.shape != (p*n, rank):
        return False
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                tensor_val = 1.0
                decomp_val = 0.0
                for c in range(rank):
                    decomp_val += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                if abs(tensor_val - decomp_val) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
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
    return {"U": U, "V": V, "W": W}

def _tensor_error(U, V, W, n, m, p):
    """Compute the squared error of the tensor reconstruction."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    
    total_error = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target = 1.0
                recon = 0.0
                for c in range(rank):
                    recon += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                total_error += (target - recon) ** 2
    return total_error

def _round_to_half_int(arr):
    """Round array to nearest 0.5."""
    return np.round(arr * 2) / 2

def _try_l_bfgs(n, m, p, rank, time_limit, seed=0):
    """Try to find a rank-rank decomposition using L-BFGS-B."""
    np.random.seed(seed)
    
    # Initialize with random values
    U_init = np.random.randn(n*m, rank) * 0.1
    V_init = np.random.randn(m*p, rank) * 0.1
    W_init = np.random.randn(p*n, rank) * 0.1
    
    # Flatten all parameters into a single vector
    params = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
    
    U_size = n*m * rank
    V_size = m*p * rank
    
    def objective(params):
        U_flat = params[:U_size]
        V_flat = params[U_size:U_size+V_size]
        W_flat = params[U_size+V_size:]
        
        U = U_flat.reshape(n*m, rank)
        V = V_flat.reshape(m*p, rank)
        W = W_flat.reshape(p*n, rank)
        
        return _tensor_error(U, V, W, n, m, p)
    
    # Use a tight time limit for each optimization
    try:
        result = minimize(objective, params, method='L-BFGS-B', 
                          options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-6, 'maxfun': 1000})
    except Exception:
        return None
    
    if result.fun < 1e-4:
        params = result.x
        U = params[:U_size].reshape(n*m, rank)
        V = params[U_size:U_size+V_size].reshape(m*p, rank)
        W = params[U_size+V_size:].reshape(p*n, rank)
        
        # Round to nearest 0.5
        U_r = _round_to_half_int(U)
        V_r = _round_to_half_int(V)
        W_r = _round_to_half_int(W)
        
        U_list = U_r.tolist()
        V_list = V_r.tolist()
        W_list = W_r.tolist()
        
        if _verify_decomposition(U_list, V_list, W_list, n, m, p):
            return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Best known: schoolbook gives rank 54
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try to find lower rank decompositions
    # We test a few specific ranks that are likely candidates for improvement
    candidate_ranks = [45, 40, 35, 30, 25]
    
    for rank in candidate_ranks:
        if rank >= best_rank:
            continue
            
        # Check time
        if time.time() - start_time > 100:
            break
        
        # Try multiple random initializations for this rank
        found = False
        for seed in range(20):
            if time.time() - start_time > 110:
                break
                
            result = _try_l_bfgs(n, m, p, rank, 2.0, seed)
            if result is not None:
                best = result
                best_rank = rank
                found = True
                break
        
        if found:
            # If we found a solution, we can try to go lower, but let's be conservative
            # to ensure we finish in time. We can break here or continue.
            # Continuing might find an even lower rank.
            # However, lower ranks are harder. Let's continue if we have time.
            pass
    
    return best

# EVOLVE-BLOCK-END
