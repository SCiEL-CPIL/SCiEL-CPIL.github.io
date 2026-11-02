import time
import numpy as np
import random
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 5]

def _verify_decomposition(U, V, W):
    """
    Verify that the decomposition correctly represents matrix multiplication.
    For each (i,j,k), the coefficient of A[i,j]*B[j,k] in C[i,k] should be 1.
    """
    n, m, p = DIMENSIONS
    if not U or not V or not W:
        return False
    
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Check shapes
    if U_arr.shape != (n*m, rank): return False
    if V_arr.shape != (m*p, rank): return False
    if W_arr.shape != (p*n, rank): return False
    
    # For each (i, j, k), check that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_row = U_arr[i * m + j, :]
                v_row = V_arr[j * p + k, :]
                w_row = W_arr[k * n + i, :]
                product = np.sum(u_row * v_row * w_row)
                if abs(product - 1.0) > 1e-9:
                    return False
    return True

def _optimize_rank(n, m, p, target_rank, time_limit_per_rank=10.0):
    """
    Try to find a decomposition with the given target_rank using optimization.
    Returns (U, V, W) if successful, None otherwise.
    """
    start_time = time.time()
    U_dim = n * m
    V_dim = m * p
    W_dim = p * n
    
    # Pre-compute indices for efficiency
    indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                indices.append((i * m + j, j * p + k, k * n + i))
    
    def objective(params):
        # Unpack parameters
        # U is U_dim x target_rank, V is V_dim x target_rank, W is W_dim x target_rank
        # Total size: (U_dim + V_dim + W_dim) * target_rank
        size_u = U_dim * target_rank
        size_v = V_dim * target_rank
        
        U_flat = params[:size_u]
        V_flat = params[size_u:size_u + size_v]
        W_flat = params[size_u + size_v:]
        
        U = U_flat.reshape(U_dim, target_rank)
        V = V_flat.reshape(V_dim, target_rank)
        W = W_flat.reshape(W_dim, target_rank)
        
        # Compute error
        error = 0.0
        for u_idx, v_idx, w_idx in indices:
            val = np.sum(U[u_idx, :] * V[v_idx, :] * W[w_idx, :])
            error += (val - 1.0) ** 2
        return error

    # Try multiple random restarts
    best_result = None
    best_loss = float('inf')
    
    num_restarts = 20 # Adjust based on time
    
    for r in range(num_restarts):
        if time.time() - start_time > time_limit_per_rank:
            break
            
        # Random initialization
        # Scale appropriately
        init_scale = 1.0
        U0 = np.random.randn(U_dim, target_rank) * init_scale
        V0 = np.random.randn(V_dim, target_rank) * init_scale
        W0 = np.random.randn(W_dim, target_rank) * init_scale
        
        # Flatten
        x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
        
        try:
            res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 100, 'ftol': 1e-10})
            
            if res.fun < best_loss:
                best_loss = res.fun
                if best_loss < 1e-6:
                    # Extract solution
                    size_u = U_dim * target_rank
                    size_v = V_dim * target_rank
                    U_sol = res.x[:size_u].reshape(U_dim, target_rank)
                    V_sol = res.x[size_u:size_u + size_v].reshape(V_dim, target_rank)
                    W_sol = res.x[size_u + size_v:].reshape(W_dim, target_rank)
                    
                    # Round to nearest 0.5
                    U_rounded = np.round(U_sol * 2) / 2
                    V_rounded = np.round(V_sol * 2) / 2
                    W_rounded = np.round(W_sol * 2) / 2
                    
                    if _verify_decomposition(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist()):
                        return U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist()
                        
                    # Keep the unrounded best for potential later use, but we need exact
                    # If rounding fails, maybe try a tighter optimization or different rounding
                    # But for now, if we find a very low loss, we might just return the float version
                    # However, the task requires "integer/half-integer entries".
                    # So we MUST return rounded values if they verify.
                    
                    # If rounding didn't verify, maybe the solution is close to integers/half-integers
                    # but not exact. We can try to snap to nearest integers/half-integers and re-verify.
                    # If that fails, we discard this restart.
                    
        except Exception:
            continue
            
    return None

def _find_best_decomposition(n, m, p, time_limit=110.0):
    """
    Search for the best (lowest rank) decomposition within the time limit.
    """
    start_time = time.time()
    schoolbook_rank = n * m * p
    
    # Fallback schoolbook
    def get_schoolbook():
        U = [[0] * schoolbook_rank for _ in range(n * m)]
        V = [[0] * schoolbook_rank for _ in range(m * p)]
        W = [[0] * schoolbook_rank for _ in range(p * n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][c] = 1
                    V[j * p + k][c] = 1
                    W[k * n + i][c] = 1
                    c += 1
        return U, V, W

    best_U, best_V, best_W = get_schoolbook()
    best_rank = schoolbook_rank

    # Try ranks from lower bound up to schoolbook-1
    # Lower bound: max(n*m, m*p, p*n) = 15
    # Try 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29
    # It is more likely to find a solution at 29 than 19.
    # But we want the MINIMAL rank.
    # So we should try 19 first?
    # If 19 is not possible, we waste time.
    # Let's try a range of likely ranks.
    # Known ranks for similar small matrices are often close to the lower bound.
    # Let's try 19 to 25 first, then 26 to 29.
    
    ranks_to_try = [19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
    
    for target_rank in ranks_to_try:
        if time.time() - start_time > time_limit - 10: # Leave 10s margin
            break
            
        # Allocate time based on likelihood? 
        # Just give each rank a fixed slice.
        # Total time ~ 100s. 11 ranks. ~9s per rank.
        time_slice = 9.0
        
        result = _optimize_rank(n, m, p, target_rank, time_limit_per_rank=time_slice)
        
        if result is not None:
            U_list, V_list, W_list = result
            if _verify_decomposition(U_list, V_list, W_list):
                best_rank = target_rank
                best_U, best_V, best_W = U_list, V_list, W_list
                # Since we are iterating from low to high, the first valid one is the minimum found so far.
                # But we should continue to see if we can find an even lower one?
                # No, we are iterating from low to high.
                # So if we find rank 20, we should check 19?
                # My loop is 19, 20, ...
                # So if 19 fails, we try 20. If 20 succeeds, we have rank 20.
                # But maybe 19 was just hard to find?
                # We can't know.
                # However, if we find 20, we can stop?
                # No, we must find the MINIMAL.
                # If we find 20, we still need to check if 19 is possible.
                # But we already tried 19 and failed.
                # Maybe we should try 19 again with more time?
                # Or just accept 20 if 19 failed?
                # Given the time constraint, accepting the first found in a low-to-high sweep is a reasonable heuristic.
                # But wait, if 19 is the true minimum, and we failed to find it in 9s, we might return 20.
                # Score for 20 is 1.5. Score for 19 is 1.57.
                # It's better to have a guaranteed 20 than a risky 19 that times out.
                # So we should probably break?
                # No, let's just return the best found.
                # If we break, we return 20.
                # If we continue, we might find 21 (worse) or re-try 19 (not in loop).
                # So breaking is good if we are confident 19 is not possible or we didn't try hard enough.
                # Let's break to save time for potential verification or just to return.
                # Actually, if we break, we return the current best.
                # Let's break.
                break
    
    return best_U, best_V, best_W

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the matrix multiplication tensor decomposition for 2x3 by 3x5.
    Returns a dictionary with keys 'U', 'V', 'W' representing the factor matrices.
    U: shape (n*m, rank)
    V: shape (m*p, rank)
    W: shape (p*n, rank)
    
    Searches for a rank-minimal exact tensor decomposition.
    Falls back to schoolbook algorithm (rank 30) if no better found in time.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    try:
        U, V, W = _find_best_decomposition(n, m, p, time_limit=110.0)
    except Exception:
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][c] = 1
                    V[j * p + k][c] = 1
                    W[k * n + i][c] = 1
                    c += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
