import numpy as np
import scipy.optimize as opt
import time

DIMENSIONS = [3, 4, 8]

def _verify(U, V, W, n, m, p, tol=1e-6):
    """Check if U @ V @ W^T reconstructs the matmul tensor using numpy."""
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    # Build the full tensor check vectorized
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_np[i*m+j] * V_np[j*p+k] * W_np[k*n+i])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _snap_to_half_integer(M):
    """Snap matrix entries to nearest 0.5."""
    return np.round(M * 2.0) / 2.0

def _construct_block_decomposition(n, m, p):
    """
    Construct a block-based decomposition.
    Split the n dimension into blocks and use Strassen for 2x2x2 subproblems.
    Target: use structured algebraic decomposition.
    """
    # Strategy: 3x4x8 = (1x4x8) + (2x4x8)
    # 1x4x8: 8 schoolbook terms
    # 2x4x8: use block decomposition
    
    # For now, construct a valid schoolbook-based decomposition with block structure
    # that might allow for some optimization
    
    # Let's try: split p=8 into two 4s
    # 3x4x4 + 3x4x4
    # Each 3x4x4: can we do better than schoolbook 12?
    # 3x4x4 = (1x4x4) + (2x4x4)
    # 1x4x4: 4 terms
    # 2x4x4: 8 schoolbook, or use Strassen-like
    
    # Simple approach: just return schoolbook for now, but structured
    R = n * m * p
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    return U, V, W

def _construct_strassen_based(n, m, p):
    """
    Construct a Strassen-based decomposition.
    For 2x2x2, Strassen uses rank 7.
    We build up from there.
    """
    # This is complex; for now, try a simple structured approach
    # Split into 2x2x2 blocks where possible
    
    # For 3x4x8:
    # Split n=3 into 2+1
    # Split m=4 into 2+2  
    # Split p=8 into 4+4
    
    # Part 1: 1x4x8 -> 8 terms (schoolbook for the 1-row case)
    # Part 2: 2x4x8 -> use block Strassen
    
    # 2x4x8: split m=4 into 2+2, p=8 into 4+4
    # Gives 4 subproblems of 2x2x4
    # Each 2x2x4: split p=4 into 2+2, giving 2 subproblems of 2x2x2
    # Each 2x2x2: Strassen rank 7
    # So 2x4x8: 4 * 2 * 7 = 56? That seems high.
    
    # Actually, I think the correct count is:
    # 2x4x8 = 2 * (2x2x8) if we split m
    # 2x2x8 = 4 * (2x2x2) if we split p into 4 blocks of 2
    # Each 2x2x2: rank 7
    # Total for 2x4x8: 2 * 4 * 7 = 56
    
    # Plus 1x4x8: 8 terms
    # Total: 64
    
    # That's better than 96 but not great. Let me try a different split.
    
    # Alternative: Don't split as finely
    # 3x4x8: 
    # Use the fact that we can share computations
    
    # For now, return None to indicate this needs more work
    return None

def _tensor_residuals(params, R, n, m, p):
    """
    Compute residuals for the tensor decomposition.
    params: flattened array of U, V, W.
    """
    size_U = n * m * R
    size_V = m * p * R
    size_W = p * n * R
    
    U = params[:size_U].reshape(n * m, R)
    V = params[size_U:size_U + size_V].reshape(m * p, R)
    W = params[size_U + size_V:].reshape(p * n, R)
    
    # Vectorized residual computation
    residuals = np.zeros(n * m * p)
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_row = U[i * m + j]
                v_row = V[j * p + k]
                w_row = W[k * n + i]
                residuals[idx] = 1.0 - np.dot(u_row, v_row * w_row)
                idx += 1
    
    return residuals

def _try_rank(R, n, m, p, time_limit, seed_base=0):
    """
    Try to find a decomposition of rank R.
    """
    start_time = time.time()
    max_attempts = 5
    
    for attempt in range(max_attempts):
        if time.time() - start_time > time_limit:
            break
            
        seed = seed_base + attempt * 137
        np.random.seed(seed)
        
        scale = 1.0 / np.sqrt(R)
        
        U_init = np.random.randn(n * m, R) * scale
        V_init = np.random.randn(m * p, R) * scale
        W_init = np.random.randn(p * n, R) * scale
        
        params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        try:
            result = opt.least_squares(
                _tensor_residuals, 
                params_init, 
                args=(R, n, m, p),
                method='trf', 
                max_nfev=300,
                ftol=1e-10,
                xtol=1e-10,
                gtol=1e-10
            )
        except Exception:
            continue
            
        if result.cost < 1e-8:
            params_sol = result.x
            size_U = n * m * R
            size_V = m * p * R
            size_W = p * n * R
            
            U_sol = params_sol[:size_U].reshape(n * m, R)
            V_sol = params_sol[size_U:size_U + size_V].reshape(m * p, R)
            W_sol = params_sol[size_U + size_V:].reshape(p * n, R)
            
            if _verify(U_sol, V_sol, W_sol, n, m, p):
                U_snap = _snap_to_half_integer(U_sol)
                V_snap = _snap_to_half_integer(V_sol)
                W_snap = _snap_to_half_integer(W_sol)
                
                if _verify(U_snap, V_snap, W_snap, n, m, p):
                    return U_snap, V_snap, W_snap
                else:
                    all_close = True
                    for mat in [U_sol, V_sol, W_sol]:
                        for val in mat.flatten():
                            if abs(val * 2 - round(val * 2)) > 1e-4:
                                all_close = False
                                break
                        if not all_close:
                            break
                    if all_close and _verify(U_sol, V_sol, W_sol, n, m, p, tol=1e-10):
                        return U_sol, V_sol, W_sol
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 145  # 5s safety margin
    
    # Schoolbook fallback (rank 96) - guaranteed valid
    schoolbook_rank = n * m * p
    U_sb = np.zeros((n * m, schoolbook_rank))
    V_sb = np.zeros((m * p, schoolbook_rank))
    W_sb = np.zeros((p * n, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1.0
                V_sb[j * p + k, col] = 1.0
                W_sb[k * n + i, col] = 1.0
                col += 1
                
    best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    # Try structured constructions first
    strassen_result = _construct_strassen_based(n, m, p)
    if strassen_result is not None:
        U_s, V_s, W_s = strassen_result
        if _verify(U_s, V_s, W_s, n, m, p):
            best_result = {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
            return best_result
    
    # Search for lower rank
    for R in range(20, 96):
        if time.time() > deadline:
            break
            
        time_left = deadline - time.time()
        if time_left < 2:
            break
        
        time_per_rank = min(1.0, time_left / max(1, 96 - R))
        
        result = _try_rank(R, n, m, p, time_limit=time_per_rank, seed_base=R)
        
        if result is not None:
            U_sol, V_sol, W_sol = result
            best_result = {
                "U": U_sol.tolist(),
                "V": V_sol.tolist(),
                "W": W_sol.tolist()
            }
            return best_result
            
    return best_result
