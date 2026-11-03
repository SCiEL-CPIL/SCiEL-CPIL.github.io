"""Combined program: search for rank-minimal exact tensor decomposition of 2x3 * 3x6 matmul."""
import time
import numpy as np
import scipy.optimize as optimize

DIMENSIONS = [2, 3, 6]
START_TIME = time.time()
TIME_LIMIT = 110.0  # safety margin before 120s kill

# EVOLVE-BLOCK-START

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """
    Verify that U, V, W exactly decompose the matmul tensor.
    U shape: (n*m, rank) - indexed by [i*m+j]
    V shape: (m*p, rank) - indexed by [j*p+k]
    W shape: (p*n, rank) - indexed by [k*n+i]
    
    For exact decomposition: for all i,j,k:
    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U[i * m + j]
                v_val = V[j * p + k]
                w_val = W[k * n + i]
                s = np.sum(u_val * v_val * w_val)
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _snap_to_half_integers(matrix):
    """Snap matrix entries to nearest 0.5."""
    return np.round(matrix * 2) / 2.0

def _snap_to_integers(matrix):
    """Snap matrix entries to nearest integer."""
    return np.round(matrix)

def _try_snaps(U, V, W, n, m, p, tol=1e-9):
    """Try multiple snapping strategies and return the first valid one."""
    # Try half-integer snap
    U_h = _snap_to_half_integers(U)
    V_h = _snap_to_half_integers(V)
    W_h = _snap_to_half_integers(W)
    if _verify_decomposition(U_h, V_h, W_h, n, m, p, tol):
        return U_h, V_h, W_h
    
    # Try integer snap
    U_i = _snap_to_integers(U)
    V_i = _snap_to_integers(V)
    W_i = _snap_to_integers(W)
    if _verify_decomposition(U_i, V_i, W_i, n, m, p, tol):
        return U_i, V_i, W_i
    
    # Try quarter-integer snap
    U_q = np.round(U * 4) / 4.0
    V_q = np.round(V * 4) / 4.0
    W_q = np.round(W * 4) / 4.0
    if _verify_decomposition(U_q, V_q, W_q, n, m, p, tol):
        return U_q, V_q, W_q
    
    return None

def _solve_rank_r(n, m, p, r, max_restarts=100, time_budget=15.0):
    """
    Try to find a rank-r decomposition for n x m x p matmul tensor.
    Uses scipy.optimize.least_squares on the sum of squares of residuals.
    """
    n_eq = n * m * p
    n_u = n * m
    n_v = m * p
    n_w = p * n
    
    def objective(params):
        U = params[:n_u*r].reshape(n_u, r)
        V = params[n_u*r : n_u*r + n_v*r].reshape(n_v, r)
        W = params[n_u*r + n_v*r :].reshape(n_w, r)
        
        residuals = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                    residuals.append(s - 1.0)
        return residuals
    
    best_U, best_V, best_W = None, None, None
    best_loss = float('inf')
    
    for restart in range(max_restarts):
        if time.time() - START_TIME > time_budget:
            break
        
        # Random initialization with varying scales
        scale = 1.0 if restart % 2 == 0 else 0.5
        U0 = np.random.randn(n_u, r) * scale
        V0 = np.random.randn(n_v, r) * scale
        W0 = np.random.randn(n_w, r) * scale
        
        x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
        
        try:
            # Use 'trf' to handle underdetermined systems (variables > residuals)
            res = optimize.least_squares(objective, x0, method='trf', max_nfev=300)
        except Exception:
            continue
        
        if res.cost < best_loss:
            best_loss = res.cost
            U_sol = res.x[:n_u*r].reshape(n_u, r)
            V_sol = res.x[n_u*r : n_u*r + n_v*r].reshape(n_v, r)
            W_sol = res.x[n_u*r + n_v*r :].reshape(n_w, r)
            
            # Try snapping
            snapped = _try_snaps(U_sol, V_sol, W_sol, n, m, p)
            if snapped is not None:
                return snapped
            
            # Check if unsnapped is effectively exact
            if _verify_decomposition(U_sol, V_sol, W_sol, n, m, p, tol=1e-8):
                # Try to refine with higher precision
                x_init = res.x.copy()
                try:
                    res2 = optimize.least_squares(objective, x_init, method='trf', max_nfev=500, ftol=1e-12, xtol=1e-12)
                    U_ref = res2.x[:n_u*r].reshape(n_u, r)
                    V_ref = res2.x[n_u*r : n_u*r + n_v*r].reshape(n_v, r)
                    W_ref = res2.x[n_u*r + n_v*r :].reshape(n_w, r)
                    snapped = _try_snaps(U_ref, V_ref, W_ref, n, m, p)
                    if snapped is not None:
                        return snapped
                except Exception:
                    pass
    
    return None

def _expand_2x3x3_to_2x3x6(U_sub, V_sub, W_sub):
    """
    Expand a rank-r 2x3x3 decomposition to a rank-2r 2x3x6 decomposition.
    """
    r_sub = U_sub.shape[1]
    r_full = 2 * r_sub
    
    U_full = np.zeros((6, r_full))
    U_full[:, :r_sub] = U_sub
    U_full[:, r_sub:] = U_sub
    
    V_full = np.zeros((18, r_full))
    W_full = np.zeros((12, r_full))
    
    for j in range(3):
        for k in range(6):
            if k < 3:
                V_full[j*6 + k, :r_sub] = V_sub[j*3 + k, :]
            else:
                V_full[j*6 + k, r_sub:] = V_sub[j*3 + (k-3), :]
                
    for k in range(6):
        for i in range(2):
            if k < 3:
                W_full[k*2 + i, :r_sub] = W_sub[k*2 + i, :]
            else:
                W_full[k*2 + i, r_sub:] = W_sub[(k-3)*2 + i, :]
                
    return U_full, V_full, W_full

def _try_block_decomposition(n, m, p):
    """
    Try to decompose 2x3x6 as two 2x3x3 blocks.
    Rank(2x3x6) = 2 * Rank(2x3x3).
    """
    n_sub, m_sub, p_sub = 2, 3, 3
    
    # Try ranks for 2x3x3: known rank is 18 (schoolbook), try 17, 16, ...
    for r_sub in [17, 16, 15, 14]:
        if time.time() - START_TIME > 60.0:
            break
        
        result = _solve_rank_r(n_sub, m_sub, p_sub, r_sub, max_restarts=80, time_budget=15.0)
        if result is not None:
            U_sub, V_sub, W_sub = result
            U_full, V_full, W_full = _expand_2x3x3_to_2x3x6(U_sub, V_sub, W_sub)
            
            ok = _verify_decomposition(U_full, V_full, W_full, n, m, p)
            if ok:
                return U_full, V_full, W_full
    
    return None

def _try_direct_rank_search(n, m, p, max_rank=35, time_budget=30.0):
    """
    Try to find a direct rank-r decomposition for the full problem.
    Search from lower ranks upward.
    """
    for r in range(20, max_rank + 1):
        if time.time() - START_TIME > time_budget:
            break
        
        result = _solve_rank_r(n, m, p, r, max_restarts=50, time_budget=5.0)
        if result is not None:
            U, V, W = result
            ok = _verify_decomposition(U, V, W, n, m, p)
            if ok:
                return U, V, W
    
    return None

def construct():
    """Construct the lowest-rank exact tensor decomposition for 2x3 * 3x6 matmul."""
    n, m, p = DIMENSIONS
    
    # Fallback to schoolbook
    U, V, W = _schoolbook(n, m, p)
    if not _verify_decomposition(U, V, W, n, m, p):
        raise ValueError("Schoolbook decomposition failed verification")
    
    best_rank = U.shape[1]
    best_U, best_V, best_W = U, V, W
    
    # Attempt to find lower rank decompositions
    # Time budget: 100 seconds (leaving 10s for safety)
    search_deadline = START_TIME + 100.0
    
    # 1. Try direct rank search from low to high
    # Schoolbook is 36. Try 20 to 35.
    # We can try a few specific ranks or a range.
    # Given the complexity, we might not find anything, but we must try.
    if time.time() < search_deadline:
        result = _try_direct_rank_search(n, m, p, max_rank=35, time_budget=60.0)
        if result is not None:
            U, V, W = result
            if _verify_decomposition(U, V, W, n, m, p):
                best_rank = U.shape[1]
                best_U, best_V, best_W = U, V, W
    
    # 2. Try block decomposition (2 x 3 x 3 blocks)
    if time.time() < search_deadline:
        result = _try_block_decomposition(n, m, p)
        if result is not None:
            U, V, W = result
            if _verify_decomposition(U, V, W, n, m, p):
                if U.shape[1] < best_rank:
                    best_rank = U.shape[1]
                    best_U, best_V, best_W = U, V, W
    
    U, V, W = best_U, best_V, best_W
    
    # Convert to lists for the return format
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
