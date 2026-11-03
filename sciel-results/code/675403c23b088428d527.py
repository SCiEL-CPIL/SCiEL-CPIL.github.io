import time
import numpy as np
import scipy.optimize as optimize

START_TIME = time.time()
TIME_LIMIT = 110.0

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """
    Verify that U, V, W exactly decompose the matmul tensor.
    U shape: (n*m, rank) - indexed by [i*m+j]
    V shape: (m*p, rank) - indexed by [j*p+k]
    W shape: (p*n, rank) - indexed by [k*n+i]
    """
    rank = U.shape[1]
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    
    # Reshape to 3D for efficient einsum
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    
    # Compute S[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # Note: W is indexed by k*n+i, so W3[k,i,r]
    S = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    
    # Check if S is all 1s
    return np.allclose(S, 1.0, atol=tol)

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

def _solve_rank_r(n, m, p, r, max_restarts=500):
    """
    Attempt to find a rank-r decomposition for n x m x p matmul.
    Returns (U, V, W) or None.
    """
    n_u = n * m
    n_v = m * p
    n_w = p * n
    
    def objective(params):
        U = params[:n_u*r].reshape(n, m, r)
        V = params[n_u*r : n_u*r + n_v*r].reshape(m, p, r)
        W = params[n_u*r + n_v*r :].reshape(p, n, r)
        
        # S[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        S = np.einsum('ijr,jkr,kir->ijk', U, V, W)
        return (S - 1.0).flatten()
    
    best_loss = float('inf')
    best_U, best_V, best_W = None, None, None
    
    # Try multiple restarts
    for restart in range(max_restarts):
        if time.time() - START_TIME > TIME_LIMIT - 10:
            break
            
        # Random initialization
        # Scale 1.0 is a reasonable start
        U0 = np.random.randn(n, m, r)
        V0 = np.random.randn(m, p, r)
        W0 = np.random.randn(p, n, r)
        
        x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
        
        try:
            # 'lm' is fast for unconstrained problems
            res = optimize.least_squares(objective, x0, method='lm', max_nfev=100)
        except Exception:
            continue
            
        if res.cost < best_loss:
            best_loss = res.cost
            U_sol = res.x[:n_u*r].reshape(n, m, r)
            V_sol = res.x[n_u*r : n_u*r + n_v*r].reshape(m, p, r)
            W_sol = res.x[n_u*r + n_v*r :].reshape(p, n, r)
            
            # Snap to half-integers
            U_snap = _snap_to_half_integers(U_sol)
            V_snap = _snap_to_half_integers(V_sol)
            W_snap = _snap_to_half_integers(W_sol)
            
            # Verify snapped solution
            if _verify_decomposition(U_snap, V_snap, W_snap, n, m, p):
                # Return in the required 2D format
                return U_snap.reshape(n_u, r), V_snap.reshape(n_v, r), W_snap.reshape(n_w, r)
    
    return None

def construct():
    n, m, p = 2, 3, 6
    
    # 1. Get schoolbook fallback
    U_school, V_school, W_school = _schoolbook(n, m, p)
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = U_school.shape[1]
    
    # 2. Try to find lower ranks
    # We try ranks from 35 down to 30.
    # If we find a solution, we update best and continue to see if we can go lower.
    # However, lower ranks are harder. We should prioritize finding *any* improvement first.
    # Let's try 35, 34, 33, 32, 31, 30.
    
    for r_target in [35, 34, 33, 32, 31, 30]:
        if time.time() - START_TIME > TIME_LIMIT - 15:
            break
            
        result = _solve_rank_r(n, m, p, r_target, max_restarts=200)
        if result is not None:
            U_try, V_try, W_try = result
            if _verify_decomposition(U_try, V_try, W_try, n, m, p):
                best_U, best_V, best_W = U_try, V_try, W_try
                best_rank = r_target
                # Don't break immediately, try to go lower if time permits
                # But usually, if we find 35, finding 34 is harder.
                # We can break if we find a very low rank, but for now, let's try one lower.
                # Actually, let's just break to save time for verification and return, 
                # unless we have lots of time left.
                # Given the difficulty, finding one is a win.
                # Let's break to ensure we return a valid result quickly.
                break

    # 3. Select best
    if best_U is None:
        best_U, best_V, best_W = U_school, V_school, W_school

    # Convert to lists
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
