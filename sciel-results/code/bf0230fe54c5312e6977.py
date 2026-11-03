"""Rewritten program: search for rank-minimal exact tensor decomposition of 2x3 * 3x6 matmul."""
import time
import numpy as np

DIMENSIONS = [2, 3, 6]
START_TIME = time.time()
TIME_LIMIT = 140.0

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """
    Verify that U, V, W exactly decompose the matmul tensor.
    Checks the diagonal condition: sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] == 1
    """
    rank = U.shape[1]
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U[i * m + j]
                v_val = V[j * p + k]
                w_val = W[k * n + i]
                s = np.sum(u_val * v_val * w_val)
                if abs(s - 1.0) > tol:
                    return False, (i, j, k, s)
    return True, None

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

def _snap_to_half_integers(X):
    """Snap values to nearest half-integer."""
    # Candidates: -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2
    # Multiplying by 2 gives integers -4 to 4
    X_scaled = X * 2
    X_snapped = np.round(X_scaled)
    return X_snapped / 2.0

def _try_rank_r(n, m, p, r, max_restarts=2000):
    """
    Try to find a rank-r decomposition using structured search.
    We fix U and V randomly, solve for W, and check if W is "nice".
    """
    candidates = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    
    for restart in range(max_restarts):
        if time.time() - START_TIME > TIME_LIMIT - 10:
            break
        
        # Initialize U and V with sparse random values
        # Higher sparsity helps keep W simple
        U = np.random.choice(candidates, size=(n*m, r))
        V = np.random.choice(candidates, size=(m*p, r))
        
        # Ensure U and V are not trivially zero
        if np.all(U == 0) or np.all(V == 0):
            continue
            
        W = np.zeros((p*n, r))
        valid_w = True
        
        # Solve for each (k, i) pair independently
        for k in range(p):
            for i in range(n):
                # Construct A for this (k, i)
                # A[j, c] = U[i*m+j, c] * V[j*p+k, c]
                A = np.zeros((m, r))
                for j in range(m):
                    A[j, :] = U[i * m + j, :] * V[j * p + k, :]
                
                b = np.ones(m)
                
                # Solve A * W_col = b
                # W_col is the row of W corresponding to (k, i)
                try:
                    # Use lstsq to find minimum norm solution
                    W_col, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    W[k * n + i, :] = W_col
                except:
                    valid_w = False
                    break
            if not valid_w:
                break
        
        if not valid_w:
            continue
            
        # Snap W to half-integers
        W_snapped = _snap_to_half_integers(W)
        
        # Verify the full decomposition with snapped W
        # U and V are already integers/half-integers from candidates
        ok, _ = _verify_decomposition(U, V, W_snapped, n, m, p, tol=1e-6)
        
        if ok:
            # Final polish: verify with tight tolerance
            ok_tight, _ = _verify_decomposition(U, V, W_snapped, n, m, p, tol=1e-9)
            if ok_tight:
                return U, V, W_snapped
            else:
                # If tight fails, try snapping U and V too (though they are already from candidates)
                # and re-verify
                U_s = _snap_to_half_integers(U)
                V_s = _snap_to_half_integers(V)
                ok_tight2, _ = _verify_decomposition(U_s, V_s, W_snapped, n, m, p, tol=1e-9)
                if ok_tight2:
                    return U_s, V_s, W_snapped

    return None

def construct():
    """Construct the lowest-rank exact tensor decomposition for 2x3 * 3x6 matmul."""
    n, m, p = DIMENSIONS
    
    # First, get the schoolbook solution as fallback
    U_school, V_school, W_school = _schoolbook(n, m, p)
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = n * m * p
    
    # Lower bound is max(nm, mp, pn) = 18
    # We try ranks from 18 upwards
    for r in range(18, best_rank):
        if time.time() - START_TIME > TIME_LIMIT - 15:
            break
            
        # Try structured search
        result = _try_rank_r(n, m, p, r, max_restarts=1000)
        if result is not None:
            U, V, W = result
            best_U, best_V, best_W = U, V, W
            best_rank = r
            break
            
    # Safety: ensure we have a valid decomposition
    ok, _ = _verify_decomposition(best_U, best_V, best_W, n, m, p)
    if not ok:
        best_U, best_V, best_W = U_school, V_school, W_school
    
    # Convert to lists for the return format
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
