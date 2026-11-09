"""Rewritten program for 2x4x5 matrix multiplication tensor rank minimization."""
import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, tol=1e-10):
    """Verify exact decomposition: for all i,j,k, sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1"""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(val - 1.0) > tol:
                    return False
    return True

def _try_rank(r, n, m, p, max_iters=500, n_trials=5, deadline=None):
    """Try to find rank-r decomposition via alternating minimization."""
    U = np.zeros((n*m, r))
    V = np.zeros((m*p, r))
    W = np.zeros((p*n, r))
    
    for trial in range(n_trials):
        if deadline and time.time() > deadline:
            break
        # Random initialization
        np.random.seed(42 + trial * 1000)
        U = np.random.randn(n*m, r) * 0.5
        V = np.random.randn(m*p, r) * 0.5
        W = np.random.randn(p*n, r) * 0.5
        
        for it in range(max_iters):
            if deadline and time.time() > deadline:
                break
            
            # Update W: for each (k,i), we need sum_c W[k*n+i,c] * (U[i*m+j,c]*V[j*p+k,c]) = 1 for all j
            # This is: for fixed (k,i), sum_j [sum_c W[k*n+i,c]*U[i*m+j,c]*V[j*p+k,c]] should relate
            # Actually the constraint is: for each (i,j,k): sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
            # For fixed (k,i): this must hold for all j. So sum_c W[k*n+i,c] * (sum over j? No...)
            
            # The constraint per (i,j,k) is scalar. Let's group differently.
            # For fixed (i,k): sum_j A[i,j]B[j,k] = sum_c (sum_j U[i*m+j,c]A[i,j]) * ... no that's wrong.
            
            # Let me think again. The tensor entry is:
            # T_{i,j,k} = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
            # 
            # For fixed i,k: this must equal 1 for every j. So:
            # For each j: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
            #
            # This is a system in W[k*n+i,:] for each j. But U and V also depend on j.
            # 
            # Better approach: treat this as: we need the bilinear form to be constant.
            # 
            # Alternative: use the fact that if we fix U and V, then for each (i,k),
            # the values sum_c U[i*m+j,c]*V[j*p+k,c] for varying j should be "balanced" 
            # such that W can make them all equal to 1.
            
            # Simpler: use gradient descent on the loss = sum_{i,j,k} (sum_c ... - 1)^2
            # But that's expensive. Let's use a direct algebraic approach.
            
            # For a given rank r, the conditions are:
            # For all (i,j,k): <U[i*m+j,:], V[j*p+k,:] * W[k*n+i,:]>_elementwise_sum = 1
            # where the sum is over c.
            
            # Let's define: for each (k,i), let w_ki = W[k*n+i,:]
            # For each j: sum_c U[i*m+j,c] * V[j*p+k,c] * w_ki[c] = 1
            # This is: (U[i*m+j,:] * V[j*p+k,:]) @ w_ki = 1  (element-wise product then dot)
            
            # So for fixed (k,i), we have m equations in r unknowns w_ki:
            # For j=0..m-1: (U[i*m+j,:] .* V[j*p+k,:]) @ w_ki = 1
            # This is a linear system: A_ki @ w_ki = ones(m)
            # where A_ki[j,c] = U[i*m+j,c] * V[j*p+k,c]
            
            # So: given U and V, W is determined by solving m linear systems of size r x r (or r unknowns, m equations).
            # If m < r, we need the system to be consistent.
            
            # Similarly, given V and W, we can solve for U, and given U and W, solve for V.
            
            # Let's do alternating: update W given U,V; update U given V,W; update V given U,W.
            
            # Update W:
            for k in range(p):
                for i in range(n):
                    # Build system: for each j, (U[i*m+j,:] * V[j*p+k,:]) @ w = 1
                    A = np.array([U[i*m+j, :] * V[j*p+k, :] for j in range(m)])  # shape (m, r)
                    b = np.ones(m)
                    # Solve A @ w = b (least squares if overdetermined)
                    if r >= m:
                        # Underdetermined or square: find w such that A@w = b
                        # Use lstsq
                        w, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
                        W[k*n+i, :] = w
                    else:
                        # r < m: need consistency
                        w, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
                        W[k*n+i, :] = w
            
            # Check if current solution is exact
            if _verify(U, V, W, n, m, p, tol=1e-8):
                # Try to clean up: snap to simple values
                U_clean = np.round(U, 10)
                V_clean = np.round(V, 10)
                W_clean = np.round(W, 10)
                if _verify(U_clean, V_clean, W_clean, n, m, p, tol=1e-10):
                    return U_clean, V_clean, W_clean
                # Also try snapping to halves
                U_half = np.round(U * 2) / 2
                V_half = np.round(V * 2) / 2
                W_half = np.round(W * 2) / 2
                if _verify(U_half, V_half, W_half, n, m, p, tol=1e-10):
                    return U_half, V_half, W_half
                return U, V, W
            
            # Update U: for each (i,j), solve for U[i*m+j,:] given V and W
            for i in range(n):
                for j in range(m):
                    # For each k: (V[j*p+k,:] * W[k*n+i,:]) @ u_ij = 1
                    A = np.array([V[j*p+k, :] * W[k*n+i, :] for k in range(p)])  # shape (p, r)
                    b = np.ones(p)
                    u, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
                    U[i*m+j, :] = u
            
            # Update V: for each (j,k), solve for V[j*p+k,:] given U and W
            for j in range(m):
                for k in range(p):
                    # For each i: (U[i*m+j,:] * W[k*n+i,:]) @ v_jk = 1
                    A = np.array([U[i*m+j, :] * W[k*n+i, :] for i in range(n)])  # shape (n, r)
                    b = np.ones(n)
                    v, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
                    V[j*p+k, :] = v
        
        # After iterations, check final solution
        if _verify(U, V, W, n, m, p, tol=1e-8):
            U_clean = np.round(U, 10)
            V_clean = np.round(V, 10)
            W_clean = np.round(W, 10)
            if _verify(U_clean, V_clean, W_clean, n, m, p, tol=1e-10):
                return U_clean, V_clean, W_clean
            U_half = np.round(U * 2) / 2
            V_half = np.round(V * 2) / 2
            W_half = np.round(W * 2) / 2
            if _verify(U_half, V_half, W_half, n, m, p, tol=1e-10):
                return U_half, V_half, W_half
            return U, V, W
    
    return None

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Deadline: stop 5 seconds before the 150s limit
    start_time = time.time()
    deadline = start_time + 145
    
    # Try decreasing ranks from schoolbook-1 downward
    # We know rank 40 works (schoolbook). Try 39, 38, ... down to some minimum.
    # Theoretical lower bound: for 2x4x5, the rank is at least... 
    # R(n,m,p) >= (n*m + m*p + p*n)/2 ... no that's not right.
    # A known lower bound: R(2,4,5) >= 2*4 + 4*5 + 5*2 - something... 
    # Actually, a simple bound: the slice rank or just try from 40 down.
    
    # First, establish we have the schoolbook solution
    U_sb = np.zeros((n*m, schoolbook_rank))
    V_sb = np.zeros((m*p, schoolbook_rank))
    W_sb = np.zeros((p*n, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = schoolbook_rank
    
    # Try to find better ranks
    # For each candidate rank, do a few trials
    for r in range(schoolbook_rank - 1, max(20, schoolbook_rank - 20), -1):
        if time.time() > deadline:
            break
        result = _try_rank(r, n, m, p, max_iters=200, n_trials=3, deadline=deadline)
        if result is not None:
            U_found, V_found, W_found = result
            if _verify(U_found, V_found, W_found, n, m, p, tol=1e-9):
                best_U, best_V, best_W = U_found, V_found, W_found
                best_rank = r
                break  # Found a valid lower rank, use it
    
    # Convert to lists with integer/half-integer entries if possible
    def to_clean_list(arr):
        arr = np.array(arr, dtype=float)
        # Try to snap to halves
        arr_half = np.round(arr * 2) / 2
        # Check if snapping changes values significantly
        if np.allclose(arr, arr_half, atol=1e-8):
            return [[float(v) for v in row] for row in arr_half]
        else:
            return [[float(v) for v in row] for row in arr]
    
    U_list = to_clean_list(best_U)
    V_list = to_clean_list(best_V)
    W_list = to_clean_list(best_W)
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
