"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def _round_to_half(x):
    """Round to nearest half-integer (integer or k+0.5)."""
    return np.round(2.0 * x) / 2.0

def _verify(U, V, W, n, m, p, tol=1e-8):
    """Check if U, V, W exactly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    rank = U.shape[1]
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # For all (i,j,k), T should be 1 if this is a valid factorization of matmul
    # Actually, the tensor for matmul is T[i,j,k] = delta (it's 1 for all i,j,k in the 
    # sense that C[i,k] = sum_j A[i,j]*B[j,k])
    # The decomposition is: C[i,k] = sum_c (sum_j U[i*m+j,c]*A[i,j]) * (sum_j V[j*p+k,c]*B[j,k]) * W[k*n+i,c]
    # Wait, let me reconsider. The standard form is:
    # C[i,k] = sum_c U[i*m+j,c]... no.
    # 
    # The tensor T has indices (i, j, k) where i in [0,n), j in [0,m), k in [0,p)
    # and T[i,j,k] = 1 for all valid (i,j,k) because A[i,j]*B[j,k] contributes to C[i,k].
    # 
    # The decomposition: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # 
    # So we need: for all i,j,k: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    # And for invalid combinations (which don't exist in this setup), it should be 0.
    # But actually all (i,j,k) with i<n, j<m, k<p are valid, so all should equal 1.
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    return max_err < tol

def _check_half_integer(U, V, W, tol=1e-10):
    """Check if all entries are half-integers (integers or k+0.5)."""
    for mat in [U, V, W]:
        mat = np.array(mat, dtype=np.float64)
        doubled = 2.0 * mat
        rounded = np.round(doubled)
        if np.max(np.abs(doubled - rounded)) > tol:
            return False
    return True

def _newton_solve(U, V, W, n, m, p, max_iter=50, tol=1e-12):
    """
    Newton-Raphson to solve: for all i,j,k:
        sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    
    Variables: all entries of U, V, W (rank * (n*m + m*p + p*n) unknowns)
    Equations: n*m*p equations
    """
    rank = U.shape[1]
    U = U.astype(np.float64).copy()
    V = V.astype(np.float64).copy()
    W = W.astype(np.float64).copy()
    
    nU = n * m
    nV = m * p
    nW = p * n
    nvar = rank * (nU + nV + nW)
    neq = n * m * p
    
    # Flatten variables
    def get_vars():
        return np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    
    def set_vars(v):
        U[:] = v[:rank*nU].reshape(nU, rank)
        V[:] = v[rank*nU:rank*(nU+nV)].reshape(nV, rank)
        W[:] = v[rank*(nU+nV):].reshape(nW, rank)
    
    def residual(v):
        """Compute residual: for each (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] - 1"""
        set_vars(v)
        res = np.zeros(neq)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    res[idx] = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :]) - 1.0
                    idx += 1
        return res
    
    def jacobian(v):
        """Compute Jacobian of residual w.r.t. variables."""
        set_vars(v)
        J = np.zeros((neq, nvar))
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # d/dU[i*m+j,c] of sum_c' U[i*m+j,c']*V[j*p+k,c']*W[k*n+i,c'] 
                    # = V[j*p+k,c] * W[k*n+i,c] for c'=c
                    J[idx, 0:rank*nU:1] = 0  # will fill properly
                    for c in range(rank):
                        # U entry (i*m+j, c)
                        col_U = i*m+j
                        J[idx, c] = V[j*p+k, c] * W[k*n+i, c]
                        # V entry (j*p+k, c)
                        col_V = j*p+k
                        J[idx, rank*nU + c] = U[col_U, c] * W[k*n+i, c]
                        # W entry (k*n+i, c)
                        col_W = k*n+i
                        J[idx, rank*(nU+nV) + c] = U[col_U, c] * V[col_V, c]
                    idx += 1
        return J
    
    v = get_vars()
    for it in range(max_iter):
        r = residual(v)
        norm_r = np.linalg.norm(r)
        if norm_r < tol:
            break
        J = jacobian(v)
        try:
            # Solve J * dv = -r
            dv = np.linalg.lstsq(J, -r, rcond=None)[0]
        except np.linalg.LinAlgError:
            # Use damped update
            dv = -0.01 * r[:nvar] if nvar <= neq else -0.01 * np.resize(r, nvar)
        v_new = v + dv
        # Check for NaN/Inf
        if not np.all(np.isfinite(v_new)):
            break
        v = v_new
    
    set_vars(v)
    return U, V, W

def _try_rank(rank, n, m, p, num_restarts=5, seed_base=42):
    """Try to find a valid decomposition of given rank."""
    np.random.seed(seed_base)
    U_init = np.random.randn(n*m, rank) * 0.5
    V_init = np.random.randn(m*p, rank) * 0.5
    W_init = np.random.randn(p*n, rank) * 0.5
    
    best_U, best_V, best_W = None, None, None
    best_err = float('inf')
    
    for restart in range(num_restarts):
        np.random.seed(seed_base + restart)
        U_r = np.random.randn(n*m, rank) * 0.5
        V_r = np.random.randn(m*p, rank) * 0.5
        W_r = np.random.randn(p*n, rank) * 0.5
        
        U_sol, V_sol, W_sol = _newton_solve(U_r, V_r, W_r, n, m, p, max_iter=30, tol=1e-12)
        
        # Round to half-integers
        U_round = _round_to_half(U_sol)
        V_round = _round_to_half(V_sol)
        W_round = _round_to_half(W_sol)
        
        # Verify
        if _verify(U_round, V_round, W, n, m, p, tol=1e-8):
            return U_round, V_round, W_round
        
        # Also try without rounding to see if close
        err = 0.0
        U_arr = np.array(U_sol)
        V_arr = np.array(V_sol)
        W_arr = np.array(W_sol)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U_arr[i*m+j, :] * V_arr[j*p+k, :] * W_arr[k*n+i, :])
                    err = max(err, abs(val - 1.0))
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U_sol.copy(), V_sol.copy(), W_sol.copy()
    
    return None, None, None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin
    
    # Schoolbook as fallback
    rank_school = n * m * p
    U_fb = [[0] * rank_school for _ in range(n * m)]
    V_fb = [[0] * rank_school for _ in range(m * p)]
    W_fb = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i*m+j][col] = 1
                V_fb[j*p+k][col] = 1
                W_fb[k*n+i][col] = 1
                col += 1
    
    # Try ranks from known lower bounds up to schoolbook
    # Known lower bound for matmul tensor rank is at least max(n*m, m*p, p*n)
    # For 2x3x10: lower bound is max(6, 30, 20) = 30
    # But actually the tensor rank lower bound is more subtle.
    # Let's try a range of ranks.
    
    # Theoretical lower bound: rank >= max(n*m, m*p, p*n) = 30
    # But for bilinear complexity, lower bounds are different.
    # Let's try from 30 up to 60 (schoolbook)
    
    best_result = None
    best_rank = rank_school
    
    # Try a few ranks
    ranks_to_try = list(range(30, min(rank_school + 1, 61)))
    
    for rank in ranks_to_try:
        if time.time() > deadline - 10:
            break
        if rank >= best_rank:
            break
        try:
            U_r, V_r, W_r = _try_rank(rank, n, m, p, num_restarts=3, seed_base=rank*100)
            if U_r is not None:
                best_rank = rank
                best_result = (U_r, V_r, W_r)
                # If we found one, we can stop or continue for lower
                if rank <= 30:
                    break
        except Exception:
            continue
    
    if best_result is not None:
        U_r, V_r, W_r = best_result
        # Convert to lists of lists
        U_list = [[float(U_r[i][j]) for j in range(best_rank)] for i in range(n*m)]
        V_list = [[float(V_r[i][j]) for j in range(best_rank)] for i in range(m*p)]
        W_list = [[float(W_r[i][j]) for j in range(best_rank)] for i in range(p*n)]
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback to schoolbook
    return {"U": U_fb, "V": V_fb, "W": W_fb}
# EVOLVE-BLOCK-END
