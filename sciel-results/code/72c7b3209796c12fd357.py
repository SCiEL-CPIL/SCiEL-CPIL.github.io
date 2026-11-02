import numpy as np
import time
import random

DIMENSIONS = [3, 5, 6]

def _get_schoolbook():
    n, m, p = DIMENSIONS
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

def _check_decomp(n, m, p, U, V, W, r, tol=1e-9):
    """Check if U,V,W of rank r exactly decompose the matmul tensor."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += W[k*n+i, c] * U[i*m+j, c] * V[j*p+k, c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _objective(n, m, p, U, V, W, r):
    """Compute sum of squared errors for the decomposition."""
    err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += W[k*n+i, c] * U[i*m+j, c] * V[j*p+k, c]
                err += (s - 1.0) ** 2
    return err

def _solve_for_W(n, m, p, U, V, r, target=1.0):
    """Given U, V, solve for W such that sum_c W[i,k,c]*U[i,j,c]*V[j,k,c] = 1 for all (i,j,k).
    For fixed (i,k), this is: sum_c W[i,k,c] * (sum_j ... ) -- actually it's linear in W.
    For each (i,k): sum_c W[i,k,c] * U[i,j,c] * V[j,k,c] = 1 for all j.
    This is a system: for fixed (i,k), unknowns are W[i,k,0..r-1].
    Equations: for each j in 0..m-1: sum_c W[i,k,c] * U[i*m+j,c] * V[j*p+k,c] = 1.
    So it's an m x r system. If m >= r, we can solve (least squares).
    """
    W = np.zeros((p*n, r))
    for i in range(n):
        for k in range(p):
            # Build system: for each j, sum_c W[i,k,c] * U[i*m+j,c] * V[j*p+k,c] = 1
            A = np.zeros((m, r))
            b = np.ones(m)
            for j in range(m):
                for c in range(r):
                    A[j, c] = U[i*m+j, c] * V[j*p+k, c]
            # Solve A @ w = b for w (r unknowns)
            if m >= r:
                w, residuals, rank_sol, sv = np.linalg.lstsq(A, b, rcond=None)
                W[k*n+i, :] = w
            else:
                # Underdetermined: use least squares (minimum norm)
                w, residuals, rank_sol, sv = np.linalg.lstsq(A, b, rcond=None)
                W[k*n+i, :] = w
    return W

def _solve_for_U(n, m, p, V, W, r, target=1.0):
    """Given V, W, solve for U.
    For fixed (j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i.
    Unknowns: U[i*m+j, 0..r-1] for all i.
    For fixed j: equations for each (i,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1.
    That's n*p equations, r*n unknowns (U[i*m+j,c] for i in 0..n-1, c in 0..r-1).
    Actually, let's think of it differently. For fixed j, the unknown vector is 
    U[:, j] reshaped as (n*r) vector: [U[0*m+j,0], U[0*m+j,1], ..., U[0*m+j,r-1], U[1*m+j,0], ...]
    Equation (i,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    This is linear in U. Let's build the system.
    """
    U = np.zeros((n*m, r))
    for j in range(m):
        # Unknowns: U[i*m+j, c] for i in 0..n-1, c in 0..r-1 -> n*r unknowns
        # Equations: for each (i,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
        A = np.zeros((n*p, n*r))
        b = np.ones(n*p)
        eq_idx = 0
        for i in range(n):
            for k in range(p):
                for c in range(r):
                    A[eq_idx, i*r + c] = V[j*p+k, c] * W[k*n+i, c]
                eq_idx += 1
        # Solve
        w, residuals, rank_sol, sv = np.linalg.lstsq(A, b, rcond=None)
        for i in range(n):
            U[i*m+j, :] = w[i*r:(i+1)*r]
    return U

def _solve_for_V(n, m, p, U, W, r, target=1.0):
    """Given U, W, solve for V.
    For fixed (i,k): sum_c W[k*n+i,c] * U[i*m+j,c] * V[j*p+k,c] = 1 for all j.
    Unknowns: V[j*p+k, 0..r-1] for all j.
    For fixed k: equations for each (i,j): sum_c W[k*n+i,c] * U[i*m+j,c] * V[j*p+k,c] = 1.
    That's n*m equations, m*r unknowns.
    """
    V = np.zeros((m*p, r))
    for k in range(p):
        # Unknowns: V[j*p+k, c] for j in 0..m-1, c in 0..r-1 -> m*r unknowns
        # Equations: for each (i,j): sum_c W[k*n+i,c] * U[i*m+j,c] * V[j*p+k,c] = 1
        A = np.zeros((n*m, m*r))
        b = np.ones(n*m)
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for c in range(r):
                    A[eq_idx, j*r + c] = W[k*n+i, c] * U[i*m+j, c]
                eq_idx += 1
        # Solve
        w, residuals, rank_sol, sv = np.linalg.lstsq(A, b, rcond=None)
        for j in range(m):
            V[j*p+k, :] = w[j*r:(j+1)*r]
    return V

def _als_iterate(n, m, p, U, V, W, r, max_iters=200, tol=1e-10):
    """Alternating least squares: fix two, solve for the third, iterate."""
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    prev_obj = _objective(n, m, p, U, V, W, r)
    
    for it in range(max_iters):
        # Solve for W given U, V
        W = _solve_for_W(n, m, p, U, V, r)
        # Solve for U given V, W
        U = _solve_for_U(n, m, p, V, W, r)
        # Solve for V given U, W
        V = _solve_for_V(n, m, p, U, W, r)
        
        obj = _objective(n, m, p, U, V, W, r)
        if abs(prev_obj - obj) < tol:
            break
        prev_obj = obj
    
    return U, V, W, obj

def _try_rank(n, m, p, r, deadline, num_restarts=30):
    """Try to find a rank-r decomposition via ALS with random restarts."""
    best_obj = float('inf')
    best_UVW = None
    
    for restart in range(num_restarts):
        if time.time() > deadline - 5:
            break
        
        # Random initialization with half-integer-like values
        scale = random.choice([0.5, 1.0, 1.5, 2.0])
        U = np.random.randn(n*m, r) * scale
        V = np.random.randn(m*p, r) * scale
        W = np.random.randn(p*n, r) * scale
        
        # Run ALS
        U, V, W, obj = _als_iterate(n, m, p, U, V, W, r, max_iters=100, tol=1e-11)
        
        if obj < best_obj:
            best_obj = obj
            best_UVW = (U.copy(), V.copy(), W.copy())
        
        # Check if we found an exact solution
        if obj < 1e-8:
            # Round to nearest half-integer and verify
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            if _check_decomp(n, m, p, U_rounded, V_rounded, W_rounded, r, tol=1e-9):
                return U_rounded, V_rounded, W_rounded, True
            # Also try without rounding
            if _check_decomp(n, m, p, U, V, W, r, tol=1e-9):
                return U, V, W, True
    
    return best_UVW[0], best_UVW[1], best_UVW[2], False if best_obj > 1e-8 else True

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin
    
    n, m, p = DIMENSIONS
    
    # Schoolbook baseline (always valid, rank = n*m*p = 90)
    schoolbook = _get_schoolbook()
    best_rank = n * m * p
    best_result = schoolbook
    
    # Try to find lower-rank decompositions
    # Known lower bounds suggest rank >= 30 (max of n*m, m*p, p*n)
    # But actual lower bounds are tighter. Let's try a range.
    
    # Prioritize ranks that are more likely to have solutions
    # Try from lower to higher, but skip obviously impossible ranks
    ranks_to_try = list(range(25, 90))
    
    for r in ranks_to_try:
        if time.time() > deadline - 15:
            break
        
        # Number of restarts based on remaining time
        remaining = deadline - time.time() - 5
        if remaining < 2:
            break
        
        num_restarts = max(10, int(remaining * 2))
        num_restarts = min(num_restarts, 50)
        
        U, V, W, found = _try_rank(n, m, p, r, deadline, num_restarts)
        
        if found:
            if r < best_rank:
                best_rank = r
                best_result = {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
            break  # Found a valid decomposition at this rank, stop
    
    return best_result
