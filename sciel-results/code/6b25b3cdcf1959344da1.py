import time
import numpy as np

DIMENSIONS = [3, 3, 8]

def _validate(U, V, W, n, m, p, tol=1e-9):
    """Check that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    
    # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # Should equal 1 for all i,j,k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_r[i,j,:] * V_r[j,k,:] * W_r[k,i,:])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _solve_for_W(U, V, n, m, p):
    R = U.shape[1]
    W = np.zeros((p*n, R))
    for i in range(n):
        for k in range(p):
            A = np.zeros((m, R))
            b = np.ones(m)
            for j in range(m):
                A[j, :] = U[i*m+j, :] * V[j*p+k, :]
            w, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            W[k*n+i, :] = w
    return W

def _solve_for_V(U, W, n, m, p):
    R = U.shape[1]
    V = np.zeros((m*p, R))
    for j in range(m):
        for k in range(p):
            A = np.zeros((n, R))
            b = np.ones(n)
            for i in range(n):
                A[i, :] = U[i*m+j, :] * W[k*n+i, :]
            v, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            V[j*p+k, :] = v
    return V

def _solve_for_U(V, W, n, m, p):
    R = V.shape[1]
    U = np.zeros((n*m, R))
    for i in range(n):
        for j in range(m):
            A = np.zeros((p, R))
            b = np.ones(p)
            for k in range(p):
                A[k, :] = V[j*p+k, :] * W[k*n+i, :]
            u, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            U[i*m+j, :] = u
    return U

def _round_to_half(x):
    return np.round(x * 2) / 2

def _alternating_minimize(U, V, W, n, m, p, max_iter=50, tol=1e-9):
    """Run alternating minimization and return best rounded solution."""
    best_rank = U.shape[1]
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iter):
        W = _solve_for_W(U, V, n, m, p)
        V = _solve_for_V(U, W, n, m, p)
        U = _solve_for_U(V, W, n, m, p)
        
        U_r = _round_to_half(U)
        V_r = _round_to_half(V)
        W_r = _round_to_half(W)
        
        if _validate(U_r, V_r, W_r, n, m, p, tol):
            return U_r, V_r, W_r, U_r.shape[1]
    
    # Check if unrounded is close
    U_r = _round_to_half(U)
    V_r = _round_to_half(V)
    W_r = _round_to_half(W)
    if _validate(U_r, V_r, W_r, n, m, p, tol):
        return U_r, V_r, W_r, U_r.shape[1]
    
    return None

def _try_rank(R, n, m, p, n_restarts=5, deadline=None):
    """Try to find a valid decomposition of rank R."""
    for restart in range(n_restarts):
        if deadline is not None and time.time() > deadline:
            break
        np.random.seed(42 + restart)
        U = np.random.randn(n*m, R) * 0.3
        V = np.random.randn(m*p, R) * 0.3
        W = np.random.randn(p*n, R) * 0.3
        
        result = _alternating_minimize(U, V, W, n, m, p, max_iter=30)
        if result is not None:
            return result
    
    return None

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W, rank

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook as fallback
    U_school, V_school, W_school, rank_school = _schoolbook(n, m, p)
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school
    
    # Try to find better decompositions using alternating minimization
    # The theoretical lower bound for 3x3x8 is at least ceil(3*3*8 / something)
    # Known: 3x3x3 has rank 19, so 3x3x8 might be around 19*3 = 57 or less
    
    # Try ranks from low to high
    # Lower bound: the tensor has 3*3*8 = 72 non-zero entries, but they're structured
    # For 3x3x8, we can think of it as 8 copies of 3x3, sharing the A factor
    
    # Try a range of ranks
    for R in range(40, best_rank + 1, 2):
        if time.time() > deadline:
            break
        result = _try_rank(R, n, m, p, n_restarts=3, deadline=deadline)
        if result is not None:
            U_r, V_r, W_r, r = result
            best_U, best_V, best_W = U_r.tolist(), V_r.tolist(), W_r.tolist()
            best_rank = r
            break
    
    # If we found a good rank, try to reduce further
    if best_rank < rank_school:
        for R in range(max(30, best_rank - 10), best_rank, 1):
            if time.time() > deadline:
                break
            result = _try_rank(R, n, m, p, n_restarts=5, deadline=deadline)
            if result is not None:
                U_r, V_r, W_r, r = result
                best_U, best_V, best_W = U_r.tolist(), V_r.tolist(), W_r.tolist()
                best_rank = r
                break
    
    return {"U": best_U, "V": best_V, "W": best_W}
