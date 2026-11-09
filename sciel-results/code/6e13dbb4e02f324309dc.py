import numpy as np
import time

DIMENSIONS = [3, 4, 5]
DEADLINE = 140.0

def _schoolbook():
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

def _verify(U, V, W, n, m, p):
    rank = len(W[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U_arr[i*m+j], V_arr[j*p+k] * W_arr[k*n+i])
                if abs(val - 1.0) > 1e-6:
                    return False
    return True

def _solve_W(U, V, n, m, p):
    """Given U (n*m x r) and V (m*p x r), solve for W (p*n x r) via least squares.
    For each (k,i) pair, solve A_{k,i} @ w = 1 where A[j,c] = U[i*m+j,c]*V[j*p+k,c].
    Returns W or None if any system is inconsistent."""
    r = U.shape[1]
    W = np.zeros((p * n, r))
    for ki in range(p * n):
        k = ki // n
        i = ki % n
        A = np.zeros((m, r))
        for j in range(m):
            A[j, :] = U[i*m + j, :] * V[j*p + k, :]
        if r >= m:
            w, residuals, rank_a, sv = np.linalg.lstsq(A, np.ones(m), rcond=None)
            if len(residuals) > 0 and residuals[0] > 1e-6:
                return None
            W[ki, :] = w
        else:
            return None
    return W

def _local_optimize(U, V, W, n, m, p, max_iters=500, tol=1e-8):
    """Gauss-Seidel style local optimization: alternate updating U, V, W.
    Each update minimizes the residual for the corresponding factor while fixing others."""
    U = U.copy()
    V = V.copy()
    W = W.copy()
    r = U.shape[1]
    
    for it in range(max_iters):
        # Update W: for each (k,i), solve A_{k,i} @ w = 1
        W_new = np.zeros_like(W)
        max_res = 0.0
        for ki in range(p * n):
            k = ki // n
            i = ki % n
            A = np.zeros((m, r))
            for j in range(m):
                A[j, :] = U[i*m + j, :] * V[j*p + k, :]
            w, residuals, rank_a, sv = np.linalg.lstsq(A, np.ones(m), rcond=None)
            if len(residuals) > 0:
                max_res = max(max_res, residuals[0])
            W_new[ki, :] = w
        W = W_new
        
        # Update U: for each (i,j), solve the system over all k
        # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
        # For fixed (i,j): sum_k [V[j*p+k,:] * W[k*n+i,:]]^T @ u = 1
        # This is a system in u (r variables) with p equations
        U_new = np.zeros_like(U)
        for i in range(n):
            for j in range(m):
                A = np.zeros((p, r))
                for k in range(p):
                    A[k, :] = V[j*p + k, :] * W[k*n + i, :]
                u, residuals, rank_a, sv = np.linalg.lstsq(A, np.ones(p), rcond=None)
                if len(residuals) > 0:
                    max_res = max(max_res, residuals[0])
                U_new[i*m + j, :] = u
        U = U_new
        
        # Update V: for each (j,k), solve over all i
        # For fixed (j,k): sum_i [U[i*m+j,:] * W[k*n+i,:]]^T @ v = 1
        V_new = np.zeros_like(V)
        for j in range(m):
            for k in range(p):
                A = np.zeros((n, r))
                for i in range(n):
                    A[i, :] = U[i*m + j, :] * W[k*n + i, :]
                v, residuals, rank_a, sv = np.linalg.lstsq(A, np.ones(n), rcond=None)
                if len(residuals) > 0:
                    max_res = max(max_res, residuals[0])
                V_new[j*p + k, :] = v
        V = V_new
        
        if max_res < tol:
            break
    
    return U, V, W, max_res

def _try_rank_structured(r, n, m, p, time_limit=20.0, seed=0, max_attempts=30):
    """Try to find a decomposition of rank r using structured random init + local optimization."""
    rng = np.random.RandomState(seed)
    start = time.time()
    num_ij = n * m
    num_jk = m * p
    num_ki = p * n
    
    for attempt in range(max_attempts):
        if time.time() - start > time_limit:
            break
        
        U = rng.randn(num_ij, r)
        V = rng.randn(num_jk, r)
        
        # Solve for W
        W = _solve_W(U, V, n, m, p)
        if W is None:
            continue
        
        # Local optimization
        U, V, W, max_res = _local_optimize(U, V, W, n, m, p, max_iters=200, tol=1e-10)
        
        if max_res < 1e-6:
            if _verify(W.tolist(), V.tolist(), U.tolist(), n, m, p):
                return U.tolist(), V.tolist(), W.tolist()
    
    return None

def _try_rank_direct(r, n, m, p, time_limit=15.0, seed=0):
    """Direct random search: fix U,V, solve W, verify."""
    rng = np.random.RandomState(seed)
    start = time.time()
    num_ij = n * m
    num_jk = m * p
    num_ki = p * n
    
    for attempt in range(100):
        if time.time() - start > time_limit:
            break
        
        U = rng.randn(num_ij, r)
        V = rng.randn(num_jk, r)
        
        W = _solve_W(U, V, n, m, p)
        if W is None:
            continue
        
        if _verify(W.tolist(), V.tolist(), U.tolist(), n, m, p):
            return U.tolist(), V.tolist(), W.tolist()
    
    return None

def _split_p_search(n, m, p, time_limit=30.0):
    """Split p into p1+p2, solve subproblems independently, combine."""
    start = time.time()
    best_rank = n * m * p
    best_decomp = None
    
    for p1 in range(1, p):
        p2 = p - p1
        time_per_sub = (time_limit - (time.time() - start)) / 2.0
        if time_per_sub < 2.0:
            break
        
        # Subproblem 1: n x m times m x p1
        rank1 = n * m * p1
        decomp1 = None
        for r1 in range(n*m*p1, max(n*m, m*p1, n*p1) - 1, -1):
            result = _try_rank_structured(r1, n, m, p1, time_limit=time_per_sub/3.0, seed=100+p1+r1)
            if result is not None:
                U1, V1, W1 = result
                decomp1 = (U1, V1, W1)
                rank1 = r1
                break
        
        if decomp1 is None:
            # Fall back to schoolbook for subproblem 1
            U1 = np.zeros((n*m, rank1))
            V1 = np.zeros((m*p1, rank1))
            W1 = np.zeros((p1*n, rank1))
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p1):
                        U1[i*m+j, col] = 1
                        V1[j*p1+k, col] = 1
                        W1[k*n+i, col] = 1
                        col += 1
            decomp1 = (U1, V1, W1)
        
        # Subproblem 2: n x m times m x p2
        rank2 = n * m * p2
        decomp2 = None
        for r2 in range(n*m*p2, max(n*m, m*p2, n*p2) - 1, -1):
            result = _try_rank_structured(r2, n, m, p2, time_limit=time_per_sub/3.0, seed=200+p2+r2)
            if result is not None:
                U2, V2, W2 = result
                decomp2 = (U2, V2, W2)
                rank2 = r2
                break
        
        if decomp2 is None:
            U2 = np.zeros((n*m, rank2))
            V2 = np.zeros((m*p2, rank2))
            W2 = np.zeros((p2*n, rank2))
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p2):
                        U2[i*m+j, col] = 1
                        V2[j*p2+k, col] = 1
                        W2[k*n+i, col] = 1
                        col += 1
            decomp2 = (U2, V2, W2)
        
        total_rank = rank1 + rank2
        if total_rank < best_rank:
            U = np.zeros((n*m, total_rank))
            V = np.zeros((m*p, total_rank))
            W = np.zeros((p*n, total_rank))
            
            U[:, :rank1] = decomp1[0]
            V[:m*p1, :rank1] = decomp1[1]
            W[:p1*n, :rank1] = decomp1[2]
            
            U[:, rank1:] = decomp2[0]
            V[m*p1:, rank1:] = decomp2[1]
            W[p1*n:, rank1:] = decomp2[2]
            
            best_rank = total_rank
            best_decomp = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        if time.time() - start > time_limit:
            break
    
    return best_decomp, best_rank

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Start with schoolbook as fallback
    best = _schoolbook()
    best_rank = n * m * p
    
    # Phase 1: Split-p structured search
    decomp, rank = _split_p_search(n, m, p, time_limit=35.0)
    if decomp is not None and rank < best_rank:
        best = decomp
        best_rank = rank
    
    # Phase 2: Direct structured search for lower ranks
    elapsed = time.time() - start_time
    remaining = DEADLINE - elapsed
    
    if remaining > 20.0:
        for r in range(best_rank - 1, 19, -1):
            if time.time() - start_time > DEADLINE - 15:
                break
            
            time_per_rank = min(12.0, (DEADLINE - 15 - (time.time() - start_time)) / 5.0)
            
            result = _try_rank_structured(r, n, m, p, time_limit=time_per_rank, seed=r*7+1)
            if result is not None:
                U, V, W = result
                best = {"U": U, "V": V, "W": W}
                best_rank = r
                break
    
    # Final safety check
    if time.time() - start_time > DEADLINE - 5:
        return _schoolbook()
    
    return best
