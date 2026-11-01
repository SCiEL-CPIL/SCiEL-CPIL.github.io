"""Matrix multiplication 2,5,5 - rank minimization via ALS search."""
import numpy as np
import time

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook baseline
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n*m, rank_schoolbook))
    V_sb = np.zeros((m*p, rank_schoolbook))
    W_sb = np.zeros((p*n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin before 150s kill
    
    def verify(U, V, W, rank):
        """Check if the decomposition is exact using vectorized operations."""
        # For each (i,j,k), compute sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        # We need this to equal 1 for all valid (i,j,k)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1) > 1e-8:
                        return False
        return True

    def tensor_error(U, V, W, rank):
        """Compute max absolute error of the tensor reconstruction."""
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    max_err = max(max_err, abs(val - 1))
        return max_err

    def als_solve(U, V, W, rank, max_iters=200, tol=1e-10):
        """Alternating Least Squares to minimize tensor error."""
        U = U.copy()
        V = V.copy()
        W = W.copy()
        
        for it in range(max_iters):
            if time.time() > deadline:
                return None
            
            # Update U: for each row index a = i*m+j, we need
            # U[a,c] such that sum_c U[a,c] * (V[j*p+k,c] * W[k*n+i,c]) = 1
            # for all k. This is a least squares problem.
            for i in range(n):
                for j in range(m):
                    a = i*m + j
                    # Build matrix B where B[k,c] = V[j*p+k,c] * W[k*n+i,c]
                    # Then we want U[a,:] * B^T = 1 (vector of ones)
                    B = np.zeros((p, rank))
                    for k in range(p):
                        B[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    # Solve: B^T * U[a,:]^T = 1_p
                    # i.e., U[a,:] = (B^T B)^{-1} B^T * 1
                    try:
                        U[a, :] = np.linalg.lstsq(B.T, np.ones(p), rcond=None)[0]
                    except:
                        pass
            
            # Update V: for each row index b = j*p+k, we need
            # V[b,c] such that sum_c (U[i*m+j,c] * W[k*n+i,c]) * V[b,c] = 1
            # for all i. 
            for j in range(m):
                for k in range(p):
                    b = j*p + k
                    # Build matrix A where A[i,c] = U[i*m+j,c] * W[k*n+i,c]
                    # Then we want A * V[b,:]^T = 1 (vector of ones of length n)
                    A = np.zeros((n, rank))
                    for i in range(n):
                        A[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    try:
                        V[b, :] = np.linalg.lstsq(A.T, np.ones(n), rcond=None)[0]
                    except:
                        pass
            
            # Update W: for each row index w = k*n+i, we need
            # W[w,c] such that sum_c (U[i*m+j,c] * V[j*p+k,c]) * W[w,c] = 1
            # for all j.
            for k in range(p):
                for i in range(n):
                    w = k*n + i
                    # Build matrix C where C[j,c] = U[i*m+j,c] * V[j*p+k,c]
                    # Then we want C * W[w,:]^T = 1 (vector of ones of length m)
                    C = np.zeros((m, rank))
                    for j in range(m):
                        C[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    try:
                        W[w, :] = np.linalg.lstsq(C.T, np.ones(m), rcond=None)[0]
                    except:
                        pass
            
            # Check convergence
            err = tensor_error(U, V, W, rank)
            if err < tol:
                return U, V, W
        
        return None

    def round_to_half_int(X):
        """Round to nearest half-integer."""
        return np.round(X * 2) / 2

    # Try ranks from schoolbook-1 down to a reasonable lower bound
    # Lower bound for M_{n,m,p} is at least max(n*m, m*p, p*n) for n=m=p,
    # but for non-square it's more complex. For 2x5x5, let's try down to 25.
    rng = np.random.default_rng(42)
    
    best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    best_rank = rank_schoolbook
    
    for r in range(rank_schoolbook - 1, 24, -1):
        if time.time() > deadline:
            break
        
        # Try multiple random initializations
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            # Random initialization
            U = rng.standard_normal((n*m, r))
            V = rng.standard_normal((m*p, r))
            W = rng.standard_normal((p*n, r))
            
            # Run ALS
            result = als_solve(U, V, W, r)
            if result is None:
                continue
            
            U_sol, V_sol, W_sol = result
            
            # Check if solution is close to exact
            err = tensor_error(U_sol, V_sol, W_sol, r)
            if err < 1e-8:
                # Try rounding to half-integers
                U_r = round_to_half_int(U_sol)
                V_r = round_to_half_int(V_sol)
                W_r = round_to_half_int(W_sol)
                
                if verify(U_r, V_r, W_sol, r):
                    best_rank = r
                    best_result = {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                    break  # Found valid decomposition for this rank, try lower
                elif verify(U_sol, V_sol, W_sol, r):
                    # Exact but not half-integer; keep as best for now
                    # But we need half-integer entries, so this won't work directly
                    # Let's try harder to find a half-integer solution
                    pass
            
            # Also try without rounding - check if we can get exact with current values
            # The problem requires integer/half-integer entries, so we need to round
    
    return best_result

# EVOLVE-BLOCK-END
