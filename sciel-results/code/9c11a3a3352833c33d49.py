"""Find a rank-minimal exact tensor decomposition for 2x4 * 4x5 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 4, 5]

def _compute_error(U, V, W, n, m, p):
    """Compute the maximum error of the decomposition."""
    rank = U.shape[1]
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    return max_err

def _als_solve_U(V, W, n, m, p, r):
    """Solve for U given V and W using least squares."""
    U = np.zeros((n*m, r))
    for i in range(n):
        for j in range(m):
            # For fixed (i,j), we need: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all k
            # This is: U[i*m+j,:] @ diag(V[j*p+k,:]*W[k*n+i,:]) = 1 for each k
            # More precisely: for each k, U[i*m+j,:] @ (V[j*p+k,:] * W[k*n+i,:]) = 1
            # So we have a system: M @ U[i*m+j,:]^T = 1 where M[k,c] = V[j*p+k,c]*W[k*n+i,c]
            M = np.zeros((p, r))
            for k in range(p):
                M[k, :] = V[j*p+k, :] * W[k*n+i, :]
            # Solve M @ x = ones(p) for x = U[i*m+j,:]^T
            try:
                U[i*m+j, :] = np.linalg.lstsq(M, np.ones(p), rcond=None)[0]
            except np.linalg.LinAlgError:
                U[i*m+j, :] = np.zeros(r)
    return U

def _als_solve_V(U, W, n, m, p, r):
    """Solve for V given U and W using least squares."""
    V = np.zeros((m*p, r))
    for j in range(m):
        for k in range(p):
            # For fixed (j,k), we need: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i
            # M[i,c] = U[i*m+j,c] * W[k*n+i,c]
            M = np.zeros((n, r))
            for i in range(n):
                M[i, :] = U[i*m+j, :] * W[k*n+i, :]
            try:
                V[j*p+k, :] = np.linalg.lstsq(M, np.ones(n), rcond=None)[0]
            except np.linalg.LinAlgError:
                V[j*p+k, :] = np.zeros(r)
    return V

def _als_solve_W(U, V, n, m, p, r):
    """Solve for W given U and V using least squares."""
    W = np.zeros((p*n, r))
    for k in range(p):
        for i in range(n):
            # For fixed (k,i), we need: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all j
            # M[j,c] = U[i*m+j,c] * V[j*p+k,c]
            M = np.zeros((m, r))
            for j in range(m):
                M[j, :] = U[i*m+j, :] * V[j*p+k, :]
            try:
                W[k*n+i, :] = np.linalg.lstsq(M, np.ones(m), rcond=None)[0]
            except np.linalg.LinAlgError:
                W[k*n+i, :] = np.zeros(r)
    return W

def _try_rank(r, n, m, p, deadline, seed=42, max_restarts=15, max_als_iters=300):
    """Try to find a rank-r decomposition using ALS with random restarts."""
    rng = np.random.RandomState(seed)
    
    best_result = None
    best_err = float('inf')
    
    for attempt in range(max_restarts):
        if time.time() > deadline - 5:
            break
        
        # Random initialization with small values to avoid numerical issues
        scale = 1.0 / np.sqrt(r)
        U = rng.randn(n*m, r) * scale
        V = rng.randn(m*p, r) * scale
        W = rng.randn(p*n, r) * scale
        
        local_best_err = float('inf')
        local_best_U, local_best_V, local_best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_als_iters):
            if time.time() > deadline - 5:
                break
            
            # ALS: alternate solving for each matrix
            U_new = _als_solve_U(V, W, n, m, p, r)
            V_new = _als_solve_V(U_new, W, n, m, p, r)
            W_new = _als_solve_W(U_new, V_new, n, m, p, r)
            
            U, V, W = U_new, V_new, W_new
            
            err = _compute_error(U, V, W, n, m, p)
            if err < local_best_err:
                local_best_err = err
                local_best_U, local_best_V, local_best_W = U.copy(), V.copy(), W.copy()
            
            # Check convergence
            if local_best_err < 1e-9:
                break
        
        if local_best_err < best_err:
            best_err = local_best_err
            best_result = (local_best_U, local_best_V, local_best_W)
        
        # If we found a good solution, return it
        if best_err < 1e-7:
            U_list = [[float(x) for x in row] for row in best_result[0]]
            V_list = [[float(x) for x in row] for row in best_result[1]]
            W_list = [[float(x) for x in row] for row in best_result[2]]
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # If we didn't find a good solution but have a best result, check it
    if best_result is not None and best_err < 1e-6:
        U_list = [[float(x) for x in row] for row in best_result[0]]
        V_list = [[float(x) for x in row] for row in best_result[1]]
        W_list = [[float(x) for x in row] for row in best_result[2]]
        return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def construct_schoolbook():
    """Return the schoolbook decomposition (rank 40)."""
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

def construct():
    """Main entry point: try to find a lower-rank decomposition, fall back to schoolbook."""
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin
    
    # Lower bound: rank >= max(n*m, m*p, p*n) = max(8, 20, 10) = 20
    # Try ranks from 20 upward
    for r in range(20, 40):
        if time.time() > deadline - 10:
            break
        result = _try_rank(r, n, m, p, deadline, seed=100+r, max_restarts=10, max_als_iters=200)
        if result is not None:
            return result
    
    # Fallback to schoolbook
    return construct_schoolbook()
