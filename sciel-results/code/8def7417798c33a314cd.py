import time
import numpy as np
import itertools

DIMENSIONS = [3, 4, 6]

def _snap_to_half_integers(mat):
    """Snap values to nearest half-integer and return the snapped matrix."""
    return np.round(mat * 2) / 2

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Check if the decomposition is exact."""
    # T[i*m+j, j*p+k, k*n+i] = 1
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    error = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                error += (val - 1.0) ** 2
    return np.sqrt(error) < tol

def _try_rank_search(n, m, p, rank, num_restarts=50, timeout=30):
    """
    Attempt to find a rank decomposition using alternating least squares.
    """
    start_time = time.time()
    
    # Pre-calculate indices for speed
    # U indices: i*m + j
    # V indices: j*p + k
    # W indices: k*n + i
    
    best_error = float('inf')
    best_U, best_V, best_W = None, None, None
    
    for restart in range(num_restarts):
        if time.time() - start_time > timeout:
            break
            
        # Initialize with random integers in {-1, 0, 1} or small range
        # This often helps avoid scaling issues and singularity
        U = np.random.randint(-1, 2, size=(n * m, rank)).astype(float)
        V = np.random.randint(-1, 2, size=(m * p, rank)).astype(float)
        W = np.random.randint(-1, 2, size=(p * n, rank)).astype(float)
        
        # Ensure some non-zero entries
        if np.all(U == 0): U[0,0] = 1
        if np.all(V == 0): V[0,0] = 1
        if np.all(W == 0): W[0,0] = 1
        
        for iteration in range(100):
            if time.time() - start_time > timeout:
                break
            
            # Update U
            # For each i, j, solve A U_ij^T = 1
            # A has shape (p, rank) where A[k, c] = V[j*p+k, c] * W[k*n+i, c]
            for i in range(n):
                for j in range(m):
                    A = np.zeros((p, rank))
                    for k in range(p):
                        A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                    b = np.ones(p)
                    # Solve least squares
                    try:
                        U[i * m + j, :], residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
                    except np.linalg.LinAlgError:
                        pass
            
            # Update V
            # For each j, k, solve A V_jk^T = 1
            # A has shape (n, rank) where A[i, c] = U[i*m+j, c] * W[k*n+i, c]
            for j in range(m):
                for k in range(p):
                    A = np.zeros((n, rank))
                    for i in range(n):
                        A[i, :] = U[i * m + j, :] * W[k * n + i, :]
                    b = np.ones(n)
                    try:
                        V[j * p + k, :], residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
                    except np.linalg.LinAlgError:
                        pass
            
            # Update W
            # For each k, i, solve A W_ki^T = 1
            # A has shape (m, rank) where A[j, c] = U[i*m+j, c] * V[j*p+k, c]
            for k in range(p):
                for i in range(n):
                    A = np.zeros((m, rank))
                    for j in range(m):
                        A[j, :] = U[i * m + j, :] * V[j * p + k, :]
                    b = np.ones(m)
                    try:
                        W[k * n + i, :], residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
                    except np.linalg.LinAlgError:
                        pass
            
            # Check error
            error = _verify_decomposition(U, V, W, n, m, p, tol=1e-10)
            if error:
                # If error is effectively 0, snap and return
                U_snap = _snap_to_half_integers(U)
                V_snap = _snap_to_half_integers(V)
                W_snap = _snap_to_half_integers(W)
                if _verify_decomposition(U_snap, V_snap, W_snap, n, m, p, tol=1e-9):
                    return U_snap, V_snap, W_snap
            else:
                if error < best_error:
                    best_error = error
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    
                if error < 1e-10:
                    U_snap = _snap_to_half_integers(U)
                    V_snap = _snap_to_half_integers(V)
                    W_snap = _snap_to_half_integers(W)
                    if _verify_decomposition(U_snap, V_snap, W_snap, n, m, p, tol=1e-9):
                        return U_snap, V_snap, W_snap

    return None

def _construct_schoolbook(n, m, p):
    """Fallback schoolbook decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin
    
    # Try ranks from low to high
    # Lower bound is max(n*m, m*p, p*n) = max(12, 24, 18) = 24
    # Let's start from 24
    min_rank = max(n * m, m * p, p * n)
    schoolbook_rank = n * m * p
    
    for r in range(min_rank, schoolbook_rank + 1):
        if time.time() > deadline - 5:
            break
        
        # Allocate more time for lower ranks
        time_budget = 20 if r < 30 else 10
        
        result = _try_rank_search(n, m, p, r, num_restarts=50, timeout=time_budget)
        if result is not None:
            U, V, W = result
            U_list = [[float(x) for x in row] for row in U]
            V_list = [[float(x) for x in row] for row in V]
            W_list = [[float(x) for x in row] for row in W]
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback to schoolbook
    U, V, W = _construct_schoolbook(n, m, p)
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    return {"U": U_list, "V": V_list, "W": W_list}
