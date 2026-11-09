import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [3, 5, 7]

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly computes matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    r = U.shape[1]
    # Check shape consistency
    if U.shape[0] != n*m or V.shape[0] != m*p or W.shape[0] != p*n:
        return False
    if V.shape[1] != r or W.shape[1] != r:
        return False
    
    # For each (i,j,k), check: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                err = abs(s - 1.0)
                if err > max_err:
                    max_err = err
    return max_err < 1e-9

def _to_halfint_list(arr):
    """Convert numpy array to list of lists with half-integer values."""
    result = []
    for row in arr:
        row_list = []
        for val in row:
            # Round to nearest half-integer
            half_int = round(val * 2) / 2.0
            row_list.append(half_int)
        result.append(row_list)
    return result

def _try_rank_als(n, m, p, r, n_iters=200, n_restarts=3, deadline=None):
    """Try to find a rank-r decomposition using alternating least squares."""
    best_err = float('inf')
    best_U = None
    best_V = None
    best_W = None
    
    for restart in range(n_restarts):
        if deadline is not None and time.time() > deadline:
            break
        
        # Random initialization
        U = np.random.randn(n*m, r) * 0.5
        V = np.random.randn(m*p, r) * 0.5
        W = np.random.randn(p*n, r) * 0.5
        
        for it in range(n_iters):
            if deadline is not None and time.time() > deadline:
                break
            
            # Update U: for each row i*m+j, solve min ||target - V_row * W_col|| 
            # where target is the vector over k of 1/(V[j*p+k,c]*W[k*n+i,c])... this is tricky
            # Actually, let's do a simpler alternating approach
            
            # Fix V and W, update U
            for i in range(n):
                for j in range(m):
                    # For row (i,j) of U, we want:
                    # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all k
                    # This is a linear system in U[i*m+j,:]
                    # Let A be a (p x r) matrix where A[k,c] = V[j*p+k,c] * W[k*n+i,c]
                    # Then we want A @ U[i*m+j,:] = 1 (vector of ones)
                    A = np.zeros((p, r))
                    for k in range(p):
                        for c in range(r):
                            A[k, c] = V[j*p+k, c] * W[k*n+i, c]
                    # Solve A @ x = ones
                    try:
                        x, residuals, rank_A, sv = np.linalg.lstsq(A, np.ones(p), rcond=None)
                        U[i*m+j, :] = x
                    except:
                        pass
            
            # Fix U and W, update V
            for j in range(m):
                for k in range(p):
                    # For row (j,k) of V, we want:
                    # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i
                    # Let A be an (n x r) matrix where A[i,c] = U[i*m+j,c] * W[k*n+i,c]
                    # Then we want A @ V[j*p+k,:] = 1 (vector of ones)
                    A = np.zeros((n, r))
                    for i in range(n):
                        for c in range(r):
                            A[i, c] = U[i*m+j, c] * W[k*n+i, c]
                    try:
                        x, residuals, rank_A, sv = np.linalg.lstsq(A, np.ones(n), rcond=None)
                        V[j*p+k, :] = x
                    except:
                        pass
            
            # Fix U and V, update W
            for k in range(p):
                for i in range(n):
                    # For row (k,i) of W, we want:
                    # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all j
                    # Let A be an (m x r) matrix where A[j,c] = U[i*m+j,c] * V[j*p+k,c]
                    # Then we want A @ W[k*n+i,:] = 1 (vector of ones)
                    A = np.zeros((m, r))
                    for j in range(m):
                        for c in range(r):
                            A[j, c] = U[i*m+j, c] * V[j*p+k, c]
                    try:
                        x, residuals, rank_A, sv = np.linalg.lstsq(A, np.ones(m), rcond=None)
                        W[k*n+i, :] = x
                    except:
                        pass
            
            # Compute error
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for c in range(r):
                            s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                        err += (s - 1.0) ** 2
            err = np.sqrt(err / (n*m*p))
            
            if err < best_err:
                best_err = err
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
            
            if err < 1e-8:
                break
        
        if best_err < 1e-8:
            break
    
    if best_err < 1e-6:
        # Try to round to half-integers and verify
        U_hi = _to_halfint_list(best_U)
        V_hi = _to_halfint_list(best_V)
        W_hi = _to_halfint_list(best_W)
        if _verify_decomposition(U_hi, V_hi, W_hi, n, m, p):
            return U_hi, V_hi, W_hi
    
    return None, None, None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # 110s deadline to be safe
    
    # Schoolbook decomposition as fallback
    U_school = [[0] * (n*m*p) for _ in range(n*m)]
    V_school = [[0] * (n*m*p) for _ in range(m*p)]
    W_school = [[0] * (n*m*p) for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j][col] = 1
                V_school[j*p+k][col] = 1
                W_school[k*n+i][col] = 1
                col += 1
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = n*m*p
    
    # Try to find a better decomposition using ALS
    # Theoretical lower bound is max(n*m, m*p, n*p) = max(15, 35, 21) = 35
    # Try ranks from 35 upward
    for r in range(35, 105, 5):
        if time.time() > deadline - 10:
            break
        
        U, V, W = _try_rank_als(n, m, p, r, n_iters=100, n_restarts=2, deadline=deadline)
        if U is not None:
            best_result = {"U": U, "V": V, "W": W}
            best_rank = r
            break
    
    return best_result
