import numpy as np
import time
from fractions import Fraction
from itertools import product

DIMENSIONS = [3, 3, 3]
DEADLINE = 120.0
SAFETY_MARGIN = 10.0

def _verify_decomposition(U, V, W, n=3, m=3, p=3, tol=1e-9):
    """
    Verify that the decomposition U, V, W correctly decomposes the 3x3 matrix multiplication tensor.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    Condition: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 for all i, j, k.
    """
    rank = len(U[0])
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                s = np.dot(U_np[u_idx], V_np[v_idx] * W_np[w_idx])
                if abs(s - 1.0) > tol:
                    return False
    return True

def _is_half_integer(x, tol=1e-9):
    return abs(x * 2 - round(x * 2)) < tol

def _check_half_integers(U, V, W, tol=1e-9):
    for M in [U, V, W]:
        for row in M:
            for x in row:
                if not _is_half_integer(x, tol):
                    return False
    return True

def _get_schoolbook(n=3, m=3, p=3):
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

def _round_to_half_ints(M, tol=1e-6):
    """Round a matrix to nearest half-integers and check if close enough."""
    M_rounded = [[round(x * 2) / 2 for x in row] for row in M]
    M_np = np.array(M, dtype=float)
    M_round_np = np.array(M_rounded, dtype=float)
    if np.max(np.abs(M_np - M_round_np)) < tol:
        return M_rounded
    return None

def _optimize_rank(r, n=3, m=3, p=3, time_limit=30.0, n_restarts=5, seed=42):
    """
    Use gradient descent to find a rank-r decomposition.
    Returns (U, V, W) as lists of lists, or None if failed.
    """
    start_time = time.time()
    np.random.seed(seed)
    
    best_loss = float('inf')
    best_solution = None
    
    for restart in range(n_restarts):
        if time.time() - start_time > time_limit:
            break
        
        # Random initialization
        U = np.random.randn(n*m, r) * 0.1
        V = np.random.randn(m*p, r) * 0.1
        W = np.random.randn(p*n, r) * 0.1
        
        # Gradient descent
        lr = 0.01
        for step in range(2000):
            if time.time() - start_time > time_limit:
                break
            
            # Compute current values
            vals = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_idx = i * m + j
                        v_idx = j * p + k
                        w_idx = k * n + i
                        vals[i, j, k] = np.dot(U[u_idx], V[v_idx] * W[w_idx])
            
            loss = np.sum((vals - 1.0) ** 2)
            
            if loss < best_loss:
                best_loss = loss
                best_solution = (U.copy(), V.copy(), W.copy())
            
            if loss < 1e-10:
                break
            
            # Compute gradients (approximate with finite differences for simplicity,
            # but this is slow. Let's use analytical gradients instead.)
            # Actually, let's use a simpler approach: coordinate descent on columns
            
            # For each column c, optimize U[:,c], V[:,c], W[:,c]
            # This is a 3-way tensor factorization problem
            
            # Use a simple approach: for each column, fix the other two and solve for the third
            for c in range(r):
                # Compute residual without column c
                res = np.ones((n, m, p))
                for c2 in range(r):
                    if c2 == c:
                        continue
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                u_idx = i * m + j
                                v_idx = j * p + k
                                w_idx = k * n + i
                                res[i, j, k] -= U[u_idx, c2] * V[v_idx, c2] * W[w_idx, c2]
                
                # Now find u_c, v_c, w_c such that sum over (i,j,k) of u_c[i*m+j] * v_c[j*p+k] * w_c[k*n+i] * res[i,j,k] is maximized
                # This is equivalent to finding the best rank-1 approximation to res
                # We can use a simple power iteration approach
                
                # Initialize
                u_c = U[:, c].copy()
                v_c = V[:, c].copy()
                w_c = W[:, c].copy()
                
                for _ in range(50):
                    # Update w_c: w_c[k*n+i] = sum_{j} u_c[i*m+j] * v_c[j*p+k] * res[i,j,k]
                    for k in range(p):
                        for i in range(n):
                            w_idx = k * n + i
                            s = 0.0
                            for j in range(m):
                                u_idx = i * m + j
                                v_idx = j * p + k
                                s += u_c[u_idx] * v_c[v_idx] * res[i, j, k]
                            w_c[w_idx] = s
                    
                    # Update u_c: u_c[i*m+j] = sum_{k} v_c[j*p+k] * w_c[k*n+i] * res[i,j,k]
                    for i in range(n):
                        for j in range(m):
                            u_idx = i * m + j
                            s = 0.0
                            for k in range(p):
                                v_idx = j * p + k
                                w_idx = k * n + i
                                s += v_c[v_idx] * w_c[w_idx] * res[i, j, k]
                            u_c[u_idx] = s
                    
                    # Update v_c: v_c[j*p+k] = sum_{i} u_c[i*m+j] * w_c[k*n+i] * res[i,j,k]
                    for j in range(m):
                        for k in range(p):
                            v_idx = j * p + k
                            s = 0.0
                            for i in range(n):
                                u_idx = i * m + j
                                w_idx = k * n + i
                                s += u_c[u_idx] * w_c[w_idx] * res[i, j, k]
                            v_c[v_idx] = s
                    
                    U[:, c] = u_c
                    V[:, c] = v_c
                    W[:, c] = w_c
        
        # After optimization, round to half-integers
        U_r = _round_to_half_ints(U.tolist())
        V_r = _round_to_half_ints(V.tolist())
        W_r = _round_to_half_ints(W.tolist())
        
        if U_r is not None and V_r is not None and W_r is not None:
            if _verify_decomposition(U_r, V_r, W_r):
                if _check_half_integers(U_r, V_r, W_r):
                    return {"U": U_r, "V": V_r, "W": W_r}
    
    return None

def _find_decomposition(n=3, m=3, p=3, max_rank=27, time_limit=100.0):
    """
    Attempt to find a rank-minimal decomposition.
    Try ranks from 23 down to 19, then fall back to schoolbook.
    """
    start_time = time.time()
    
    # Try to find solutions with decreasing rank
    for r in [23, 22, 21, 20, 19]:
        if time.time() - start_time > time_limit - 5:
            break
        
        time_for_this_rank = min(20.0, time_limit - (time.time() - start_time))
        result = _optimize_rank(r, n, m, p, time_limit=time_for_this_rank, n_restarts=3)
        
        if result is not None:
            return result
    
    # Fallback to Schoolbook
    return _get_schoolbook(n, m, p)

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Deadline guard
    deadline = start_time + DEADLINE - SAFETY_MARGIN
    
    # Try to find a better decomposition
    result = _find_decomposition(n, m, p, max_rank=27, time_limit=DEADLINE - SAFETY_MARGIN)
    
    # Safety check: ensure we didn't exceed time
    if time.time() > deadline:
        return _get_schoolbook(n, m, p)
        
    return result
