import time
import random
import numpy as np

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START

def _build_schoolbook(n, m, p):
    """Return the standard (schoolbook) rank-n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W


def _verify(U, V, W, n, m, p, tol=1e-8):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                if abs(s - 1.0) > tol:
                    return False
    return True


def _round_to_half(arr):
    """Round each entry to the nearest half-integer."""
    return np.round(arr * 2.0) / 2.0


def _als_update_U(U, V, W, n, m, p):
    """
    Update U using Alternating Least Squares.
    For each (i, j), we solve for U[i*m+j, :] given V and W.
    The constraint is: for all k, sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1.
    Let Z be a p x rank matrix where Z[k, c] = V[j*p+k, c] * W[k*n+i, c].
    We want U_row such that Z @ U_row = 1_vector.
    Min norm solution: U_row = (Z^T Z)^-1 Z^T 1 = Z^T (Z Z^T)^-1 1.
    """
    rank = U.shape[1]
    U_new = np.zeros_like(U)
    ones = np.ones(p)
    
    for i in range(n):
        for j in range(m):
            # Construct Z (p x rank)
            Z = np.zeros((p, rank))
            for k in range(p):
                Z[k, :] = V[j * p + k, :] * W[k * n + i, :]
            
            # Solve Z @ u = ones
            # If rank > p, we have more variables than equations.
            # Min norm solution: u = Z^T (Z Z^T)^-1 ones
            # If rank == p, we can invert Z.
            # If rank < p, it might be inconsistent, but we assume rank >= p for feasibility.
            
            try:
                ZZt = Z @ Z.T
                # Add small regularization for stability
                ZZt += 1e-9 * np.eye(p)
                u = Z.T @ np.linalg.solve(ZZt, ones)
            except np.linalg.LinAlgError:
                # Fallback to pseudo-inverse
                u = np.linalg.lstsq(Z, ones, rcond=None)[0]
            
            U_new[i * m + j, :] = u
            
    return U_new


def _als_update_V(U, V, W, n, m, p):
    """
    Update V using ALS.
    For each (j, k), we solve for V[j*p+k, :] given U and W.
    Constraint: for all i, sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1.
    Let Z be an n x rank matrix where Z[i, c] = U[i*m+j, c] * W[k*n+i, c].
    We want V_row such that Z @ V_row = 1_vector.
    """
    rank = V.shape[1]
    V_new = np.zeros_like(V)
    ones = np.ones(n)
    
    for j in range(m):
        for k in range(p):
            Z = np.zeros((n, rank))
            for i in range(n):
                Z[i, :] = U[i * m + j, :] * W[k * n + i, :]
            
            try:
                ZZt = Z @ Z.T
                ZZt += 1e-9 * np.eye(n)
                v = Z.T @ np.linalg.solve(ZZt, ones)
            except np.linalg.LinAlgError:
                v = np.linalg.lstsq(Z, ones, rcond=None)[0]
            
            V_new[j * p + k, :] = v
            
    return V_new


def _als_update_W(U, V, W, n, m, p):
    """
    Update W using ALS.
    For each (k, i), we solve for W[k*n+i, :] given U and V.
    Constraint: for all j, sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1.
    Let Z be an m x rank matrix where Z[j, c] = U[i*m+j, c] * V[j*p+k, c].
    We want W_row such that Z @ W_row = 1_vector.
    """
    rank = W.shape[1]
    W_new = np.zeros_like(W)
    ones = np.ones(m)
    
    for k in range(p):
        for i in range(n):
            Z = np.zeros((m, rank))
            for j in range(m):
                Z[j, :] = U[i * m + j, :] * V[j * p + k, :]
            
            try:
                ZZt = Z @ Z.T
                ZZt += 1e-9 * np.eye(m)
                w = Z.T @ np.linalg.solve(ZZt, ones)
            except np.linalg.LinAlgError:
                w = np.linalg.lstsq(Z, ones, rcond=None)[0]
            
            W_new[k * n + i, :] = w
            
    return W_new


def _find_decomp_vectorized(n, m, p, max_rank, time_limit=10, n_trials=50):
    """
    Try to find a decomposition of rank <= max_rank for M_{n,m,p} using ALS.
    Returns (U, V, W) or (None, None, None).
    """
    start_time = time.time()
    lb = max(n * m, m * p, p * n)
    
    # Try ranks from lb to max_rank
    for rank in range(lb, max_rank + 1):
        if time.time() - start_time > time_limit:
            break
            
        for trial in range(n_trials):
            if time.time() - start_time > time_limit:
                break
                
            # Random initialization
            U = np.random.randn(n * m, rank) * 0.1
            V = np.random.randn(m * p, rank) * 0.1
            W = np.random.randn(p * n, rank) * 0.1
            
            best_error = float('inf')
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            for it in range(100):
                if time.time() - start_time > time_limit:
                    break
                    
                # ALS steps
                U = _als_update_U(U, V, W, n, m, p)
                V = _als_update_V(U, V, W, n, m, p)
                W = _als_update_W(U, V, W, n, m, p)
                
                # Calculate error
                error = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                            error += (s - 1.0) ** 2
                
                if error < best_error:
                    best_error = error
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    
                if error < 1e-10:
                    break
            
            # Check if we found a solution
            if best_error < 1e-8:
                U_r = _round_to_half(best_U)
                V_r = _round_to_half(best_V)
                W_r = _round_to_half(best_W)
                
                if _verify(U_r, V_r, W_r, n, m, p):
                    return U_r, V_r, W_r
            else:
                # Try rounding the best solution found
                U_r = _round_to_half(best_U)
                V_r = _round_to_half(best_V)
                W_r = _round_to_half(best_W)
                
                if _verify(U_r, V_r, W_r, n, m, p):
                    return U_r, V_r, W_r
    
    return None, None, None


def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix by a 3x8 matrix.
    """
    n, m, p = DIMENSIONS  # 3, 3, 8
    deadline = time.time() + 140
    
    # Split p=8 into 3, 3, 2
    p1, p2, p3 = 3, 3, 2
    
    # Find decompositions for sub-problems
    # M_{3,3,3} - known to have rank 23
    U1, V1, W1 = _find_decomp_vectorized(n, m, p1, 27, time_limit=25, n_trials=30)
    if U1 is None:
        U1, V1, W1 = _build_schoolbook(n, m, p1)
        rank1 = n * m * p1
    else:
        rank1 = U1.shape[1]
        
    # M_{3,3,3} (second part)
    U2, V2, W2 = _find_decomp_vectorized(n, m, p2, 27, time_limit=25, n_trials=30)
    if U2 is None:
        U2, V2, W2 = _build_schoolbook(n, m, p2)
        rank2 = n * m * p2
    else:
        rank2 = U2.shape[1]
        
    # M_{3,3,2}
    U3, V3, W3 = _find_decomp_vectorized(n, m, p3, 18, time_limit=20, n_trials=30)
    if U3 is None:
        U3, V3, W3 = _build_schoolbook(n, m, p3)
        rank3 = n * m * p3
    else:
        rank3 = U3.shape[1]
        
    total_rank = rank1 + rank2 + rank3
    
    # Combine
    U = np.hstack([U1, U2, U3])
    
    # V: shape (m*p, total_rank)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)
    
    # Fill V
    for j in range(m):
        for k in range(p1):
            V[j * p + k, :rank1] = V1[j * p1 + k, :]
        for k in range(p2):
            V[j * p + (p1 + k), rank1:rank1+rank2] = V2[j * p2 + k, :]
        for k in range(p3):
            V[j * p + (p1 + p2 + k), rank1+rank2:] = V3[j * p3 + k, :]
            
    # Fill W
    for i in range(n):
        for k in range(p1):
            W[k * n + i, :rank1] = W1[k * n + i, :]
        for k in range(p2):
            W[(p1 + k) * n + i, rank1:rank1+rank2] = W2[k * n + i, :]
        for k in range(p3):
            W[(p1 + p2 + k) * n + i, rank1+rank2:] = W3[k * n + i, :]
            
    # Verify combined solution
    if _verify(U, V, W, n, m, p):
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    else:
        # Fallback to schoolbook
        U0, V0, W0 = _build_schoolbook(n, m, p)
        return {
            "U": U0.tolist(),
            "V": V0.tolist(),
            "W": W0.tolist()
        }

# EVOLVE-BLOCK-END
