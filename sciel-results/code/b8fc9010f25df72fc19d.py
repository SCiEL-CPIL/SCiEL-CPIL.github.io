"""Tensor decomposition for 2x3 * 3x6 matrix multiplication."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _schoolbook_decomposition(n, m, p):
    """Return the trivial schoolbook rank-n*m*p decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W

def _verify_decomposition(n, m, p, U, V, W):
    """
    Verify that the decomposition is exact.
    """
    rank = len(U[0]) if U else 0
    T = np.zeros((n, m, p), dtype=np.float64)
    for r in range(rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
    expected = np.ones((n, m, p))
    return np.allclose(T, expected, atol=1e-10)

def _is_half_integer(x):
    """Check if x is an integer or half-integer."""
    if isinstance(x, int):
        return True
    if isinstance(x, float):
        return abs(x - round(x)) < 1e-9 or abs(x - (round(x * 2) / 2)) < 1e-9
    return False

def _sanitize_and_verify(n, m, p, U, V, W):
    """
    Round U, V, W to nearest half-integers and verify.
    Returns sanitized lists if valid, else None.
    """
    U_rounded = [[round(x * 2) / 2 for x in row] for row in U]
    V_rounded = [[round(x * 2) / 2 for x in row] for row in V]
    W_rounded = [[round(x * 2) / 2 for x in row] for row in W]
    
    # Check if all are half-integers (they should be by construction of rounding)
    # But verify the decomposition
    if _verify_decomposition(n, m, p, U_rounded, V_rounded, W_rounded):
        return U_rounded, V_rounded, W_rounded
    return None

def _try_sparse_search(n, m, p, target_rank, deadline):
    """
    Try to find a decomposition of the given rank using alternating least squares.
    """
    nm = n * m
    mp = m * p
    pn = p * n
    
    for attempt in range(3): # Reduced attempts to save time
        if time.time() > deadline:
            break
        
        np.random.seed(42 + attempt)
        U = np.random.randn(nm, target_rank) * 0.1
        V = np.random.randn(mp, target_rank) * 0.1
        W = np.random.randn(pn, target_rank) * 0.1
        
        for iteration in range(100): # Reduced iterations
            if time.time() > deadline:
                break
            
            # Fix V and W, solve for U
            U_new = np.zeros_like(U)
            for i in range(n):
                for j in range(m):
                    A_i = np.zeros((p, target_rank))
                    for k in range(p):
                        for r in range(target_rank):
                            A_i[k, r] = V[j * p + k, r] * W[k * n + i, r]
                    try:
                        sol = np.linalg.lstsq(A_i, np.ones(p), rcond=None)[0]
                        U_new[i * m + j, :] = sol
                    except:
                        U_new[i * m + j, :] = U[i * m + j, :]
            U = U_new
            
            # Fix U and W, solve for V
            V_new = np.zeros_like(V)
            for j in range(m):
                for k in range(p):
                    A_j = np.zeros((n, target_rank))
                    for i in range(n):
                        for r in range(target_rank):
                            A_j[i, r] = U[i * m + j, r] * W[k * n + i, r]
                    try:
                        sol = np.linalg.lstsq(A_j, np.ones(n), rcond=None)[0]
                        V_new[j * p + k, :] = sol
                    except:
                        V_new[j * p + k, :] = V[j * p + k, :]
            V = V_new
            
            # Fix U and V, solve for W
            W_new = np.zeros_like(W)
            for k in range(p):
                for i in range(n):
                    A_k = np.zeros((m, target_rank))
                    for j in range(m):
                        for r in range(target_rank):
                            A_k[j, r] = U[i * m + j, r] * V[j * p + k, r]
                    try:
                        sol = np.linalg.lstsq(A_k, np.ones(m), rcond=None)[0]
                        W_new[k * n + i, :] = sol
                    except:
                        W_new[k * n + i, :] = W[k * n + i, :]
            W = W_new
            
            # Check convergence and sanitize
            if iteration % 20 == 0:
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(target_rank))
                            err += abs(val - 1.0)
                if err < 1e-6:
                    sanitized = _sanitize_and_verify(n, m, p, U.tolist(), V.tolist(), W.tolist())
                    if sanitized:
                        return sanitized
    
    return None

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 2x3 * 3x6 matrix multiplication.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100  # Safety margin
    
    # First, get the schoolbook decomposition as fallback
    U_school, V_school, W_school = _schoolbook_decomposition(n, m, p)
    
    # Try to find a lower-rank decomposition
    # Lower bound 18, try up to 35
    for target_rank in range(max(n*m, m*p, p*n), n*m*p):
        if time.time() > deadline:
            break
        
        result = _try_sparse_search(n, m, p, target_rank, deadline)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    # Fall back to schoolbook
    return {"U": U_school, "V": V_school, "W": W_school}
# EVOLVE-BLOCK-END
