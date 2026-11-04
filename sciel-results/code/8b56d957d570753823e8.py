import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [2, 5, 5]

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1
                c += 1
    return U, V, W

def _verify_exact(n, m, p, U, V, W):
    """Verify exact decomposition using Fraction arithmetic."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    u = Fraction(U[i * m + j][c])
                    v = Fraction(V[j * p + k][c])
                    w = Fraction(W[k * n + i][c])
                    s += u * v * w
                if s != 1:
                    return False
    return True

def _verify_float(n, m, p, U, V, W, tol=1e-9):
    """Quick float verification."""
    rank = len(U[0])
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                if abs(s - 1.0) > tol:
                    return False
    return True

def _als_decompose(n, m, p, target_rank, max_restarts=100, max_iter=200, seed_offset=0):
    """
    Use Alternating Least Squares to find a rank-`target_rank` decomposition.
    Returns (U, V, W) as float arrays or None if not converged.
    """
    best_error = float('inf')
    best_params = None
    
    for restart in range(max_restarts):
        np.random.seed(10000 + seed_offset + restart)
        
        # Initialize with small random values
        U = np.random.randn(n * m, target_rank) * 0.5
        V = np.random.randn(m * p, target_rank) * 0.5
        W = np.random.randn(p * n, target_rank) * 0.5
        
        error = float('inf')
        
        for it in range(max_iter):
            # Update U
            for i in range(n):
                for j in range(m):
                    A = np.zeros((p, target_rank))
                    for k in range(p):
                        A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                    try:
                        x, _, _, _ = np.linalg.lstsq(A, np.ones(p), rcond=None)
                        U[i * m + j, :] = x
                    except np.linalg.LinAlgError:
                        pass

            # Update V
            for j in range(m):
                for k in range(p):
                    B = np.zeros((n, target_rank))
                    for i in range(n):
                        B[i, :] = U[i * m + j, :] * W[k * n + i, :]
                    try:
                        y, _, _, _ = np.linalg.lstsq(B, np.ones(n), rcond=None)
                        V[j * p + k, :] = y
                    except np.linalg.LinAlgError:
                        pass

            # Update W
            for k in range(p):
                for i in range(n):
                    C = np.zeros((m, target_rank))
                    for j in range(m):
                        C[j, :] = U[i * m + j, :] * V[j * p + k, :]
                    try:
                        z, _, _, _ = np.linalg.lstsq(C, np.ones(m), rcond=None)
                        W[k * n + i, :] = z
                    except np.linalg.LinAlgError:
                        pass
            
            # Compute error
            error = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                        error += (s - 1.0)**2
            
            if error < 1e-12:
                break
        
        if error < best_error:
            best_error = error
            best_params = (U.copy(), V.copy(), W.copy())
        
        if best_error < 1e-10:
            break
    
    if best_error < 1e-8 and best_params is not None:
        return best_params
    return None

def _round_to_rational(U, V, W, denom):
    """Round to multiples of 1/denom and verify."""
    U_r = np.round(U * denom) / denom
    V_r = np.round(V * denom) / denom
    W_r = np.round(W * denom) / denom
    
    # Convert to lists of Fractions for exact verification
    U_list = [[Fraction(int(x * denom)) / denom for x in row] for row in U_r.tolist()]
    V_list = [[Fraction(int(x * denom)) / denom for x in row] for row in V_r.tolist()]
    W_list = [[Fraction(int(x * denom)) / denom for x in row] for row in W_r.tolist()]
    
    # Quick float check first
    if not _verify_float(2, 5, 5, U_r, V_r, W_r, tol=1e-6):
        return None
    
    # Exact check
    if _verify_exact(2, 5, 5, U_list, V_list, W_list):
        # Convert back to simple numeric values
        U_out = [[float(f) for f in row] for row in U_list]
        V_out = [[float(f) for f in row] for row in V_list]
        W_out = [[float(f) for f in row] for row in W_list]
        return U_out, V_out, W_out
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Schoolbook decomposition as baseline
    U_school, V_school, W_school = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_U = U_school.tolist()
    best_V = V_school.tolist()
    best_W = W_school.tolist()

    # Known lower bound for 2x5x5 is 21
    # Try ranks from lower bound upward
    for target_rank in range(21, best_rank):
        if time.time() > deadline:
            break
        
        # Try ALS decomposition
        sol = _als_decompose(n, m, p, target_rank, max_restarts=50, max_iter=100, seed_offset=target_rank)
        
        if sol is not None:
            U_float, V_float, W_float = sol
            
            # Try rounding to integer
            res = _round_to_rational(U_float, V_float, W_float, 1)
            if res:
                best_rank = target_rank
                best_U, best_V, best_W = res
                break
            
            # Try rounding to half-integer
            res = _round_to_rational(U_float, V_float, W_float, 2)
            if res:
                best_rank = target_rank
                best_U, best_V, best_W = res
                break
            
            # Try rounding to quarter-integer
            res = _round_to_rational(U_float, V_float, W_float, 4)
            if res:
                best_rank = target_rank
                best_U, best_V, best_W = res
                break
            
            # Try rounding to eighth-integer
            res = _round_to_rational(U_float, V_float, W_float, 8)
            if res:
                best_rank = target_rank
                best_U, best_V, best_W = res
                break
        
        # Check time
        if time.time() > deadline - 5:
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}
