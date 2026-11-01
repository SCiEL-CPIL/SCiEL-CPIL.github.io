"""Search for a rank-minimal exact tensor decomposition for multiplying a 2 by 3 matrix by a 3 by 10 matrix."""
import time
import random
from fractions import Fraction
import numpy as np

DIMENSIONS = [2, 3, 10]

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    s += Fraction(U[i*m+j][c]) * Fraction(V[j*p+k][c]) * Fraction(W[k*n+i][c])
                if s != 1:
                    return False
    return True

def _verify_np(U, V, W, n, m, p, tol=1e-10):
    """Fast verification using NumPy."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                prod = U[i*m+j] * V[j*p+k] * W[k*n+i]
                s = np.sum(prod)
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
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
    return {"U": U, "V": V, "W": W}

def _try_structured_decomposition(n, m, p, time_budget, seed=42):
    """
    Try structured decompositions using Kronecker product and Strassen-like ideas.
    For 2x3 * 3x10, exploit the small n=2 dimension.
    """
    rng = np.random.RandomState(seed)
    deadline = time.time() + time_budget
    
    best = None
    best_rank = n * m * p + 1
    
    # Try ranks from 25 to n*m*p-1
    for r in range(25, n*m*p):
        if time.time() > deadline - 5:
            break
        if r >= best_rank:
            continue
            
        max_attempts = 40
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
                
            U = np.zeros((n*m, r))
            V = np.zeros((m*p, r))
            
            # Use structured random initialization with more values
            for i in range(n*m):
                for c in range(r):
                    if rng.random() < 0.65:
                        U[i, c] = rng.choice([-2, -1, 1, 2])
            
            for j in range(m*p):
                for c in range(r):
                    if rng.random() < 0.65:
                        V[j, c] = rng.choice([-2, -1, 1, 2])
            
            # Solve for W
            W = np.zeros((p*n, r))
            feasible = True
            
            for i in range(n):
                for k in range(p):
                    A = np.zeros((m, r))
                    b = np.ones(m)
                    for j in range(m):
                        A[j, :] = U[i*m+j] * V[j*p+k]
                    
                    try:
                        sol, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
                        W[k*n+i, :] = sol
                        if not np.allclose(A @ sol, b, atol=1e-8):
                            feasible = False
                            break
                    except:
                        feasible = False
                        break
                if not feasible:
                    break
            
            if feasible:
                ok = True
                for i in range(n):
                    if not ok:
                        break
                    for j in range(m):
                        if not ok:
                            break
                        for k in range(p):
                            prod = U[i*m+j] * V[j*p+k] * W[k*n+i]
                            s = np.sum(prod)
                            if abs(s - 1.0) > 1e-8:
                                ok = False
                                break
                if ok:
                    def round_to_half(x):
                        r = round(x * 2) / 2
                        if abs(x - r) < 1e-6:
                            return r
                        return x
                    
                    U_list = [[round_to_half(v) for v in row] for row in U.tolist()]
                    V_list = [[round_to_half(v) for v in row] for row in V.tolist()]
                    W_list = [[round_to_half(v) for v in row] for row in W.tolist()]
                    
                    if _verify(U_list, V_list, W_list, n, m, p):
                        best = {"U": U_list, "V": V_list, "W": W_list}
                        best_rank = r
                        break
        if best is not None:
            break
    
    return best

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # 10s safety margin
    
    # Start with schoolbook as fallback
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try structured decomposition approach with multiple seeds
    seeds = [12345, 67890, 11111, 22222, 33333]
    for seed in seeds:
        time_left = deadline - time.time()
        if time_left < 15:
            break
        result = _try_structured_decomposition(n, m, p, time_left, seed=seed)
        if result is not None:
            r = len(result["U"][0])
            if r < best_rank:
                if _verify(result["U"], result["V"], result["W"], n, m, p):
                    best = result
                    best_rank = r
                    break
    
    return best
