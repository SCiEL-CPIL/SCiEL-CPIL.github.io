"""Search for a rank-minimal exact tensor decomposition for multiplying a 2 by 3 matrix by a 3 by 10 matrix."""
import time
import random
import itertools
from fractions import Fraction

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

def _verify_np(U, V, W, n, m, p):
    """Fast verification using NumPy."""
    import numpy as np
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                prod = U[i*m+j] * V[j*p+k] * W[k*n+i]
                s = np.sum(prod)
                if abs(s - 1.0) > 1e-10:
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
    Try structured approaches based on known results for small matrix multiplication.
    For 2x3 * 3x10, we can try to exploit the small n=2 dimension.
    """
    import numpy as np
    rng = np.random.RandomState(seed)
    deadline = time.time() + time_budget
    
    best = None
    best_rank = n * m * p + 1
    
    # Try a few candidate ranks
    candidate_ranks = [30, 35, 40, 45, 50, 55, 60]
    candidate_ranks = [r for r in candidate_ranks if r < best_rank]
    
    for r in candidate_ranks:
        if time.time() > deadline - 10:
            break
        # Try random search at this rank
        result = _try_random_search(n, m, p, r, deadline - time.time(), seed=seed + r)
        if result is not None:
            best = result
            best_rank = r
            break
        # Also try with some zero columns to reduce effective complexity
        for trial in range(50):
            if time.time() > deadline - 10:
                break
            # Sparse random: each entry is 0 with prob 0.5, else ±1
            U = rng.randint(0, 2, size=(n*m, r)) * 2 - 1
            V = rng.randint(0, 2, size=(m*p, r)) * 2 - 1
            W = rng.randint(0, 2, size=(p*n, r)) * 2 - 1
            # Make some entries zero
            U[rng.random((n*m, r)) < 0.5] = 0
            V[rng.random((m*p, r)) < 0.5] = 0
            W[rng.random((p*n, r)) < 0.5] = 0
            
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
                        if abs(s - 1.0) > 1e-10:
                            ok = False
                            break
            if ok:
                best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                best_rank = r
                break
        if best is not None:
            break
    
    return best

def _try_random_search(n, m, p, max_rank, time_budget, seed=42):
    """Try to find a decomposition of rank <= max_rank using random search."""
    import numpy as np
    rng = np.random.RandomState(seed)
    
    best = None
    deadline = time.time() + time_budget
    
    for r in range(max_rank, n*m*p + 1):
        if time.time() > deadline - 5:
            break
        # Try random sign patterns
        attempts = 0
        max_attempts = 200
        while attempts < max_attempts and time.time() < deadline - 5:
            attempts += 1
            # Generate random ±1 matrices
            U = rng.randint(0, 2, size=(n*m, r)) * 2 - 1
            V = rng.randint(0, 2, size=(m*p, r)) * 2 - 1
            W = rng.randint(0, 2, size=(p*n, r)) * 2 - 1
            
            # Verify using the contraction
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
                        if abs(s - 1.0) > 1e-10:
                            ok = False
                            break
            if ok:
                best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                return best
    return best

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # 10s safety margin
    
    # Start with schoolbook as fallback
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try structured/random search
    time_left = deadline - time.time()
    if time_left > 5:
        result = _try_structured_decomposition(n, m, p, time_left, seed=12345)
        if result is not None:
            r = len(result["U"][0])
            if r < best_rank:
                # Verify with exact arithmetic
                if _verify(result["U"], result["V"], result["W"], n, m, p):
                    best = result
                    best_rank = r
    
    # One more attempt with different seed if time remains
    time_left = deadline - time.time()
    if time_left > 5 and best_rank == n * m * p:
        result = _try_structured_decomposition(n, m, p, time_left, seed=99999)
        if result is not None:
            r = len(result["U"][0])
            if r < best_rank:
                if _verify(result["U"], result["V"], result["W"], n, m, p):
                    best = result
                    best_rank = r
    
    return best
