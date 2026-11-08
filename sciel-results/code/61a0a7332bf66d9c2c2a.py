"""Combined approach: structured initialization + refined optimization for 2x3x6 matrix mult."""
import time
import random
import math
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 6]
_START = time.time()
_DEADLINE = 120.0
_MARGIN = 15.0

def _remaining():
    return _DEADLINE - _MARGIN - (time.time() - _START)

def _residual(U, V, W, n, m, p):
    """Return max absolute tensor residual over all (i,j,k)."""
    worst = 0.0
    r = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                uc = U[i * m + j]
                vc = V[j * p + k]
                wc = W[k * n + i]
                for c in range(r):
                    s += uc[c] * vc[c] * wc[c]
                diff = abs(s - 1.0)
                if diff > worst:
                    worst = diff
    return worst

def _try_round_verify(U, V, W, tol=1e-9):
    """Round to half-integers and check exactness."""
    Ur = [[round(v * 2) / 2 for v in row] for row in U]
    Vr = [[round(v * 2) / 2 for v in row] for row in V]
    Wr = [[round(v * 2) / 2 for v in row] for row in W]
    n, m, p = DIMENSIONS
    r = len(Ur[0]) if Ur else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += Ur[i * m + j][c] * Vr[j * p + k][c] * Wr[k * n + i][c]
                if abs(s - 1.0) > tol:
                    return None
    return {"U": Ur, "V": Vr, "W": Wr}

def _refine(U, V, W, n, m, p, iters=50):
    """Gauss-Seidel style refinement to drive residual toward zero."""
    r = len(U[0]) if U else 0
    for _ in range(iters):
        # Update W: for each (i,k), solve m linear equations
        for i in range(n):
            for k in range(p):
                A = []
                for j in range(m):
                    row = [U[i * m + j][c] * V[j * p + k][c] for c in range(r)]
                    A.append(row)
                Aarr = np.array(A, dtype=float)
                b = np.ones(m)
                try:
                    sol, *_ = np.linalg.lstsq(Aarr, b, rcond=None)
                    for c in range(r):
                        W[k * n + i][c] = float(sol[c])
                except Exception:
                    pass
        # Update U: for each (i,j), solve p linear equations
        for i in range(n):
            for j in range(m):
                A = []
                for k in range(p):
                    row = [V[j * p + k][c] * W[k * n + i][c] for c in range(r)]
                    A.append(row)
                Aarr = np.array(A, dtype=float)
                b = np.ones(p)
                try:
                    sol, *_ = np.linalg.lstsq(Aarr, b, rcond=None)
                    for c in range(r):
                        U[i * m + j][c] = float(sol[c])
                except Exception:
                    pass
        # Update V: for each (j,k), solve n linear equations
        for j in range(m):
            for k in range(p):
                A = []
                for i in range(n):
                    row = [U[i * m + j][c] * W[k * n + i][c] for c in range(r)]
                    A.append(row)
                Aarr = np.array(A, dtype=float)
                b = np.ones(n)
                try:
                    sol, *_ = np.linalg.lstsq(Aarr, b, rcond=None)
                    for c in range(r):
                        V[j * p + k][c] = float(sol[c])
                except Exception:
                    pass
    return U, V, W

def _schoolbook(n, m, p):
    """Standard schoolbook decomposition of rank n*m*p."""
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

def _structured_2x3x3_times_1x1x2(n, m, p):
    """Try to use a 2x3x3 decomposition tensored with a 1x1x2 structure.
    
    For 2x3 * 3x6, we can view the output as 2x6 = 2*(3x2).
    We can try: for each pair (i, k) where k = 3*a + b, 
    the computation involves A[i,:] * B[:, 3*a+b].
    
    One approach: split the 3x6 matrix into two 3x3 blocks.
    C[:, 0:3] = A * B[:, 0:3] and C[:, 3:6] = A * B[:, 3:6].
    
    So we need two 2x3*3x3 multiplications, each of which can be done
    with some rank. If we find a rank-r decomposition for 2x3*3x3,
    we can use 2r total rank for 2x3*3x6 by doubling.
    
    But actually, we can share some factors if we're clever.
    """
    # For now, just return schoolbook as fallback
    return _schoolbook(n, m, p)

def _random_init_halfint(rank, rows, rng):
    """Random half-integer initialization."""
    return [[rng.choice([-1, -0.5, 0.5, 1]) for _ in range(rank)] for _ in range(rows)]

def _random_init_float(rank, rows, rng, scale=2.0):
    """Random float initialization."""
    return [[rng.uniform(-scale, scale) for _ in range(rank)] for _ in range(rows)]

def _try_rank(rank, n, m, p, rng, max_attempts=30):
    """Try to find a valid decomposition of given rank."""
    best_res = float('inf')
    best_result = None
    
    for attempt in range(max_attempts):
        if _remaining() < 10.0:
            break
        
        # Choose initialization strategy
        if attempt % 3 == 0:
            U = _random_init_halfint(rank, n * m, rng)
            V = _random_init_halfint(rank, m * p, rng)
            W = _random_init_halfint(rank, p * n, rng)
        elif attempt % 3 == 1:
            U = _random_init_float(rank, n * m, rng, scale=1.0)
            V = _random_init_float(rank, m * p, rng, scale=1.0)
            W = _random_init_float(rank, p * n, rng, scale=1.0)
        else:
            # Sparse initialization
            U = [[0.0] * rank for _ in range(n * m)]
            V = [[0.0] * rank for _ in range(m * p)]
            W = [[0.0] * rank for _ in range(p * n)]
            for row in U:
                for c in range(rank):
                    if rng.random() < 0.3:
                        row[c] = rng.choice([-1, -0.5, 0.5, 1])
            for row in V:
                for c in range(rank):
                    if rng.random() < 0.3:
                        row[c] = rng.choice([-1, -0.5, 0.5, 1])
            for row in W:
                for c in range(rank):
                    if rng.random() < 0.3:
                        row[c] = rng.choice([-1, -0.5, 0.5, 1])
        
        U, V, W = _refine(U, V, W, n, m, p, iters=40)
        res = _residual(U, V, W, n, m, p)
        
        if res < best_res:
            best_res = res
            if res < 1e-7:
                rounded = _try_round_verify(U, V, W)
                if rounded is not None:
                    return rounded
            best_result = (U, V, W)
    
    return None

def construct():
    n, m, p = DIMENSIONS
    best = _schoolbook(n, m, p)
    best_rank = n * m * p  # 36
    
    rng = random.Random(42)
    
    # Try ranks from just below schoolbook down to a reasonable minimum
    # For 2x3x6, known lower bounds suggest rank >= 27 or so
    for rank in range(best_rank - 1, 26, -1):
        if _remaining() < 15.0:
            break
        
        result = _try_rank(rank, n, m, p, rng, max_attempts=25)
        if result is not None:
            best = result
            best_rank = rank
            # Found a valid decomposition, try to go lower
            continue
    
    return best

if __name__ == "__main__":
    print(construct())
