"""Program for rank-minimal exact tensor decomposition of 3x5x6 matrix multiplication."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def _verify_exact(U, V, W, n, m, p):
    """
    Verify exact decomposition using integer/rational arithmetic.
    U, V, W are lists of lists with integer or half-integer entries.
    Returns True if the decomposition exactly equals the multiplication tensor.
    """
    rank = len(U[0]) if U else 0
    
    if len(U) != n * m:
        return False
    if len(V) != m * p:
        return False
    if len(W) != p * n:
        return False
    for row in U:
        if len(row) != rank:
            return False
    for row in V:
        if len(row) != rank:
            return False
    for row in W:
        if len(row) != rank:
            return False
    
    # Convert to Fractions for exact arithmetic
    Uf = [[Fraction(x) for x in row] for row in U]
    Vf = [[Fraction(x) for x in row] for row in V]
    Wf = [[Fraction(x) for x in row] for row in W]
    
    # Build tensor and check each entry equals 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = Fraction(0)
                for c in range(rank):
                    total += Uf[i * m + j][c] * Vf[j * p + k][c] * Wf[k * n + i][c]
                if total != 1:
                    return False
    return True

def _verify_float(U, V, W, n, m, p, atol=1e-9):
    """Fast float verification for pruning during search."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    tensor = np.zeros((n, m, p))
    for c in range(rank):
        U_c = U[:, c].reshape(n, m)
        V_c = V[:, c].reshape(m, p)
        W_c = W[:, c].reshape(p, n)
        # Efficient tensor accumulation
        for i in range(n):
            for j in range(m):
                tensor[i, j, :] += U_c[i, j] * (V_c[j, :] * W_c[:, i])
    
    return np.allclose(tensor, 1.0, atol=atol)

def _schoolbook(n, m, p):
    """Return the schoolbook (rank = n*m*p) decomposition."""
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

def _try_structured_reduction(n, m, p, time_limit):
    """
    Try structured rank reductions by sharing terms.
    
    Key idea: In the schoolbook decomposition, each term (i,j,k) has
    U[i*m+j]=1, V[j*p+k]=1, W[k*n+i]=1. If we can merge two terms
    that share the same U or V or W pattern, we reduce rank.
    
    Strategy: Try to find pairs of schoolbook terms that can be combined.
    Two terms (i1,j1,k1) and (i2,j2,k2) can potentially be merged if
    they share a coordinate, creating a new term with combined coefficients.
    
    More practically: try removing one term at a time and see if the
    remaining terms can be adjusted (with integer/half-integer coefficients)
    to still produce the correct tensor.
    """
    start_time = time.time()
    best_rank = n * m * p
    best_solution = None
    
    # Start with schoolbook and try to remove terms
    U = [[0] * (n * m * p) for _ in range(n * m)]
    V = [[0] * (n * m * p) for _ in range(m * p)]
    W = [[0] * (n * m * p) for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # Try removing each term and see if we can compensate
    # This is a greedy approach: try to remove term c and adjust others
    
    for c_remove in range(n * m * p):
        if time.time() - start_time > time_limit:
            break
        
        # Skip if already removed
        if U[0][c_remove] == 0 and V[0][c_remove] == 0 and W[0][c_remove] == 0:
            continue
        
        # Check if term c_remove is active
        active = False
        for idx in range(n * m):
            if U[idx][c_remove] != 0:
                active = True
                break
        if not active:
            continue
        
        # Try removing this term
        U_copy = [row[:] for row in U]
        V_copy = [row[:] for row in V]
        W_copy = [row[:] for row in W]
        
        # Zero out term c_remove
        for idx in range(n * m):
            U_copy[idx][c_remove] = 0
        for idx in range(m * p):
            V_copy[idx][c_remove] = 0
        for idx in range(p * n):
            W_copy[idx][c_remove] = 0
        
        # Check if the decomposition still works without this term
        # This is expensive, so only do it for a subset
        if (c_remove % 10) == 0:  # Check every 10th term
            if _verify_float(U_copy, V_copy, W_copy, n, m, p):
                # Try to find integer/half-integer solution
                if _verify_exact(U_copy, V_copy, W_copy, n, m, p):
                    best_rank = best_rank - 1
                    best_solution = (U_copy, V_copy, W_copy)
    
    return best_solution, best_rank

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110.0  # 110s budget, 10s safety margin
    
    # Try structured reduction
    best_solution, best_rank = _try_structured_reduction(n, m, p, 80.0)
    
    if best_solution is not None and best_rank < n * m * p:
        return {"U": best_solution[0], "V": best_solution[1], "W": best_solution[2]}
    
    # Fallback to schoolbook
    return _schoolbook(n, m, p)
# EVOLVE-BLOCK-END
