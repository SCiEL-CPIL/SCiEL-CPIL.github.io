"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, tol=1e-10):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    rank = len(U[0]) if U else 0
    # Build the tensor T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # Should equal 1 for all i,j,k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
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

def _try_rank(n, m, p, rank, time_budget, seed=0):
    """
    Try to find a rank-rank decomposition using random initialization + iterative refinement.
    Uses the approach: we need U (n*m x rank), V (m*p x rank), W (p*n x rank) such that
    sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for all i,j,k.
    
    We use a gradient-free / alternating approach or direct linear algebra.
    """
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)
    
    start = time.time()
    
    # We'll use an iterative approach:
    # Initialize U, V, W randomly.
    # Then alternate: fix two, solve for the third (linear system).
    
    U = np_rng.randn(n * m, rank) * 0.1
    V = np_rng.randn(m * p, rank) * 0.1
    W = np_rng.randn(p * n, rank) * 0.1
    
    # Scale to make initial products O(1)
    U /= (np.linalg.norm(U) + 1e-15)
    V /= (np.linalg.norm(V) + 1e-15)
    W /= (np.linalg.norm(W) + 1e-15)
    
    best_err = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for iteration in range(200):
        if time.time() - start > time_budget:
            break
        
        # Fix V and W, solve for U.
        # For each (i,j): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all k
        # This is a system: for fixed (i,j), we have p equations in rank unknowns (U[i*m+j,:])
        # A[i,j] @ U[i*m+j,:] = b[i,j] where A[i,j][k,c] = V[j*p+k,c]*W[k*n+i,c]
        
        for i in range(n):
            for j in range(m):
                # Build A: p x rank
                A = np.zeros((p, rank))
                for k in range(p):
                    for c in range(rank):
                        A[k, c] = V[j*p+k, c] * W[k*n+i, c]
                b = np.ones(p)
                # Solve least squares
                try:
                    sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    U[i*m+j, :] = sol
                except:
                    pass
        
        # Fix U and W, solve for V.
        for j in range(m):
            for k in range(p):
                A = np.zeros((n, rank))
                for i in range(n):
                    for c in range(rank):
                        A[i, c] = U[i*m+j, c] * W[k*n+i, c]
                b = np.ones(n)
                try:
                    sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    V[j*p+k, :] = sol
                except:
                    pass
        
        # Fix U and V, solve for W.
        for k in range(p):
            for i in range(n):
                A = np.zeros((m, rank))
                for j in range(m):
                    for c in range(rank):
                        A[j, c] = U[i*m+j, c] * V[j*p+k, c]
                b = np.ones(m)
                try:
                    sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    W[k*n+i, :] = sol
                except:
                    pass
        
        # Compute error
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    err += (s - 1.0)**2
        err = np.sqrt(err / (n*m*p))
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if err < 1e-12:
            break
    
    if best_err < 1e-10:
        # Convert to lists
        U_list = [[float(x) for x in row] for row in best_U]
        V_list = [[float(x) for x in row] for row in best_V]
        W_list = [[float(x) for x in row] for row in best_W]
        return U_list, V_list, W_list, best_err
    return None, None, None, best_err

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin before 150s kill
    
    # Start with schoolbook
    schoolbook = _schoolbook(n, m, p)
    schoolbook_rank = n * m * p  # 42
    
    best_result = schoolbook
    best_rank = schoolbook_rank
    
    # Try ranks from theoretical lower bound up to schoolbook
    # Lower bound: max(n*m, m*p, p*n) = max(6, 21, 14) = 21
    # Also, the rank of matrix multiplication is at least the max of the dimensions
    # For 2x3 by 3x7: lower bound is at least 21 (from m*p)
    
    lower_bound = max(n*m, m*p, p*n)
    
    # Try ranks from lower_bound to schoolbook_rank - 1
    for rank in range(lower_bound, schoolbook_rank):
        if time.time() > deadline - 10:
            break
        
        time_budget = min(30, deadline - time.time())
        if time_budget < 5:
            break
        
        # Try a few seeds
        for seed in range(5):
            if time.time() > deadline - 5:
                break
            
            U, V, W, err = _try_rank(n, m, p, rank, time_budget, seed=seed)
            if U is not None:
                # Verify
                if _verify(U, V, W, n, m, p, tol=1e-9):
                    best_result = {"U": U, "V": V, "W": W}
                    best_rank = rank
                    return best_result
                else:
                    # Close but not exact; try to polish
                    pass
    
    return best_result
# EVOLVE-BLOCK-END
