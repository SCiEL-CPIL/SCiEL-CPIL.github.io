"""
Search for a rank-minimal exact tensor decomposition for multiplying a 3x4 matrix by a 4x6 matrix.
Uses a structured approach: start with schoolbook, then attempt to find lower-rank decompositions
using a combination of known algebraic identities and heuristic search.

For 3x4x6 matrix multiplication, the schoolbook rank is 3*4*6 = 72.
We try to find a decomposition with rank < 72.
"""
import numpy as np
import random
from itertools import product
import time

DIMENSIONS = [3, 4, 6]

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition is exact."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
                if abs(val - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _try_strassen_like(n, m, p, max_attempts=1000, time_limit=30):
    """
    Try to find a lower-rank decomposition using a greedy/local search approach.
    Start from schoolbook and try to merge terms or find cancellations.
    """
    start_time = time.time()
    U, V, W = _schoolbook(n, m, p)
    current_rank = n * m * p
    
    # Try to reduce rank by finding linear dependencies or using algebraic tricks
    # For now, try a random perturbation + rounding approach
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = current_rank
    
    for attempt in range(max_attempts):
        if time.time() - start_time > time_limit:
            break
            
        # Try a random projection: pick a random target rank below current
        target_rank = random.randint(max(10, current_rank - 5), current_rank - 1)
        
        # Random initialization
        U_rand = np.random.randn(n * m, target_rank) * 0.5
        V_rand = np.random.randn(m * p, target_rank) * 0.5
        W_rand = np.random.randn(p * n, target_rank) * 0.5
        
        # Alternating minimization
        for iteration in range(200):
            if time.time() - start_time > time_limit:
                break
            
            # Update U: for each (i,j), find U[i*m+j,:] to minimize error
            for i in range(n):
                for j in range(m):
                    # residual for this (i,j)
                    res = np.zeros(p, dtype=float)
                    for k in range(p):
                        val = 0.0
                        for r in range(target_rank):
                            val += U_rand[i*m+j, r] * V_rand[j*p+k, r] * W_rand[k*n+i, r]
                        res[k] = 1.0 - val
                    # Update U[i*m+j,:] to reduce error
                    # This is a linear system: min ||res||^2 subject to new U
                    # Simple gradient step
                    grad = np.zeros(target_rank)
                    for k in range(p):
                        for r in range(target_rank):
                            grad[r] += res[k] * V_rand[j*p+k, r] * W_rand[k*n+i, r]
                    U_rand[i*m+j, :] -= 0.01 * grad
            
            # Update V
            for j in range(m):
                for k in range(p):
                    res = np.zeros(n, dtype=float)
                    for i in range(n):
                        val = 0.0
                        for r in range(target_rank):
                            val += U_rand[i*m+j, r] * V_rand[j*p+k, r] * W_rand[k*n+i, r]
                        res[i] = 1.0 - val
                    grad = np.zeros(target_rank)
                    for i in range(n):
                        for r in range(target_rank):
                            grad[r] += res[i] * U_rand[i*m+j, r] * W_rand[k*n+i, r]
                    V_rand[j*p+k, :] -= 0.01 * grad
            
            # Update W
            for k in range(p):
                for i in range(n):
                    res = np.zeros(m, dtype=float)
                    for j in range(m):
                        val = 0.0
                        for r in range(target_rank):
                            val += U_rand[i*m+j, r] * V_rand[j*p+k, r] * W_rand[k*n+i, r]
                        res[j] = 1.0 - val
                    grad = np.zeros(target_rank)
                    for j in range(m):
                        for r in range(target_rank):
                            grad[r] += res[j] * U_rand[i*m+j, r] * V_rand[j*p+k, r]
                    W_rand[k*n+i, :] -= 0.01 * grad
        
        # Check if we found a solution
        if _verify(U_rand, V_rand, W_rand, n, m, p, tol=1e-6):
            best_rank = target_rank
            best_U, best_V, best_W = U_rand.copy(), V_rand.copy(), W_rand.copy()
            break
        
        # Try rounding to half-integers
        U_rounded = np.round(U_rand * 2) / 2
        V_rounded = np.round(V_rand * 2) / 2
        W_rounded = np.round(W_rand * 2) / 2
        if _verify(U_rounded, V_rounded, W_rounded, n, m, p, tol=1e-6):
            best_rank = target_rank
            best_U, best_V, best_W = U_rounded.copy(), V_rounded.copy(), W_rounded.copy()
            break
    
    return best_U, best_V, best_W, best_rank

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Get schoolbook decomposition
    U, V, W = _schoolbook(n, m, p)
    
    # Try to find a lower rank decomposition
    U_try, V_try, W_try, rank_try = _try_strassen_like(n, m, p, max_attempts=500, time_limit=90)
    
    if rank_try < schoolbook_rank:
        return {"U": U_try.tolist(), "V": V_try.tolist(), "W": W_try.tolist()}
    else:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
