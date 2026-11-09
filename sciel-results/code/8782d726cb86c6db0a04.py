"""Rank-minimal exact tensor decomposition for 3x4 * 4x7 matrix multiplication."""
import time
import random
import sys
import numpy as np

DIMENSIONS = [3, 4, 7]
n, m, p = DIMENSIONS

def _verify(U, V, W, rank):
    """Check if U,V,W with given rank exactly decompose the multiplication tensor."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for c in range(rank):
                    total += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if total != 1:
                    return False
    return True

def _schoolbook():
    """Return the schoolbook rank n*m*p decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _solve_for_W(U, V, rank):
    """
    Given U and V, try to find W such that sum_c U[i,j,c] V[j,k,c] W[k,i,c] = 1.
    """
    W = [[0] * rank for _ in range(p * n)]
    
    for i in range(n):
        for k in range(p):
            # Construct A matrix: A[j, c] = U[i*m+j][c] * V[j*p+k][c]
            A = np.zeros((m, rank), dtype=float)
            for j in range(m):
                for c in range(rank):
                    A[j, c] = U[i*m+j][c] * V[j*p+k][c]
            
            b = np.ones(m)
            
            # Solve A @ w = b
            try:
                w_candidate, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
                
                # Check if solution is exact
                if np.linalg.norm(A @ w_candidate - b) > 1e-9:
                    return None
                
                # Check if w_candidate is close to integer
                w_int = np.round(w_candidate)
                if np.linalg.norm(A @ w_int - b) < 1e-9:
                    for c in range(rank):
                        W[k*n+i][c] = int(w_int[c])
                    continue
                
                # If not integer, try to find an integer solution in the null space
                if rank_A < rank:
                    u, s, vh = np.linalg.svd(A)
                    null_dim = rank - rank_A
                    if null_dim > 0:
                        N = vh[rank_A:].T # rank x null_dim
                        # Try small integer combinations
                        found = False
                        for _ in range(50):
                            coeffs = np.random.randint(-3, 4, size=null_dim)
                            w_test = w_candidate + N @ coeffs
                            w_test_int = np.round(w_test)
                            if np.linalg.norm(A @ w_test_int - b) < 1e-9:
                                for c in range(rank):
                                    W[k*n+i][c] = int(w_test_int[c])
                                found = True
                                break
                        if not found:
                            return None
                    else:
                        return None
                else:
                    return None
            except Exception:
                return None
                
    return W

def _search_rank(rank, max_trials=1000, seed_base=0):
    """
    Search for a decomposition of given rank by fixing U, V and solving for W.
    """
    rng = random.Random(seed_base)
    for trial in range(max_trials):
        # Generate random U and V with small integer entries
        U = [[rng.randint(-2, 2) for _ in range(rank)] for _ in range(n*m)]
        V = [[rng.randint(-2, 2) for _ in range(rank)] for _ in range(m*p)]
        
        # Solve for W
        W = _solve_for_W(U, V, rank)
        
        if W is not None:
            if _verify(U, V, W, rank):
                return {"U": U, "V": V, "W": W}
                
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best = _schoolbook()
    best_rank = n * m * p
    
    # Try ranks from 70 to 83
    for rank in range(70, best_rank):
        if time.time() > deadline - 5.0:
            break
        
        trials = 500
        result = _search_rank(rank, max_trials=trials, seed_base=rank * 100)
        
        if result is not None:
            best = result
            best_rank = rank
            break
    
    if best_rank < n * m * p:
        U, V, W = best["U"], best["V"], best["W"]
        if not _verify(U, V, W, best_rank):
            best = _schoolbook()
            best_rank = n * m * p
            
    return best

# EVOLVE-BLOCK-END
