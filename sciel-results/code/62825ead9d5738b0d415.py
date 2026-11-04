"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

def _verify(U, V, W, n, m, p):
    """Verify that the tensor decomposition is exact."""
    r = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1) > 1e-9:
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
    return U, V, W

def _als_decompose(n, m, p, rank, max_iters=200, tol=1e-10, seed=0):
    """
    Use ALS to find a rank-r decomposition of the all-ones tensor.
    T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    """
    rng = np.random.RandomState(seed)
    
    U = rng.randn(n * m, rank) * 0.1
    V = rng.randn(m * p, rank) * 0.1
    W = rng.randn(p * n, rank) * 0.1
    
    # Precompute index mappings
    # For each (i,j), U index is i*m+j
    # For each (j,k), V index is j*p+k  
    # For each (k,i), W index is k*n+i
    
    for iteration in range(max_iters):
        # Update U: for each (i,j), U[i*m+j, :] = argmin sum_k (1 - V[j*p+k,:] dot W[k*n+i,:])^2 * ...
        # Actually, for fixed V and W, U[i*m+j, c] appears in terms where we sum over k
        # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # For fixed V, W: minimize sum_{i,j,k} (1 - sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c])^2
        # Derivative w.r.t U[i*m+j, c]: -2 * sum_k (1 - T[i,j,k]) * V[j*p+k,c] * W[k*n+i,c]
        # This is a linear system. Let's set up the normal equations.
        
        # For each (i,j): 
        # sum_c U[i*m+j,c] * (sum_k V[j*p+k,c]*W[k*n+i,c] * V[j*p+k,c'] * W[k*n+i,c']) = sum_k V[j*p+k,c]*W[k*n+i,c]
        # Let A[i,j] be a rank x rank matrix: A[i,j][c,c'] = sum_k V[j*p+k,c]*W[k*n+i,c']*V[j*p+k,c']*W[k*n+i,c']
        # Wait, that's not right. Let me redo.
        
        # T[i,j,k] = sum_c U[i*m+j,c] * (V[j*p+k,c] * W[k*n+i,c])
        # Let B[i,j,k,c] = V[j*p+k,c] * W[k*n+i,c]
        # Then T[i,j,k] = sum_c U[i*m+j,c] * B[i,j,k,c]
        # We want T[i,j,k] = 1 for all i,j,k
        # For fixed (i,j): sum_k (1 - sum_c U[i*m+j,c]*B[i,j,k,c])^2
        # = sum_k 1 - 2*sum_k sum_c U[i*m+j,c]*B[i,j,k,c] + sum_k (sum_c U[i*m+j,c]*B[i,j,k,c])^2
        # Derivative w.r.t U[i*m+j,c']: 
        # -2*sum_k B[i,j,k,c'] + 2*sum_k B[i,j,k,c'] * sum_c U[i*m+j,c]*B[i,j,k,c]
        # = 0
        # So: sum_k B[i,j,k,c'] * (sum_c U[i*m+j,c]*B[i,j,k,c] - 1) = 0
        # sum_c (sum_k B[i,j,k,c']*B[i,j,k,c]) * U[i*m+j,c] = sum_k B[i,j,k,c']
        
        # Let M[i,j] be rank x rank: M[i,j][c',c] = sum_k B[i,j,k,c'] * B[i,j,k,c]
        # Let b[i,j] be rank: b[i,j][c'] = sum_k B[i,j,k,c']
        # Solve M[i,j] @ U[i*m+j] = b[i,j]
        
        for i in range(n):
            for j in range(m):
                # Compute B[i,j,k,c] = V[j*p+k,c] * W[k*n+i,c]
                # M[c',c] = sum_k B[i,j,k,c'] * B[i,j,k,c]
                # b[c'] = sum_k B[i,j,k,c']
                M = np.zeros((rank, rank))
                b = np.zeros(rank)
                for k in range(p):
                    Bk = V[j*p+k, :] * W[k*n+i, :]  # shape (rank,)
                    M += np.outer(Bk, Bk)
                    b += Bk
                # Add small regularization
                M += 1e-10 * np.eye(rank)
                try:
                    U[i*m+j, :] = np.linalg.solve(M, b)
                except np.linalg.LinAlgError:
                    U[i*m+j, :] = np.linalg.lstsq(M, b, rcond=None)[0]
        
        # Update V: for each (j,k)
        for j in range(m):
            for k in range(p):
                M = np.zeros((rank, rank))
                b = np.zeros(rank)
                for i in range(n):
                    Bi = U[i*m+j, :] * W[k*n+i, :]  # shape (rank,)
                    M += np.outer(Bi, Bi)
                    b += Bi
                M += 1e-10 * np.eye(rank)
                try:
                    V[j*p+k, :] = np.linalg.solve(M, b)
                except np.linalg.LinAlgError:
                    V[j*p+k, :] = np.linalg.lstsq(M, b, rcond=None)[0]
        
        # Update W: for each (k,i)
        for k in range(p):
            for i in range(n):
                M = np.zeros((rank, rank))
                b = np.zeros(rank)
                for j in range(m):
                    Bj = U[i*m+j, :] * V[j*p+k, :]  # shape (rank,)
                    M += np.outer(Bj, Bj)
                    b += Bj
                M += 1e-10 * np.eye(rank)
                try:
                    W[k*n+i, :] = np.linalg.solve(M, b)
                except np.linalg.LinAlgError:
                    W[k*n+i, :] = np.linalg.lstsq(M, b, rcond=None)[0]
        
        # Check convergence
        if iteration % 10 == 0 or iteration == max_iters - 1:
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for c in range(rank):
                            s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                        err += (s - 1.0) ** 2
            if err < tol:
                break
    
    return U, V, W

def _try_exact(U, V, W, n, m, p, rank):
    """
    Try to find exact (integer/half-integer) entries that satisfy the decomposition.
    Use the numerical solution as a guide.
    """
    # Check if current numerical solution is close to exact
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                max_err = max(max_err, abs(s - 1.0))
    
    if max_err < 1e-8:
        # Try to round to nearest half-integer
        U_rounded = [[round(U[i, c] * 2) / 2 for c in range(rank)] for i in range(n * m)]
        V_rounded = [[round(V[i, c] * 2) / 2 for c in range(rank)] for i in range(m * p)]
        W_rounded = [[round(W[i, c] * 2) / 2 for c in range(rank)] for i in range(p * n)]
        
        # Verify rounded solution
        if _verify(U_rounded, V_rounded, W_rounded, n, m, p):
            return U_rounded, V_rounded, W_rounded
    
    # Try integer rounding
    U_int = [[round(U[i, c]) for c in range(rank)] for i in range(n * m)]
    V_int = [[round(V[i, c]) for c in range(rank)] for i in range(m * p)]
    W_int = [[round(W[i, c]) for c in range(rank)] for i in range(p * n)]
    
    if _verify(U_int, V_int, W_int, n, m, p):
        return U_int, V_int, W_int
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Safety margin
    
    # Start with schoolbook
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_result = (U_sb, V_sb, W_sb)
    
    # Try to find lower rank decompositions
    # Based on structure: 8 = 4*2, so if we have 3x4x4 rank r, we get 3x4x8 rank 2r
    # Also try direct search
    
    # Try ranks from low to high
    # Known: schoolbook is 96. Let's try to beat that.
    # Theoretical lower bound: max(n*m, m*p, p*n) = max(12, 32, 24) = 32
    # But tensor rank is generally higher.
    
    # Try ranks: 48, 56, 64, 72, 80, 88, 95
    candidate_ranks = [48, 56, 64, 72, 80, 88, 92, 95]
    
    for rank in candidate_ranks:
        if time.time() > deadline - 10:
            break
        
        # Try multiple random seeds
        found = False
        for seed in range(10):
            if time.time() > deadline - 10:
                break
            
            U, V, W = _als_decompose(n, m, p, rank, max_iters=300, seed=seed)
            
            # Check if it's exact
            result = _try_exact(U, V, W, n, m, p, rank)
            if result is not None:
                U_exact, V_exact, W_exact = result
                if rank < best_rank:
                    best_rank = rank
                    best_result = (U_exact, V_exact, W_exact)
                found = True
                break
        
        if found:
            # Found a valid decomposition at this rank, can stop
            break
    
    # Return best result
    U, V, W = best_result
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
