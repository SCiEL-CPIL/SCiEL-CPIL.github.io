import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Leave 20s safety margin
    
    # Build the target tensor T[i*m+j, j*p+k, k*n+i] = 1 for valid (i,j,k), 0 otherwise
    # The matrix multiplication tensor M_{n,m,p} has entries:
    # T[a,b,c] = 1 if there exist i,j,k with a=i*m+j, b=j*p+k, c=k*n+i, else 0
    # There are exactly n*m*p = 120 non-zero entries, each equal to 1.
    
    # We want to find rank r such that there exist U (n*m x r), V (m*p x r), W (p*n x r)
    # with: for all (a,b,c) in nonzero set: sum_k U[a,k]*V[b,k]*W[c,k] = 1
    # and for all other (a,b,c): sum_k U[a,k]*V[b,k]*W[c,k] = 0
    
    # Build the list of non-zero entries
    nonzero_entries = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                nonzero_entries.append((a, b, c))
    
    num_nonzero = len(nonzero_entries)
    
    # Schoolbook decomposition (fallback)
    def schoolbook():
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
    
    fallback = schoolbook()
    
    # Verify a decomposition
    def verify(U, V, W, tol=1e-8):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        r = U.shape[1]
        
        # Check non-zero entries
        max_err = 0.0
        for (a, b, c) in nonzero_entries:
            val = np.dot(U[a, :], V[b, :] * W[c, :])
            max_err = max(max_err, abs(val - 1.0))
        
        if max_err > tol:
            return False, max_err
        
        # Check that invalid entries are zero
        # We need to check all (a,b,c) that are NOT in nonzero_entries
        # But this is expensive for large tensors. Let's be smart about it.
        # The full tensor size is (n*m) * (m*p) * (p*n) = 20*30*24 = 14400
        # For each invalid (a,b,c), we need sum_k U[a,k]*V[b,k]*W[c,k] = 0
        
        # Actually, for the tensor to be exactly the matrix multiplication tensor,
        # we need all entries to match. Let's check a sample of invalid entries.
        # But for correctness, we should check all.
        
        # Build a set of nonzero entries for quick lookup
        nz_set = set(nonzero_entries)
        
        # Check all entries
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    if (a, b, c) not in nz_set:
                        val = np.dot(U[a, :], V[b, :] * W[c, :])
                        max_err = max(max_err, abs(val))
        
        return max_err <= tol, max_err
    
    # Try to find a lower rank decomposition
    # Strategy: Use ALS (Alternating Least Squares) with careful initialization
    # and try to reduce rank by merging similar columns
    
    def try_als(r, max_iters=200, tol=1e-10):
        if time.time() > deadline:
            return None
        
        # Initialize with schoolbook-like structure but with rank r
        # We'll use random initialization and ALS
        
        U = np.random.randn(n*m, r) * 0.5
        V = np.random.randn(m*p, r) * 0.5
        W = np.random.randn(p*n, r) * 0.5
        
        # ALS: For each mode, solve a linear system
        # For fixed V and W, U[a, :] should satisfy:
        # sum_k U[a,k] * V[b,k] * W[c,k] = 1 for (a,b,c) in nonzero
        # This is a linear system in U[a, :]
        
        # Precompute for efficiency
        # For each a, the equations involve b and c such that (a,b,c) is in nonzero
        # a = i*m+j, so for fixed a, i = a//m, j = a%m
        # b = j*p+k, c = k*n+i
        # So for fixed a (i,j), we have k from 0 to p-1, and b = j*p+k, c = k*n+i
        
        # For fixed a, the equations are:
        # For each k in 0..p-1: sum_r U[a,r] * V[j*p+k, r] * W[k*n+i, r] = 1
        # This is: U[a, :] dot (V[j*p+k, :] * W[k*n+i, :]) = 1
        # So we have p equations in r unknowns for U[a, :]
        
        # Similarly for V and W.
        
        for iteration in range(max_iters):
            if time.time() > deadline:
                return None
            
            # Update U
            for a in range(n * m):
                i = a // m
                j = a % m
                # Equations: for k in 0..p-1
                # U[a, :] dot (V[j*p+k, :] * W[k*n+i, :]) = 1
                A = np.array([V[j*p+k, :] * W[k*n+i, :] for k in range(p)])
                b_vec = np.ones(p)
                # Solve least squares
                if A.shape[0] >= A.shape[1]:
                    U[a, :], residuals, rank, sv = np.linalg.lstsq(A, b_vec, rcond=None)
                else:
                    # Underdetermined: use pseudo-inverse
                    U[a, :] = np.linalg.pinv(A) @ b_vec
            
            # Update V
            for b in range(m * p):
                j = b // p
                k = b % p
                # Equations: for i in 0..n-1
                # V[b, :] dot (U[i*m+j, :] * W[k*n+i, :]) = 1
                A = np.array([U[i*m+j, :] * W[k*n+i, :] for i in range(n)])
                b_vec = np.ones(n)
                if A.shape[0] >= A.shape[1]:
                    V[b, :], residuals, rank, sv = np.linalg.lstsq(A, b_vec, rcond=None)
                else:
                    V[b, :] = np.linalg.pinv(A) @ b_vec
            
            # Update W
            for c in range(p * n):
                k = c // n
                i = c % n
                # Equations: for j in 0..m-1
                # W[c, :] dot (U[i*m+j, :] * V[j*p+k, :]) = 1
                A = np.array([U[i*m+j, :] * V[j*p+k, :] for j in range(m)])
                b_vec = np.ones(m)
                if A.shape[0] >= A.shape[1]:
                    W[c, :], residuals, rank, sv = np.linalg.lstsq(A, b_vec, rcond=None)
                else:
                    W[c, :] = np.linalg.pinv(A) @ b_vec
        
        # Check if we found a valid solution
        valid, err = verify(U, V, W, tol=1e-6)
        if valid:
            # Convert to list format
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        return None
    
    # Try ranks from a reasonable lower bound upward
    # Lower bound considerations:
    # For M_{n,m,p}, rank >= max(n*m, m*p, p*n) is not always true
    # But rank >= max(n, m, p) * something
    # Let's try a few ranks
    
    best_result = None
    best_rank = n * m * p  # schoolbook rank
    
    # Try some candidate ranks
    # For 4x5x6, schoolbook is 120
    # Let's try to find something better
    
    candidate_ranks = []
    # Start from a lower estimate
    # The border rank of M_{4,5,6} is unknown, but let's try some values
    for r in range(30, 121):
        candidate_ranks.append(r)
    
    # Sort by rank (lower is better)
    candidate_ranks.sort()
    
    for r in candidate_ranks:
        if time.time() > deadline - 10:
            break
        
        # Try ALS with this rank
        result = try_als(r, max_iters=50)
        if result is not None:
            best_result = result
            best_rank = r
            break  # Found a valid solution, stop
    
    if best_result is None:
        best_result = fallback
    
    return best_result
