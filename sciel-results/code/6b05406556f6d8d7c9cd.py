"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [5, 5, 5]

import numpy as np
import time
import random
from itertools import product

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for all i,j,k."""
    rank = U.shape[1]
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i,j,k] = np.dot(U[i*m+j, :], V[j*p+k, :]) * 0  # placeholder
                # Actually: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
                # We can compute this as: for each c, U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
                # Let's use einsum or manual
                vals = U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :]
                T[i,j,k] = np.sum(vals)
    return np.allclose(T, 1.0, atol=tol)


def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    """Faster verification using numpy broadcasting."""
    # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U_r = U.reshape(n, m, -1)
    V_r = V.reshape(m, p, -1)
    W_r = W.reshape(p, n, -1)
    
    # T[i,j,k] = sum_c U_r[i,j,c] * V_r[j,k,c] * W_r[k,i,c]
    # Use einsum: 'ijc,jkc,kic->ijk'
    T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r, optimize=True)
    return np.allclose(T, 1.0, atol=tol)


def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U, V, W


def _try_optimize_from_schoolbook(n, m, p, target_rank, time_budget=30):
    """Try to find a lower-rank decomposition by starting from schoolbook and optimizing."""
    start_time = time.time()
    deadline = start_time + time_budget
    
    # Start with schoolbook at target_rank+delta and try to zero out columns
    # Alternative: parameterize and use gradient descent
    
    # Let's try a different approach: assume the decomposition has a specific structure
    # For 5x5x5, try to find a decomposition with rank < 125
    
    # One approach: use the fact that we can have shared structure
    # Try rank = target_rank and initialize randomly, then optimize
    
    best_rank = n * m * p
    best_result = None
    
    for rank in range(target_rank, min(target_rank + 10, n*m*p + 1)):
        if time.time() > deadline:
            break
        
        # Try multiple random initializations
        for trial in range(20):
            if time.time() > deadline - 5:
                break
            
            np.random.seed(trial * 1000 + rank)
            U = np.random.randn(n*m, rank) * 0.1
            V = np.random.randn(m*p, rank) * 0.1
            W = np.random.randn(p*n, rank) * 0.1
            
            # Initialize with some structure
            # Set up initial values closer to schoolbook
            for c in range(min(rank, n*m*p)):
                i, j, k = np.unravel_index(c, (n, m, p))
                U[i*m+j, c] = 1.0
                V[j*p+k, c] = 1.0
                W[k*n+i, c] = 1.0
            
            # Compute current error
            def compute_error():
                U_r = U.reshape(n, m, -1)
                V_r = V.reshape(m, p, -1)
                W_r = W.reshape(p, n, -1)
                T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r, optimize=True)
                return np.sum((T - 1.0)**2)
            
            err = compute_error()
            if err < 1e-8:
                best_rank = rank
                best_result = (U.copy(), V.copy(), W.copy())
                break
            
            # Simple gradient-free optimization: coordinate descent
            # For each column, try to adjust it to reduce error
            # This is too slow for large problems
            
            # Alternative: use Nelder-Mead or similar
            # But with 3*25*rank parameters, this is too many
            
            # Let's try a simpler approach: just check if random init works
            if err < 1e-6:
                # Close enough, try to refine
                for step in range(100):
                    if time.time() > deadline - 5:
                        break
                    # Perturb randomly
                    mask = np.random.rand(n*m, rank) < 0.1
                    U[mask] += np.random.randn(*mask.shape) * 0.01
                    mask = np.random.rand(m*p, rank) < 0.1
                    V[mask] += np.random.randn(*mask.shape) * 0.01
                    mask = np.random.rand(p*n, rank) < 0.1
                    W[mask] += np.random.randn(*mask.shape) * 0.01
                    
                    err = compute_error()
                    if err < 1e-8:
                        best_rank = rank
                        best_result = (U.copy(), V.copy(), W.copy())
                        break
    
    if best_result is not None:
        return best_rank, best_result
    return n*m*p, None


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to find lower-rank decompositions
    # Strategy 1: Try ranks from 124 down to 100
    for rank in range(124, 99, -1):
        if time.time() > deadline - 20:
            break
        
        # Try a structured approach: assume the decomposition has symmetry
        # For 5x5x5, try to exploit the fact that 5 is prime
        
        # One idea: use a circulant or FFT-based structure
        # Another: use a block-diagonal structure
        
        # Let's try: assume U, V, W have entries that are sums of a few basis vectors
        # This is a heuristic
        
        found = False
        for trial in range(50):
            if time.time() > deadline - 20:
                break
            
            np.random.seed(trial * 7919 + rank)
            
            # Random initialization with sparsity
            U_r = np.zeros((n*m, rank))
            V_r = np.zeros((m*p, rank))
            W_r = np.zeros((p*n, rank))
            
            for c in range(rank):
                # Each column has a few non-zero entries
                nnz = random.randint(3, 8)
                idx = np.random.choice(n*m, nnz, replace=False)
                U_r[idx, c] = np.random.choice([-1, 0.5, 1, 2], nnz)
                
                idx = np.random.choice(m*p, nnz, replace=False)
                V_r[idx, c] = np.random.choice([-1, 0.5, 1, 2], nnz)
                
                idx = np.random.choice(p*n, nnz, replace=False)
                W_r[idx, c] = np.random.choice([-1, 0.5, 1, 2], nnz)
            
            if _verify_fast(U_r, V_r, W_r, n, m, p, tol=1e-9):
                best_rank = rank
                best_result = {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                found = True
                break
        
        if found:
            break
    
    # Strategy 2: Try to use known algebraic identities
    # For example, use the fact that (a+b)(c+d) = ac + ad + bc + bd
    # This can sometimes reduce rank
    
    if not found and time.time() < deadline - 30:
        # Try a more systematic search
        # Assume the decomposition has a specific form:
        # U[i*m+j, c] = f(i, j, c) where f is a polynomial in i, j
        # This is a heuristic that may or may not work
        
        # Try rank 120 with a structured ansatz
        for rank in [124, 123, 122, 121, 120]:
            if time.time() > deadline - 15:
                break
            
            for trial in range(30):
                if time.time() > deadline - 15:
                    break
                
                np.random.seed(trial * 31 + rank)
                
                # Initialize with schoolbook-like structure but with some merging
                U_r = np.zeros((n*m, rank))
                V_r = np.zeros((m*p, rank))
                W_r = np.zeros((p*n, rank))
                
                # Assign each (i,j,k) to a column, allowing some sharing
                assignments = {}
                col = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            if col >= rank:
                                # Need to share columns
                                # Find a similar (i',j',k') to share with
                                best_sim = None
                                best_dist = float('inf')
                                for (i2, j2, k2), c2 in assignments.items():
                                    dist = abs(i-i2) + abs(j-j2) + abs(k-k2)
                                    if dist < best_dist:
                                        best_dist = dist
                                        best_sim = (i2, j2, k2, c2)
                                
                                if best_sim is not None and best_dist <= 2:
                                    # Share the column, adjust values
                                    i2, j2, k2, c2 = best_sim
                                    # This is a heuristic - may not work
                                    continue
                                else:
                                    # Skip this term (will cause error)
                                    continue
                            
                            assignments[(i,j,k)] = col
                            U_r[i*m+j, col] = 1.0
                            V_r[j*p+k, col] = 1.0
                            W_r[k*n+i, col] = 1.0
                            col += 1
                
                if _verify_fast(U_r, V_r, W_r, n, m, p, tol=1e-9):
                    best_rank = rank
                    best_result = {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                    break
    
    return best_result
# EVOLVE-BLOCK-END
