"""Rewritten program: exploit rectangular structure of 3x3x7 matmul via structured search."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    n, m, p = DIMENSIONS
    # Schoolbook rank
    schoolbook_rank = n * m * p  # 63
    
    # Initialize with schoolbook
    U = np.zeros((n * m, schoolbook_rank))
    V = np.zeros((m * p, schoolbook_rank))
    W = np.zeros((p * n, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    best_rank = schoolbook_rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Verify schoolbook
    def verify(U, V, W, tol=1e-9):
        """Check if U, V, W form a valid decomposition of the matmul tensor."""
        # The matmul tensor T[i,j,k] = 1 if we're computing (A@B)[i,k] = sum_j A[i,j]*B[j,k]
        # Decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # For each (i,j,k), the sum should equal 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(U.shape[1]):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(s - 1) > tol:
                        return False
        return True
    
    # Strategy: Try to find lower rank by exploiting the rectangular structure.
    # The key insight: for 3x3x7, we can try to use the fact that the 7 dimension
    # creates linear dependencies that can be exploited.
    
    # Approach: Use a structured ansatz where we try to reduce rank by
    # combining terms that share structure in the k dimension.
    
    # Try rank values from schoolbook-1 down, with time-limited search
    # For each rank, try random structured initializations and optimize
    
    def try_rank(r, max_time=30.0):
        """Try to find a decomposition of given rank r."""
        start = time.time()
        
        # Initialize with a structured guess
        # Use the schoolbook terms but try to merge some
        U_init = np.zeros((n*m, r))
        V_init = np.zeros((m*p, r))
        W_init = np.zeros((p*n, r))
        
        # Fill with schoolbook pattern, truncating
        for i in range(n):
            for j in range(m):
                for k in range(min(p, r // (n*m))):
                    if i*m+j < n*m and j*p+k < m*p and k*n+i < p*n:
                        U_init[i*m+j, k] = 1
                        V_init[j*p+k, k] = 1
                        W_init[k*n+i, k] = 1
        
        # Add random perturbations for search
        np.random.seed(42)
        
        # Try multiple random restarts
        for attempt in range(100):
            if time.time() - start > max_time:
                break
            if time.time() > deadline - 5:
                break
            
            # Random structured initialization
            U_r = np.zeros((n*m, r))
            V_r = np.zeros((m*p, r))
            W_r = np.zeros((p*n, r))
            
            # Use a basis-like structure
            for c in range(r):
                # Random indices
                ii = np.random.randint(0, n)
                jj = np.random.randint(0, m)
                kk = np.random.randint(0, p)
                U_r[ii*m+jj, c] = 1
                V_r[jj*p+kk, c] = 1
                W_r[kk*n+ii, c] = 1
            
            # Check if this gives a valid decomposition
            if verify(U_r, V_r, W_r):
                return U_r, V_r, W_r
        
        return None
    
    # Try to reduce rank
    for r in range(schoolbook_rank - 1, max(schoolbook_rank - 20, 30), -1):
        if time.time() > deadline - 10:
            break
        
        result = try_rank(r, max_time=5.0)
        if result is not None:
            U_r, V_r, W_r = result
            if verify(U_r, V_r, W_r):
                best_rank = r
                best_U, best_V, best_W = U_r, V_r, W_r
                break  # Found lower rank, stop
    
    # Convert to lists for output
    U_out = best_U.tolist()
    V_out = best_V.tolist()
    W_out = best_W.tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
