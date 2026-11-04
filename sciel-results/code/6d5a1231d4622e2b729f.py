"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 120  # 120s to be safe within 150s
    
    # Index mappings
    # U index: i*m + j, where i in [0,n), j in [0,m)
    # V index: j*p + k, where j in [0,m), k in [0,p)
    # W index: k*n + i, where k in [0,p), i in [0,n)
    
    # The tensor equation: for all (i,j,k) in [0,n)x[0,m)x[0,p):
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    best_rank = n * m * p  # schoolbook
    best_solution = None
    
    # Generate schoolbook solution as baseline (Integers are valid half-integers)
    U_sb = np.zeros((n*m, best_rank))
    V_sb = np.zeros((m*p, best_rank))
    W_sb = np.zeros((p*n, best_rank))
    for col in range(best_rank):
        i = col // (m*p)
        j = (col // p) % m
        k = col % p
        U_sb[i*m+j, col] = 1.0
        V_sb[j*p+k, col] = 1.0
        W_sb[k*n+i, col] = 1.0
    best_solution = (U_sb, V_sb, W_sb)
    
    def check_solution(U, V, W, tol=1e-8):
        """Check if the solution is valid."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True

    def is_valid_half_integer(arr, tol=1e-9):
        """Check if all entries are integers or half-integers."""
        for val in arr.flatten():
            if abs(val * 2 - round(val * 2)) > tol:
                return False
        return True

    # Use Alternating Least Squares (ALS) to find low-rank decompositions
    search_time_limit = 80.0 # seconds - use most of the budget
    start_time = time.time()
    
    # Theoretical lower bound for matrix multiplication rank
    # For n x m times m x p, rank >= max(n*m, m*p, p*n) / max(n,m,p) is not tight
    # Known bounds: rank >= n*m*p / max(n,m,p) for some cases, but let's start from 30
    min_rank_to_try = 30
    
    def als_decompose(rank, max_iter=50, tol=1e-10):
        """
        Find U, V, W of given rank such that
        sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all (i,j,k)
        """
        # Initialize with random half-integer values
        np.random.seed(hash((n,m,p,rank)) % 2**32)
        vals = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
        
        U = np.random.choice(vals, size=(n*m, rank))
        V = np.random.choice(vals, size=(m*p, rank))
        W = np.random.choice(vals, size=(p*n, rank))
        
        # Ensure no column is all zeros
        for c in range(rank):
            if np.allclose(U[:, c], 0):
                U[np.random.randint(n*m), c] = 1.0
            if np.allclose(V[:, c], 0):
                V[np.random.randint(m*p), c] = 1.0
            if np.allclose(W[:, c], 0):
                W[np.random.randint(p*n), c] = 1.0
        
        for iteration in range(max_iter):
            if time.time() - start_time > search_time_limit:
                break
            
            # Update U: for each (i,j), we need sum_c U[i*m+j,c] * (V[j*p+k,c] * W[k*n+i,c]) = 1
            # This is a linear system: for each (i,j), U[i*m+j,:] @ diag(V[j*p+k,:]*W[k*n+i,:]) = 1
            # But this needs to hold for ALL k simultaneously
            # Better approach: treat as tensor completion
            
            # For fixed V and W, each entry of U is determined by:
            # For each (i,j): the vector U[i*m+j,:] must satisfy
            # For all k: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
            # This is rank * p equations for rank unknowns (U[i*m+j,:])
            # Usually overdetermined, so we use least squares
            
            # Build the system for U
            U_new = np.zeros_like(U)
            for i in range(n):
                for j in range(m):
                    # For this (i,j), we have p equations (one for each k)
                    # A[c, k] = V[j*p+k, c] * W[k*n+i, c]
                    # We want U[i*m+j, :] @ A = ones(p)
                    # i.e., A.T @ U[i*m+j, :].T = ones(p)
                    A = np.zeros((rank, p))
                    for k in range(p):
                        A[:, k] = V[j*p+k, :] * W[k*n+i, :]
                    # Solve A.T @ x = ones(p) for x = U[i*m+j, :]
                    try:
                        U_new[i*m+j, :], residuals, rank_A, sv = np.linalg.lstsq(A.T, np.ones(p), rcond=None)
                    except:
                        U_new[i*m+j, :] = U[i*m+j, :]
            
            U = U_new
            
            # Update V similarly
            V_new = np.zeros_like(V)
            for j in range(m):
                for k in range(p):
                    # For this (j,k), we have n equations (one for each i)
                    # A[c, i] = U[i*m+j, c] * W[k*n+i, c]
                    # We want V[j*p+k, :] @ A = ones(n)
                    # i.e., A.T @ V[j*p+k, :].T = ones(n)
                    A = np.zeros((rank, n))
                    for i in range(n):
                        A[:, i] = U[i*m+j, :] * W[k*n+i, :]
                    try:
                        V_new[j*p+k, :], residuals, rank_A, sv = np.linalg.lstsq(A.T, np.ones(n), rcond=None)
                    except:
                        V_new[j*p+k, :] = V[j*p+k, :]
            
            V = V_new
            
            # Update W similarly
            W_new = np.zeros_like(W)
            for k in range(p):
                for i in range(n):
                    # For this (k,i), we have m equations (one for each j)
                    # A[c, j] = U[i*m+j, c] * V[j*p+k, c]
                    # We want W[k*n+i, :] @ A = ones(m)
                    # i.e., A.T @ W[k*n+i, :].T = ones(m)
                    A = np.zeros((rank, m))
                    for j in range(m):
                        A[:, j] = U[i*m+j, :] * V[j*p+k, :]
                    try:
                        W_new[k*n+i, :], residuals, rank_A, sv = np.linalg.lstsq(A.T, np.ones(m), rcond=None)
                    except:
                        W_new[k*n+i, :] = W[k*n+i, :]
            
            W = W_new
            
            # Check convergence
            if check_solution(U, V, W, tol=1e-6):
                # Snap to half-integers if close
                U = np.round(U * 2) / 2
                V = np.round(V * 2) / 2
                W = np.round(W * 2) / 2
                if check_solution(U, V, W, tol=1e-8):
                    return U, V, W
                else:
                    # If snapping broke it, keep the floating point version
                    return U, V, W
        
        # Final check
        if check_solution(U, V, W, tol=1e-8):
            return U, V, W
        
        return None
    
    # Try ranks from lower bound up to schoolbook
    for rank in range(min_rank_to_try, best_rank):
        if time.time() - start_time > search_time_limit:
            break
        
        # Try multiple random restarts with ALS
        max_restarts = 5
        found = False
        for restart in range(max_restarts):
            if time.time() - start_time > search_time_limit:
                break
            
            result = als_decompose(rank)
            if result is not None:
                U, V, W = result
                best_rank = rank
                best_solution = (U, V, W)
                found = True
                break
        
        if found:
            break
            
    # If no better solution found, use schoolbook
    U, V, W = best_solution
    
    # Final safety check: ensure returned values are valid
    if not check_solution(U, V, W, tol=1e-6):
        # Fall back to schoolbook
        U = U_sb
        V = V_sb
        W = W_sb
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
