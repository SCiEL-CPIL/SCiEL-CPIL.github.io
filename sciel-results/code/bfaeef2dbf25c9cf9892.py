"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # 110s to be safe within 150s
    
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

    def als_decompose(rank, max_iter=100, tol=1e-10):
        """
        Find U, V, W of given rank such that
        sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all (i,j,k)
        """
        # Initialize with random real values
        np.random.seed(hash((n,m,p,rank)) % 2**32)
        
        U = np.random.randn(n*m, rank) * 0.1
        V = np.random.randn(m*p, rank) * 0.1
        W = np.random.randn(p*n, rank) * 0.1
        
        # Ensure no column is all zeros
        for c in range(rank):
            if np.allclose(U[:, c], 0):
                U[np.random.randint(n*m), c] = 1.0
            if np.allclose(V[:, c], 0):
                V[np.random.randint(m*p), c] = 1.0
            if np.allclose(W[:, c], 0):
                W[np.random.randint(p*n), c] = 1.0
        
        for iteration in range(max_iter):
            if time.time() > deadline:
                break
            
            # Update U
            U_new = np.zeros_like(U)
            for i in range(n):
                for j in range(m):
                    # A[c, k] = V[j*p+k, c] * W[k*n+i, c]
                    # We want U[i*m+j, :] @ A = ones(p)
                    # i.e., A.T @ U[i*m+j, :].T = ones(p)
                    A = np.zeros((rank, p))
                    for k in range(p):
                        A[:, k] = V[j*p+k, :] * W[k*n+i, :]
                    try:
                        U_new[i*m+j, :], residuals, rank_A, sv = np.linalg.lstsq(A.T, np.ones(p), rcond=None)
                    except:
                        U_new[i*m+j, :] = U[i*m+j, :]
            U = U_new
            
            # Update V
            V_new = np.zeros_like(V)
            for j in range(m):
                for k in range(p):
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
            
            # Update W
            W_new = np.zeros_like(W)
            for k in range(p):
                for i in range(n):
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
            
            # Check convergence and snap to half-integers
            if check_solution(U, V, W, tol=1e-5):
                U_snap = np.round(U * 2) / 2
                V_snap = np.round(V * 2) / 2
                W_snap = np.round(W * 2) / 2
                if check_solution(U_snap, V_snap, W_snap, tol=1e-8):
                    return U_snap, V_snap, W_snap
        
        # Final check
        if check_solution(U, V, W, tol=1e-8):
            U_snap = np.round(U * 2) / 2
            V_snap = np.round(V * 2) / 2
            W_snap = np.round(W * 2) / 2
            if check_solution(U_snap, V_snap, W_snap, tol=1e-8):
                return U_snap, V_snap, W_snap
        
        return None

    # Try ranks from lower bound up to schoolbook
    # Theoretical lower bound for matrix multiplication rank
    # For n x m times m x p, rank >= max(n*m, m*p, p*n)
    min_rank_to_try = max(n*m, m*p, p*n)
    
    # Use most of the budget
    max_restarts_per_rank = 20
    
    for rank in range(min_rank_to_try, best_rank + 1):
        if time.time() > deadline:
            break
        
        found = False
        for restart in range(max_restarts_per_rank):
            if time.time() > deadline:
                break
            
            # Use a different seed for each restart
            np.random.seed(hash((n,m,p,rank,restart)) % 2**32)
            
            result = als_decompose(rank)
            if result is not None:
                U, V, W = result
                best_rank = rank
                best_solution = (U, V, W)
                found = True
                break
        
        if found:
            # If we found a solution, we can break if we hit the lower bound
            # or continue to see if we can find an even lower one? 
            # Since we iterate upwards, the first found is the lowest for this search order.
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
