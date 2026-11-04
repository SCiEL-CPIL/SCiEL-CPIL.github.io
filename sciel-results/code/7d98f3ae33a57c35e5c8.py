"""Tensor decomposition for 3x4x8 matrix multiplication."""
import numpy as np
import time
import sys

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal (or near-minimal) exact tensor decomposition for
    multiplying a 3x4 matrix by a 4x8 matrix.
    
    Returns a dict with keys 'U', 'V', 'W' where:
    - U has shape (n*m, rank) = (12, rank)
    - V has shape (m*p, rank) = (32, rank)
    - W has shape (p*n, rank) = (24, rank)
    
    The decomposition satisfies: for all i,j,k
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    """
    n, m, p = DIMENSIONS  # 3, 4, 8
    
    # Start with the schoolbook decomposition as fallback
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W
    
    U, V, W = schoolbook()
    best_rank = n * m * p
    
    # Try to find a lower-rank decomposition using ALS with random initialization
    # and rank reduction
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    def verify_decomposition(U, V, W, tol=1e-10):
        """Check if the decomposition is exact."""
        n, m, p = DIMENSIONS
        # Check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
        # This is equivalent to checking that the tensor T[i,j,k] = sum_r U[i,j,r]*V[j,k,r]*W[k,i,r]
        # equals 1 for all i,j,k
        
        # Build the tensor from the decomposition
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def try_als(U, V, W, max_iter=500, tol=1e-8, time_limit=None):
        """
        Alternating Least Squares to refine the decomposition.
        For each mode, fix the other two and solve a least squares problem.
        """
        n, m, p = DIMENSIONS
        nU, nV, nW = n * m, m * p, p * n
        rank = U.shape[1]
        
        # Precompute some useful quantities
        # T[i,j,k] = 1 for all i,j,k
        
        # Mode-1 (U): For each (i,j), U[i*m+j, :] should satisfy
        # sum_r U[i*m+j, r] * (V[j*p+k, r] * W[k*n+i, r]) = 1 for all k
        # This is: U[i*m+j, :] * A1[i,j] = ones(p)
        # where A1[i,j] has rows indexed by k, columns by r
        # A1[i,j, k, r] = V[j*p+k, r] * W[k*n+i, r]
        
        # Mode-2 (V): For each (j,k), V[j*p+k, :] should satisfy
        # sum_r U[i*m+j, r] * (V[j*p+k, r] * W[k*n+i, r]) = 1 for all i
        # V[j*p+k, :] * A2[j,k] = ones(n)
        # where A2[j,k, i, r] = U[i*m+j, r] * W[k*n+i, r]
        
        # Mode-3 (W): For each (k,i), W[k*n+i, :] should satisfy
        # sum_r U[i*m+j, r] * (V[j*p+k, r] * W[k*n+i, r]) = 1 for all j
        # W[k*n+i, :] * A3[k,i] = ones(m)
        # where A3[k,i, j, r] = U[i*m+j, r] * V[j*p+k, r]
        
        for it in range(max_iter):
            if time_limit is not None and time.time() > time_limit:
                break
            
            old_U = U.copy()
            old_V = V.copy()
            old_W = W.copy()
            
            # Update U
            for i in range(n):
                for j in range(m):
                    # A1[i,j, k, r] = V[j*p+k, r] * W[k*n+i, r]
                    A1 = np.zeros((p, rank))
                    for k in range(p):
                        A1[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    # Solve U[i*m+j, :] * A1 = ones(p)
                    # i.e., A1.T @ U[i*m+j, :].T = ones(p)
                    sol, _, _, _ = np.linalg.lstsq(A1.T, np.ones(p), rcond=None)
                    U[i*m+j, :] = sol
            
            # Update V
            for j in range(m):
                for k in range(p):
                    # A2[j,k, i, r] = U[i*m+j, r] * W[k*n+i, r]
                    A2 = np.zeros((n, rank))
                    for i in range(n):
                        A2[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    # Solve V[j*p+k, :] * A2 = ones(n)
                    sol, _, _, _ = np.linalg.lstsq(A2.T, np.ones(n), rcond=None)
                    V[j*p+k, :] = sol
            
            # Update W
            for k in range(p):
                for i in range(n):
                    # A3[k,i, j, r] = U[i*m+j, r] * V[j*p+k, r]
                    A3 = np.zeros((m, rank))
                    for j in range(m):
                        A3[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    # Solve W[k*n+i, :] * A3 = ones(m)
                    sol, _, _, _ = np.linalg.lstsq(A3.T, np.ones(m), rcond=None)
                    W[k*n+i, :] = sol
            
            # Check convergence
            diff = (np.max(np.abs(U - old_U)) + np.max(np.abs(V - old_V)) + 
                    np.max(np.abs(W - old_W)))
            if diff < tol:
                break
        
        return U, V, W
    
    def check_and_round(U, V, W):
        """
        Check if the decomposition is exact and try to round to half-integers.
        Returns (rank, U, V, W) if successful, else None.
        """
        n, m, p = DIMENSIONS
        # First check if it's close to exact
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        
        if max_err > 1e-6:
            return None
        
        # Try to round to half-integers
        def try_round(mat, scale=2):
            """Round to nearest half-integer, check if still exact."""
            rounded = np.round(mat * scale) / scale
            # Check sparsity pattern
            nz = np.where(np.abs(rounded) > 1e-10)
            if len(nz[0]) > 50:  # Too many non-zeros, not sparse enough
                return None
            return rounded
        
        U_r = try_round(U)
        V_r = try_round(V)
        W_r = try_round(W)
        
        if U_r is None or V_r is None or W_r is None:
            return None
        
        # Verify the rounded decomposition
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_r[i*m+j, :], V_r[j*p+k, :] * W_r[k*n+i, :])
                    err = abs(val - 1.0)
                    if err > 1e-10:
                        return None
                    max_err = max(max_err, err)
        
        return U_r, V_r, W_r
    
    # Try various ranks from low to high
    # The theoretical lower bound is max(n*p, m*p, n*m) = max(24, 32, 12) = 32
    # But for matrix multiplication, the rank is at least n*m*p / something
    # Let's try ranks from 32 up
    
    best_result = None
    
    for rank in range(32, 97):
        if time.time() > deadline - 20:  # Leave 20s for final checks
            break
        
        # Try a few random initializations
        for trial in range(3):
            if time.time() > deadline - 10:
                break
            
            np.random.seed(42 + rank * 100 + trial)
            
            # Random initialization
            U_init = np.random.randn(n * m, rank) * 0.1
            V_init = np.random.randn(m * p, rank) * 0.1
            W_init = np.random.randn(p * n, rank) * 0.1
            
            # Scale to make the product closer to 1
            # The product should be 1, so we need appropriate scaling
            # Let's try to scale so that the expected product is around 1
            scale = 1.0 / (rank ** 0.5)
            U_init = U_init * scale
            V_init = V_init * scale
            W_init = W_init * scale
            
            # Run ALS
            U_als, V_als, W_als = try_als(U_init, V_init, W_init, 
                                           max_iter=200, tol=1e-8,
                                           time_limit=deadline - 15)
            
            # Check if it's close to exact
            max_err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U_als[i*m+j, :], V_als[j*p+k, :] * W_als[k*n+i, :])
                        err = abs(val - 1.0)
                        if err > max_err:
                            max_err = err
            
            if max_err < 1e-4:
                # Close enough, try to round
                result = check_and_round(U_als, V_als, W_als)
                if result is not None:
                    U_r, V_r, W_r = result
                    if best_result is None or rank < best_result[0]:
                        best_result = (rank, U_r, V_r, W_r)
                        break  # Found a valid decomposition at this rank, move to next
        if best_result is not None and best_result[0] == rank:
            break
    
    if best_result is not None:
        rank, U_r, V_r, W_r = best_result
        return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
    
    # Fallback: return schoolbook
    U, V, W = schoolbook()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
