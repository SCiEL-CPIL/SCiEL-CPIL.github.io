"""
Refined program for matrix multiplication tensor decomposition.
Task: 3x3 by 3x5 matrix multiplication.
Goal: Find rank-minimal decomposition.
Strategy: Structured search using linear algebra to find lower-rank decompositions.
"""
import time
import numpy as np

DIMENSIONS = [3, 3, 5]

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    deadline = start_time + 140
    
    def get_trivial_decomposition():
        """
        Constructs the standard schoolbook tensor decomposition.
        Rank = n * m * p
        """
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, idx] = 1.0
                    V[j * p + k, idx] = 1.0
                    W[k * n + i, idx] = 1.0
                    idx += 1
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    def verify_decomposition(U, V, W, tol=1e-8):
        """
        Verify that the decomposition correctly represents matrix multiplication.
        U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
        The tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        should equal 1 if i,j,k form a valid product term, 0 otherwise.
        
        Actually, for matrix multiplication C = A * B where A is n×m and B is m×p:
        C[i,k] = sum_j A[i,j] * B[j,k]
        
        The multiplication tensor has T[i,j,k] = 1 for all valid (i,j,k) indices.
        A rank-1 decomposition term contributes U[i,j] * V[j,k] * W[k,i] to T[i,j,k].
        
        So we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
        """
        rank = U.shape[1]
        # Check all (i, j, k) combinations
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def try_rank_decomposition(rank, max_iter=100, seed=42):
        """
        Try to find a rank-r decomposition using iterative refinement.
        Uses the approach of initializing randomly and then using gradient-like updates.
        """
        np.random.seed(seed)
        
        # Initialize randomly
        U = np.random.randn(n * m, rank) * 0.5
        V = np.random.randn(m * p, rank) * 0.5
        W = np.random.randn(p * n, rank) * 0.5
        
        # We want: for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        # This is a system of n*m*p equations in rank*3*(n*m + m*p + p*n) unknowns (but structured)
        
        # Use iterative approach: fix two, solve for third
        best_err = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for iteration in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            # Try cycling through which matrix to update
            for which in range(3):
                if time.time() > deadline - 5:
                    break
                
                # Compute residuals for each (i,j,k)
                residuals = np.zeros((n, m, p))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for r in range(rank):
                                val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                            residuals[i, j, k] = val - 1.0
                
                err = np.sum(residuals ** 2)
                if err < best_err:
                    best_err = err
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
                if err < 1e-10:
                    return best_U, best_V, best_W, True
                
                # Update one matrix at a time using linear least squares
                if which == 0:  # Update U
                    # For each (i,j), we have: sum_r U[i*m+j, r] * (V[j*p+k, r] * W[k*n+i, r]) = 1
                    # This is: U[i*m+j, :] @ diag(V[j*p+k,:] * W[k*n+i,:]) = 1 for all k
                    # Stack all k: A[i*m+j, k, r] = V[j*p+k, r] * W[k*n+i, r]
                    # We want U[i*m+j, r] such that sum_r U[i*m+j, r] * A[i*m+j, k, r] = 1 for all k
                    # This is a linear system: (A @ U[i*m+j, :].T) = ones(p)
                    for i in range(n):
                        for j in range(m):
                            # Build system: for each k, sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
                            # Let x = U[i*m+j, :], then sum_r x[r] * (V[j*p+k, r] * W[k*n+i, r]) = 1
                            # Matrix form: B @ x = ones(p) where B[k, r] = V[j*p+k, r] * W[k*n+i, r]
                            B = np.zeros((p, rank))
                            for k in range(p):
                                B[k, :] = V[j * p + k, :] * W[k * n + i, :]
                            # Solve B @ x = ones(p) in least squares sense
                            try:
                                x, _, _, _ = np.linalg.lstsq(B, np.ones(p), rcond=None)
                                U[i * m + j, :] = x
                            except:
                                pass
                
                elif which == 1:  # Update V
                    for j in range(m):
                        for k in range(p):
                            # For each (i), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
                            # Let x = V[j*p+k, :], then sum_r (U[i*m+j, r] * W[k*n+i, r]) * x[r] = 1
                            B = np.zeros((n, rank))
                            for i in range(n):
                                B[i, :] = U[i * m + j, :] * W[k * n + i, :]
                            try:
                                x, _, _, _ = np.linalg.lstsq(B, np.ones(n), rcond=None)
                                V[j * p + k, :] = x
                            except:
                                pass
                
                elif which == 2:  # Update W
                    for k in range(p):
                        for i in range(n):
                            # For each (j), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
                            # Let x = W[k*n+i, :], then sum_r (U[i*m+j, r] * V[j*p+k, r]) * x[r] = 1
                            B = np.zeros((m, rank))
                            for j in range(m):
                                B[j, :] = U[i * m + j, :] * V[j * p + k, :]
                            try:
                                x, _, _, _ = np.linalg.lstsq(B, np.ones(m), rcond=None)
                                W[k * n + i, :] = x
                            except:
                                pass
        
        # Final check
        if best_err < 1e-8:
            return best_U, best_V, best_W, True
        return None, None, None, False
    
    # First, try to find a lower rank decomposition
    # Known: trivial rank is 45. Try to find something lower.
    # Start with a few candidate ranks
    best_result = None
    best_rank = n * m * p  # 45
    
    # Try ranks from a reasonable lower bound up
    # Theoretical lower bound for matmul is max(n*m, m*p, p*n) = max(9, 15, 15) = 15
    # But practical lower bounds are higher. Try a range.
    candidate_ranks = [20, 22, 24, 25, 26, 28, 30, 32, 35, 40, 44]
    
    for rank in candidate_ranks:
        if time.time() > deadline - 10:
            break
        
        # Try multiple random seeds
        found = False
        for seed in range(10):
            if time.time() > deadline - 10:
                break
            
            U, V, W, success = try_rank_decomposition(rank, max_iter=50, seed=seed)
            if success:
                # Verify
                if verify_decomposition(U, V, W, tol=1e-8):
                    best_result = (U, V, W)
                    best_rank = rank
                    found = True
                    break
        
        if found:
            break  # Found a valid decomposition, stop
    
    # If we found something better than trivial, return it
    if best_result is not None:
        U, V, W = best_result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Otherwise, return the trivial decomposition
    return get_trivial_decomposition()
