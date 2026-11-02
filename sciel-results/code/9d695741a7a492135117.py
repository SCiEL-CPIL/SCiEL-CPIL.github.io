"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    import time
    import random
    import numpy as np
    from fractions import Fraction

    deadline = time.time() + 115  # safety margin before 120s limit

    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n

    def _verify(U, V, W):
        """Verify that U, V, W form a valid tensor decomposition."""
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(s - 1) > 1e-10:
                        return False
        return True

    def _schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W, "rank": rank}

    def _als_decomposition(rank, max_iter=200, tol=1e-12, seed=42):
        """
        Alternating Least Squares to find a rank-r decomposition of the matmul tensor.
        The tensor T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
        We want U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] summed over c to equal 1.
        """
        random.seed(seed)
        np.random.seed(seed)
        
        # Initialize
        U = np.random.randn(nm, rank) * 0.1
        V = np.random.randn(mp, rank) * 0.1
        W = np.random.randn(pn, rank) * 0.1
        
        # Precompute index mappings
        # For each (i,j,k), we need sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
        
        for iteration in range(max_iter):
            if time.time() > deadline - 5:
                break
                
            # Update U
            for i in range(n):
                for j in range(m):
                    # We need: sum_k sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = p (sum over k)
                    # Actually, for each (i,j,k), the constraint is separate.
                    # ALS approach: fix V and W, solve for U[i*m+j,:].
                    # The equations are: for each k, sum_c U[i*m+j,c] * (V[j*p+k,c] * W[k*n+i,c]) = 1
                    # Let B[k, c] = V[j*p+k, c] * W[k*n+i, c]
                    # Then for each k: U[i*m+j,:] @ B[k,:] = 1
                    # This is a linear system: B @ U[i*m+j,:]^T = 1 (vector of ones of length p)
                    
                    B = np.zeros((p, rank))
                    for k in range(p):
                        B[k, :] = V[j * p + k, :] * W[k * n + i, :]
                    
                    # Solve B @ x = ones(p)
                    try:
                        x, residuals, rank_b, sv = np.linalg.lstsq(B, np.ones(p), rcond=None)
                        U[i * m + j, :] = x
                    except:
                        pass
            
            # Update V
            for j in range(m):
                for k in range(p):
                    # For each i: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
                    # Let A[i, c] = U[i*m+j, c] * W[k*n+i, c]
                    # Then for each i: A[i,:] @ V[j*p+k,:]^T = 1
                    # A @ V[j*p+k,:]^T = ones(n)
                    
                    A = np.zeros((n, rank))
                    for i in range(n):
                        A[i, :] = U[i * m + j, :] * W[k * n + i, :]
                    
                    try:
                        x, residuals, rank_a, sv = np.linalg.lstsq(A, np.ones(n), rcond=None)
                        V[j * p + k, :] = x
                    except:
                        pass
            
            # Update W
            for k in range(p):
                for i in range(n):
                    # For each j: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
                    # Let C[j, c] = U[i*m+j, c] * V[j*p+k, c]
                    # Then for each j: C[j,:] @ W[k*n+i,:]^T = 1
                    # C @ W[k*n+i,:]^T = ones(m)
                    
                    C = np.zeros((m, rank))
                    for j in range(m):
                        C[j, :] = U[i * m + j, :] * V[j * p + k, :]
                    
                    try:
                        x, residuals, rank_c, sv = np.linalg.lstsq(C, np.ones(m), rcond=None)
                        W[k * n + i, :] = x
                    except:
                        pass
            
            # Check convergence
            if iteration % 10 == 0:
                err = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = np.sum(U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :])
                            err += (s - 1) ** 2
                if err < tol:
                    break
        
        # Convert to lists
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        
        if _verify(U_list, V_list, W_list):
            return {"U": U_list, "V": V_list, "W": W_list, "rank": rank}
        
        return None

    def _try_rank(rank, num_trials=3):
        """Try multiple random initializations for a given rank."""
        for trial in range(num_trials):
            if time.time() > deadline - 10:
                break
            result = _als_decomposition(rank, max_iter=100, seed=42 + trial)
            if result is not None:
                return result
        return None

    # Start with schoolbook
    best = _schoolbook()
    
    # Try to find lower rank decompositions
    # For 3x3x8, schoolbook is 72. Theoretical lower bound is at least max(9, 24, 24) = 24
    # Let's try ranks from 60 down to 30
    
    for rank in range(60, 25, -1):
        if time.time() > deadline - 15:
            break
        
        result = _try_rank(rank)
        if result is not None:
            best = result
            # Don't break immediately; try to go lower
            continue
    
    return {"U": best["U"], "V": best["V"], "W": best["W"]}
# EVOLVE-BLOCK-END
