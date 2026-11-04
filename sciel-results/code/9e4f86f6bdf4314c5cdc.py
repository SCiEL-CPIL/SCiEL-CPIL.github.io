"""Program to find rank-minimal exact tensor decomposition for 3x3 * 3x6 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin

    n, m, p = DIMENSIONS
    # A is n x m, B is m x p, C is n x p
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)  <-- Note: W index is k*n + i (output C is n x p, but trace convention says W[k*n+i, r])
    
    # The tensor equation is:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i, j, k.
    
    # Strategy: Exploit rectangular structure
    # C = A * B where A is 3x3, B is 3x6
    # Split B into B_left (3x3) and B_right (3x3)
    # C = A*B_left + A*B_right
    # This is two independent 3x3*3x3 multiplications
    # Rank of 3x3*3x3 is 23 (Landsberg bound)
    # Total rank should be 23 + 23 = 46
    
    # First, create schoolbook fallback
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1.0
                V_school[j * p + k, col] = 1.0
                W_school[k * n + i, col] = 1.0
                col += 1
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school
    
    # Try to find rank-46 decomposition using Kronecker product structure
    # For 3x3*3x3, use explicit rank-23 decomposition
    # We'll use a numerical optimization approach with good initialization
    
    target_rank = 46
    
    # Initialize with structure: first 23 ranks for left half, next 23 for right half
    np.random.seed(42)
    
    # Create initial guess based on Strassen-like structure
    U = np.zeros((n * m, target_rank))
    V = np.zeros((m * p, target_rank))
    W = np.zeros((p * n, target_rank))
    
    # For the left half (k=0,1,2), use one set of 23 rank-1 terms
    # For the right half (k=3,4,5), use another set of 23 rank-1 terms
    
    # Initialize with random values but with structure
    for r in range(23):
        # Left half terms
        U[:, r] = np.random.randn(n * m) * 0.1
        V[:, r] = np.random.randn(m * p) * 0.1
        # Only affect left half of W
        W[:3*n, r] = np.random.randn(3*n) * 0.1
        W[3*n:, r] = 0.0
        
        # Right half terms
        r2 = r + 23
        U[:, r2] = np.random.randn(n * m) * 0.1
        V[:, r2] = np.random.randn(m * p) * 0.1
        # Only affect right half of W
        W[:3*n, r2] = 0.0
        W[3*n:, r2] = np.random.randn(3*n) * 0.1
    
    # Alternating least squares optimization
    for iteration in range(2000):
        if time.time() > deadline - 10:
            break
        
        # Update U
        for i in range(n):
            for j in range(m):
                # For each (i,j), solve: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all k
                # M[k, r] = V[j*p+k, r] * W[k*n+i, r]
                M = np.outer(V[j*p:j*p+p, :], W[:, :])  # shape (p, target_rank)
                # Actually W is indexed by k*n+i, so for fixed i, we need W[k*n+i, r] for k=0..p-1
                M = np.array([V[j*p+k, r] * W[k*n+i, r] for k in range(p) for r in range(target_rank)]).reshape(p, target_rank)
                try:
                    sol, residuals, rank_m, sv = np.linalg.lstsq(M.T, np.ones(p), rcond=None)
                    U[i * m + j, :] = sol
                except:
                    pass
        
        # Update V
        for j in range(m):
            for k in range(p):
                # For each (j,k), solve: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i
                M = np.array([U[i*m+j, r] * W[k*n+i, r] for i in range(n) for r in range(target_rank)]).reshape(n, target_rank)
                try:
                    sol, residuals, rank_m, sv = np.linalg.lstsq(M.T, np.ones(n), rcond=None)
                    V[j * p + k, :] = sol
                except:
                    pass
        
        # Update W
        for k in range(p):
            for i in range(n):
                # For each (k,i), solve: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all j
                M = np.array([U[i*m+j, r] * V[j*p+k, r] for j in range(m) for r in range(target_rank)]).reshape(m, target_rank)
                try:
                    sol, residuals, rank_m, sv = np.linalg.lstsq(M.T, np.ones(m), rcond=None)
                    W[k * n + i, :] = sol
                except:
                    pass
        
        # Check convergence every 20 iterations
        if iteration % 20 == 0:
            error = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(target_rank))
                        error += (val - 1.0) ** 2
            if error < 1e-8:
                break
            if iteration % 100 == 0:
                pass  # Could print for debugging
    
    # Check if rank-46 works
    error = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(target_rank))
                error += (val - 1.0) ** 2
    
    if error < 1e-6:
        best_U, best_V, best_W = U, V, W
        best_rank = target_rank
    else:
        # Try with more iterations or different initialization
        np.random.seed(123)
        U2 = np.random.randn(n * m, target_rank)
        V2 = np.random.randn(m * p, target_rank)
        W2 = np.random.randn(p * n, target_rank)
        
        for iteration in range(2000):
            if time.time() > deadline - 5:
                break
            
            # Update U
            for i in range(n):
                for j in range(m):
                    M = np.array([V2[j*p+k, r] * W2[k*n+i, r] for k in range(p) for r in range(target_rank)]).reshape(p, target_rank)
                    try:
                        sol, residuals, rank_m, sv = np.linalg.lstsq(M.T, np.ones(p), rcond=None)
                        U2[i * m + j, :] = sol
                    except:
                        pass
            
            # Update V
            for j in range(m):
                for k in range(p):
                    M = np.array([U2[i*m+j, r] * W2[k*n+i, r] for i in range(n) for r in range(target_rank)]).reshape(n, target_rank)
                    try:
                        sol, residuals, rank_m, sv = np.linalg.lstsq(M.T, np.ones(n), rcond=None)
                        V2[j * p + k, :] = sol
                    except:
                        pass
            
            # Update W
            for k in range(p):
                for i in range(n):
                    M = np.array([U2[i*m+j, r] * V2[j*p+k, r] for j in range(m) for r in range(target_rank)]).reshape(m, target_rank)
                    try:
                        sol, residuals, rank_m, sv = np.linalg.lstsq(M.T, np.ones(m), rcond=None)
                        W2[k * n + i, :] = sol
                    except:
                        pass
            
            if iteration % 50 == 0:
                error2 = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = sum(U2[i*m+j, r] * V2[j*p+k, r] * W2[k*n+i, r] for r in range(target_rank))
                            error2 += (val - 1.0) ** 2
                if error2 < 1e-8:
                    break
        
        error2 = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U2[i*m+j, r] * V2[j*p+k, r] * W2[k*n+i, r] for r in range(target_rank))
                    error2 += (val - 1.0) ** 2
        
        if error2 < 1e-6:
            best_U, best_V, best_W = U2, V2, W2
            best_rank = target_rank
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
