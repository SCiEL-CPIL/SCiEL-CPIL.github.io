"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition as fallback
    def schoolbook():
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
        return {"U": U, "V": V, "W": W}
    
    # Verify a decomposition is correct
    def verify(U, V, W):
        n, m, p = DIMENSIONS
        rank = len(U[0]) if U else 0
        # Convert to numpy for efficient check
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        # The tensor T[i,j,k] = 1 if i,j,k are the indices in A[i,j]*B[j,k]
        # Decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # Check a few random (i,j,k) tuples
        random.seed(42)
        for _ in range(100):
            i = random.randint(0, n-1)
            j = random.randint(0, m-1)
            k = random.randint(0, p-1)
            expected = 1.0
            computed = 0.0
            for r in range(rank):
                computed += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
            if abs(computed - expected) > 1e-9:
                return False
        return True
    
    # Try to find a lower rank decomposition using alternating least squares
    def als_search(rank, max_iters=200):
        """Try to find a rank-rank decomposition using ALS."""
        n, m, p = DIMENSIONS
        # Initialize with random values
        random.seed(0)
        np.random.seed(0)
        U = np.random.randn(n*m, rank) * 0.1
        V = np.random.randn(m*p, rank) * 0.1
        W = np.random.randn(p*n, rank) * 0.1
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                return None
            
            # Update U: fix V and W, solve for U
            # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            # For each (i,j), U[i*m+j, :] should satisfy:
            # sum_r U[i*m+j, r] * (V[j*p+k, r] * W[k*n+i, r]) = 1 for all k
            # Let B[i,j,k,r] = V[j*p+k, r] * W[k*n+i, r]
            # Then U[i*m+j, :] = B[i,j,k,:] * 1 for all k (consensus)
            
            U_new = np.zeros_like(U)
            for i in range(n):
                for j in range(m):
                    # Collect all k equations
                    # For each r, we need U[i*m+j, r] such that sum over k of ... 
                    # Actually, let's think of it differently.
                    # The constraint is: for all i,j,k: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
                    # This is an overdetermined system for U[i*m+j, :].
                    # Let A_vec be a p x rank matrix where A_vec[k, r] = V[j*p+k, r] * W[k*n+i, r]
                    # Then we want A_vec @ U[i*m+j, :]^T = ones(p)
                    # Least squares: U[i*m+j, :]^T = pinv(A_vec) @ ones(p)
                    A_mat = np.zeros((p, rank))
                    for k in range(p):
                        for r in range(rank):
                            A_mat[k, r] = V[j*p+k, r] * W[k*n+i, r]
                    try:
                        sol = np.linalg.lstsq(A_mat, np.ones(p), rcond=None)[0]
                        U_new[i*m+j, :] = sol
                    except:
                        U_new[i*m+j, :] = U[i*m+j, :]
            
            # Update V: fix U and W
            V_new = np.zeros_like(V)
            for j in range(m):
                for k in range(p):
                    A_mat = np.zeros((n, rank))
                    for i in range(n):
                        for r in range(rank):
                            A_mat[i, r] = U[i*m+j, r] * W[k*n+i, r]
                    try:
                        sol = np.linalg.lstsq(A_mat, np.ones(n), rcond=None)[0]
                        V_new[j*p+k, :] = sol
                    except:
                        V_new[j*p+k, :] = V[j*p+k, :]
            
            # Update W: fix U and V
            W_new = np.zeros_like(W)
            for k in range(p):
                for i in range(n):
                    A_mat = np.zeros((m, rank))
                    for j in range(m):
                        for r in range(rank):
                            A_mat[j, r] = U[i*m+j, r] * V[j*p+k, r]
                    try:
                        sol = np.linalg.lstsq(A_mat, np.ones(m), rcond=None)[0]
                        W_new[k*n+i, :] = sol
                    except:
                        W_new[k*n+i, :] = W[k*n+i, :]
            
            U, V, W = U_new, V_new, W_new
        
        # Verify
        U_list = [[float(U[i*m+j, r]) for r in range(rank)] for i in range(n) for j in range(m)]
        V_list = [[float(V[j*p+k, r]) for r in range(rank)] for j in range(m) for k in range(p)]
        W_list = [[float(W[k*n+i, r]) for r in range(rank)] for k in range(p) for i in range(n)]
        
        if verify(U_list, V_list, W_list):
            return {"U": U_list, "V": V_list, "W": W_list}
        return None
    
    # Try ranks from low to high
    # Known bounds: rank(3x3x6) >= 23 (from literature), schoolbook is 54
    # Let's try a range
    for target_rank in [23, 24, 25, 26, 27, 28, 30, 32, 35, 40, 45, 50]:
        if time.time() > deadline - 10:
            break
        result = als_search(target_rank, max_iters=100)
        if result is not None:
            return result
    
    # Fall back to schoolbook
    return schoolbook()
# EVOLVE-BLOCK-END
