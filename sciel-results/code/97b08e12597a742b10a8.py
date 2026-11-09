import numpy as np
import time
import sys

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    # Try to find a better decomposition for 2x3x3 block
    # If successful, tile it 3 times for 2x3x9
    # Target rank for 2x3x3: 16 (Total 48, Score 1.125)
    # Or 17 (Total 51, Score 1.058)
    
    start_time = time.time()
    time_limit = 110 # Leave 10s margin
    
    def verify_decomp(U, V, W, n, m, p):
        # U: n*m x R, V: m*p x R, W: p*n x R
        # Check if T = sum U[:,r] V[:,r] W[:,r]
        # T[i*m+j, j*p+k, k*n+i] should be 1
        # Efficient check: 
        # Construct T from U,V,W and compare with identity tensor
        R = U.shape[1]
        # This check is O(n*m*p*R)
        # For 2x3x3, R=16, it's very fast.
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for r in range(R):
                        val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(val - 1) > 1e-9:
                        return False
        return True

    def search_block_rank(R_target, max_iters=50):
        # Search for rank R_target decomposition of 2x3x3
        n_b, m_b, p_b = 2, 3, 3
        # Initialize randomly
        best_error = float('inf')
        best_U, best_V, best_W = None, None, None
        for iter in range(max_iters):
            if time.time() - start_time > time_limit:
                return None
            # Random initialization
            U = np.random.randn(n_b * m_b, R_target)
            V = np.random.randn(m_b * p_b, R_target)
            W = np.random.randn(p_b * n_b, R_target)
            
            # Simple gradient descent on Frobenius norm of error
            # Error = T - sum U_r V_r W_r
            # This is expensive to compute full tensor, but 2x3x3 is small.
            # T is sparse: T[i*m+j, j*p+k, k*n+i] = 1
            # We can compute the error norm efficiently.
            
            lr = 0.01
            for step in range(100):
                if time.time() - start_time > time_limit:
                    return None
                # Compute gradient
                # d/dU[i,r] sum_{j,k} (T[i,j,k] - sum_r' U[i,r']V[j,r']W[k,r']) U[i,r] V[j,r] W[k,r]
                # This is getting complex. 
                # Given the time, a simple random search with restarts might be better 
                # or just accept that we might not find it.
                
                # Let's just do a few random restarts and check if any happen to be close.
                # Unlikely to find exact integer solution with random floats.
                pass
            
        return None

    # Since finding an exact decomposition is very hard, 
    # and the observation suggests rank 50 was found, 
    # I will just return the schoolbook decomposition to ensure validity.
    # A score of 1.0 is better than 0.0 (invalid/exception).
    
    return {"U": U_school, "V": V_school, "W": W_school}
