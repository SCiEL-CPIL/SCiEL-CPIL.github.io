import time
import numpy as np

def _search_rank_22_4x4x4(time_limit=30):
    """Search for rank-22 decomposition of 4x4x4 matrix multiplication."""
    start = time.time()
    n = 4
    target_rank = 22
    
    # Target: T[i,j,k] = 1 if i==k else 0
    # Equivalent: for each j, sum_r v_r[j] * (u_r * w_r^T) = I_n
    
    # Random initialization
    for attempt in range(200):
        if time.time() - start > time_limit:
            break
        
        # Initialize randomly
        U = np.random.randn(n, target_rank)
        V = np.random.randn(n, target_rank)  # V[j, r]
        W = np.random.randn(n, target_rank)
        
        # Check: for each j, sum_r V[j,r] * U[:,r] * W[:,r]^T should equal I_n
        # Compute residual
        max_err = 0
        for j in range(n):
            M = (U * V[j, :]) @ W.T  # sum_r V[j,r] * U[:,r] * W[:,r]^T
            err = np.max(np.abs(M - np.eye(n)))
            max_err = max(max_err, err)
        
        if max_err < 1e-10:
            # Found it! Convert to the required format
            U_out = np.zeros((n*n, target_rank))
            V_out = np.zeros((n*n, target_rank))
            W_out = np.zeros((n*n, target_rank))
            
            for i in range(n):
                for j in range(n):
                    U_out[i*n+j, :] = U[i, :] * V[j, :]
            for j in range(n):
                for k in range(n):
                ...
