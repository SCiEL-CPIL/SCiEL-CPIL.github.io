"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]
import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Safety margin
    
    # Schoolbook decomposition
    rank_school = n * m * p  # 45
    U_fb = np.zeros((n*m, rank_school), dtype=np.float64)
    V_fb = np.zeros((m*p, rank_school), dtype=np.float64)
    W_fb = np.zeros((p*n, rank_school), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i*m+j, col] = 1.0
                V_fb[j*p+k, col] = 1.0
                W_fb[k*n+i, col] = 1.0
                col += 1
    
    # Try to find a lower rank decomposition using structured search
    # The tensor is T[i,j,k] = 1 for all i,j,k
    # We need: sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = 1
    
    # Try ranks from 44 down to some minimum
    # Use a more efficient verification
    
    def verify_fast(U, V, W):
        r = U.shape[1]
        U_r = U.reshape(n, m, r)
        V_r = V.reshape(m, p, r)
        W_r = W.reshape(p, n, r)
        result = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
        return np.allclose(result, np.ones((n,m,p)), atol=1e-8, rtol=1e-8)
    
    best_rank = rank_school
    best_result = None
    
    # Try to reduce rank by merging terms
    # For each pair of columns in the schoolbook, check if they can be merged
    # This is complex, so let's try a randomized approach with more structure
    
    # Try random search with small integer values for ranks 44 down to 30
    for rank in range(44, 29, -1):
        if time.time() > deadline:
            break
        
        # Try multiple random initializations
        for attempt in range(100):
            if time.time() > deadline:
                break
            
            # Use small integers
            U_try = np.random.randint(-2, 3, size=(n*m, rank)).astype(np.float64)
            V_try = np.random.randint(-2, 3, size=(m*p, rank)).astype(np.float64)
            W_try = np.random.randint(-2, 3, size=(p*n, rank)).astype(np.float64)
            
            if verify_fast(U_try, V_try, W_try):
                best_rank = rank
                best_result = {
                    "U": U_try.tolist(),
                    "V": V_try.tolist(),
                    "W": W_try.tolist()
                }
                break
        
        if best_result is not None:
            continue
    
    if best_result is not None:
        return best_result
    
    # Return schoolbook
    return {
        "U": U_fb.tolist(),
        "V": V_fb.tolist(),
        "W": W_fb.tolist()
    }
# EVOLVE-BLOCK-END
