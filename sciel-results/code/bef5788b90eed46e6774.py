"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-6):
    """Check if sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all i,j,k."""
    # Vectorized verification
    # Reshape to (n, m, p, rank)
    U_reshaped = U.reshape(n, m, 1, -1)
    V_reshaped = V.reshape(1, m, p, -1)
    W_reshaped = W.reshape(p, 1, n, -1)
    
    # Broadcast and sum
    tensor = U_reshaped * V_reshaped * W_reshaped
    sums = np.sum(tensor, axis=3)
    
    # Check if all sums are close to 1
    if not np.allclose(sums, 1.0, atol=tol):
        return False
    return True

def _als_vectorized(n, m, p, rank, max_iters=200, seed=0, deadline=None):
    """Vectorized ALS for matrix multiplication tensor."""
    rng = np.random.RandomState(seed)
    U = rng.randn(n * m, rank)
    V = rng.randn(m * p, rank)
    W = rng.randn(p * n, rank)
    
    best_err = float('inf')
    best_U, best_V, best_W = U, V, W
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline - 2.0:
            break
            
        # Update U
        # For each i, j, solve min || (V[j,:] * W[i,:])^T U[i*m+j,:] - 1_p ||
        # We can vectorize this by iterating over i and j, but using vectorized ops inside
        for i in range(n):
            for j in range(m):
                # A is p x rank
                # A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                # V[j*p : j*p+p, :] is (p, rank)
                # W[:, i] is (p, rank) -> need to align
                # W is (p*n, rank). W[k*n+i, :] for k in 0..p-1.
                # This is not a simple slice if n>1.
                # W indices: 0, n, 2n, ..., (p-1)n + i
                idx_W = np.arange(p) * n + i
                A = V[j*p : j*p+p, :] * W[idx_W, :]
                # Solve A^T x = 1_p
                # x = (A A^T)^{-1} A 1_p
                try:
                    # Using lstsq for stability
                    U[i*m+j, :] = np.linalg.lstsq(A.T, np.ones(p), rcond=None)[0]
                except:
                    pass
        
        # Update V
        for j in range(m):
            for k in range(p):
                # B is n x rank
                # B[i, :] = U[i*m+j, :] * W[k*n+i, :]
                # U indices: 0, m, ..., (n-1)m + j
                idx_U = np.arange(n) * m + j
                # W indices: k*n + 0, k*n + 1, ..., k*n + n-1
                idx_W = np.arange(n) + k*n
                B = U[idx_U, :] * W[idx_W, :]
                try:
                    V[j*p+k, :] = np.linalg.lstsq(B.T, np.ones(n), rcond=None)[0]
                except:
                    pass
        
        # Update W
        for k in range(p):
            for i in range(n):
                # C is m x rank
                # C[j, :] = U[i*m+j, :] * V[j*p+k, :]
                # U indices: i*m + 0, i*m + 1, ..., i*m + m-1
                idx_U = np.arange(m) + i*m
                # V indices: 0*p+k, 1*p+k, ..., (m-1)p+k
                idx_V = np.arange(m) * p + k
                C = U[idx_U, :] * V[idx_V, :]
                try:
                    W[k*n+i, :] = np.linalg.lstsq(C.T, np.ones(m), rcond=None)[0]
                except:
                    pass
        
        # Compute error
        U_reshaped = U.reshape(n, m, 1, -1)
        V_reshaped = V.reshape(1, m, p, -1)
        W_reshaped = W.reshape(p, 1, n, -1)
        tensor = U_reshaped * V_reshaped * W_reshaped
        sums = np.sum(tensor, axis=3)
        err = np.sqrt(np.sum((sums - 1.0) ** 2))
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            if best_err < 1e-8:
                break
                
    if best_err < 1e-6:
        return best_U, best_V, best_W, best_err
    return None, None, None, best_err

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # Try to find rank-23 decomposition for 3x3x3
    # If found, construct rank-46 for 3x3x6
    target_ranks_3x3 = [23, 24, 25, 26, 27]
    
    found_better = False
    for r3 in target_ranks_3x3:
        if time.time() > deadline - 10.0:
            break
        for seed in range(10):
            if time.time() > deadline - 10.0:
                break
            U3, V3, W3, err = _als_vectorized(3, 3, 3, r3, max_iters=500, seed=seed, deadline=deadline)
            if U3 is not None and err < 1e-6:
                # Construct 3x3x6 factors
                total_rank = 2 * r3
                U = np.zeros((n*m, total_rank))
                V = np.zeros((m*p, total_rank))
                W = np.zeros((p*n, total_rank))
                
                # Copy slice 0 (k=0,1,2)
                U[:, 0:r3] = U3
                # V3 is 3x3. V is 3x6.
                # V[j, k] for k in 0,1,2
                V[:, 0:3] = V3
                # W3 is 3x3. W is 6x3.
                # W[k, i] for k in 0,1,2
                W[0:3, :] = W3
                
                # Copy slice 1 (k=3,4,5)
                U[:, r3:total_rank] = U3
                V[:, 3:6] = V3
                W[3:6, :] = W3
                
                # Verify
                if _verify(U, V, W, n, m, p, tol=1e-5):
                    best_U, best_V, best_W = U, V, W
                    found_better = True
                    break
        if found_better:
            break
            
    # Convert to lists
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
