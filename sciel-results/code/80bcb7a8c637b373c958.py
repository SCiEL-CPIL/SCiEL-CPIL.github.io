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
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(s - 1.0) > tol:
                    return False
    return True

def _als_3x3x3(rank, max_iters=200, seed=0, deadline=None):
    """Find rank-r decomposition for 3x3x3 multiplication tensor using optimized ALS."""
    rng = np.random.RandomState(seed)
    n = m = p = 3
    U = rng.randn(n*m, rank)
    V = rng.randn(m*p, rank)
    W = rng.randn(p*n, rank)
    
    best_err = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline - 1.0:
            break
            
        # Update U: for each (i,j), solve A^T u = 1 where A[k,:] = V[j*3+k,:] * W[k*3+i,:]
        for i in range(n):
            for j in range(m):
                A = np.zeros((p, rank))
                for k in range(p):
                    A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                try:
                    U[i*m+j, :] = np.linalg.lstsq(A.T, np.ones(p), rcond=None)[0]
                except:
                    pass
        
        # Update V: for each (j,k), solve B^T v = 1 where B[i,:] = U[i*3+j,:] * W[k*3+i,:]
        for j in range(m):
            for k in range(p):
                B = np.zeros((n, rank))
                for i in range(n):
                    B[i, :] = U[i*m+j, :] * W[k*n+i, :]
                try:
                    V[j*p+k, :] = np.linalg.lstsq(B.T, np.ones(n), rcond=None)[0]
                except:
                    pass
        
        # Update W: for each (k,i), solve C^T w = 1 where C[j,:] = U[i*3+j,:] * V[j*3+k,:]
        for k in range(p):
            for i in range(n):
                C = np.zeros((m, rank))
                for j in range(m):
                    C[j, :] = U[i*m+j, :] * V[j*p+k, :]
                try:
                    W[k*n+i, :] = np.linalg.lstsq(C.T, np.ones(m), rcond=None)[0]
                except:
                    pass
        
        # Compute error
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    err += (s - 1.0) ** 2
        err = np.sqrt(err)
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            if best_err < 1e-10:
                break
                
    if best_err < 1e-6:
        return best_U, best_V, best_W, best_err
    return None, None, None, best_err

def _combine_3x3x3_to_3x3x6(U3, V3, W3):
    """Combine two 3x3x3 rank-r decompositions into a 3x3x6 rank-2r decomposition."""
    n, m, p = 3, 3, 6
    r3 = U3.shape[1]
    total_rank = 2 * r3
    
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    # Slice 0: k = 0,1,2
    U[:, 0:r3] = U3
    for j in range(m):
        for k in range(3):
            V[j*p+k, 0:r3] = V3[j*3+k, 0:r3]
    for k in range(3):
        for i in range(n):
            W[k*n+i, 0:r3] = W3[k*3+i, 0:r3]
            
    # Slice 1: k = 3,4,5
    U[:, r3:total_rank] = U3
    for j in range(m):
        for k in range(3):
            V[j*p+(3+k), r3:total_rank] = V3[j*3+k, 0:r3]
    for k in range(3):
        for i in range(n):
            W[(3+k)*n+i, r3:total_rank] = W3[k*3+i, 0:r3]
            
    return U, V, W

def _als_general(n, m, p, rank, max_iters=500, seed=0, deadline=None):
    """Find rank-r decomposition for n x m x p multiplication tensor."""
    rng = np.random.RandomState(seed)
    U = rng.randn(n*m, rank)
    V = rng.randn(m*p, rank)
    W = rng.randn(p*n, rank)
    
    best_err = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline - 2.0:
            break
            
        # Update U
        for i in range(n):
            for j in range(m):
                A = np.zeros((p, rank))
                for k in range(p):
                    A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                try:
                    U[i*m+j, :] = np.linalg.lstsq(A.T, np.ones(p), rcond=None)[0]
                except:
                    pass
        
        # Update V
        for j in range(m):
            for k in range(p):
                B = np.zeros((n, rank))
                for i in range(n):
                    B[i, :] = U[i*m+j, :] * W[k*n+i, :]
                try:
                    V[j*p+k, :] = np.linalg.lstsq(B.T, np.ones(n), rcond=None)[0]
                except:
                    pass
        
        # Update W
        for k in range(p):
            for i in range(n):
                C = np.zeros((m, rank))
                for j in range(m):
                    C[j, :] = U[i*m+j, :] * V[j*p+k, :]
                try:
                    W[k*n+i, :] = np.linalg.lstsq(C.T, np.ones(m), rcond=None)[0]
                except:
                    pass
        
        # Compute error
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    err += (s - 1.0) ** 2
        err = np.sqrt(err)
        
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
    best_U, best_V, best_W, best_rank = U_sb, V_sb, W_sb, n*m*p
    
    # Strategy 1: Find a low rank decomposition for 3x3x3 and double it
    # This gives rank 2*r3 for 3x3x6
    # Try r3 from 26 down to 23 with more seeds and iterations
    for r3 in range(26, 22, -1):
        if time.time() > deadline - 20.0:
            break
        found = False
        for seed in range(20):
            if time.time() > deadline - 20.0:
                break
            U3, V3, W3, err = _als_3x3x3(r3, max_iters=150, seed=seed, deadline=deadline)
            if U3 is not None and err < 1e-6:
                # Construct 3x3x6 factors
                total_rank = 2 * r3
                if total_rank < best_rank:
                    U, V, W = _combine_3x3x3_to_3x3x6(U3, V3, W3)
                    
                    # Verify
                    if _verify(U, V, W, n, m, p, tol=1e-5):
                        best_U, best_V, best_W, best_rank = U, V, W, total_rank
                        found = True
                        break
        if found:
            break

    # Strategy 2: Direct ALS for 3x3x6
    # Try ranks from best_rank - 1 down to 46
    for R in range(best_rank - 1, 45, -1):
        if time.time() > deadline - 5.0:
            break
        found = False
        for seed in range(5):
            if time.time() > deadline - 5.0:
                break
            U, V, W, err = _als_general(n, m, p, R, max_iters=50, seed=seed, deadline=deadline)
            if U is not None and err < 1e-6:
                if _verify(U, V, W, n, m, p, tol=1e-5):
                    best_U, best_V, best_W, best_rank = U, V, W, R
                    found = True
                    break
        if found:
            break

    # Convert to lists
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
