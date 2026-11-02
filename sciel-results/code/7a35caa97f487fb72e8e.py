"""Matrix multiplication 4,4,6 - rank-minimal exact tensor decomposition search."""
import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U,V,W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    # Build the multiplication tensor: T[i,j,k] = 1 if output (i,k) depends on A[i,j]*B[j,k]
    # U shape: (n*m, rank), V shape: (m*p, rank), W shape: (p*n, rank)
    # The decomposition says: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We need to check that this equals the identity tensor for matrix multiplication.
    # The matrix multiplication tensor has T[i,j,k] = 1 for all valid (i,j,k) since
    # C[i,k] = sum_j A[i,j]*B[j,k].
    
    # Actually the tensor for matrix multiplication is:
    # T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # because C[i,k] = sum_j A[i,j] * B[j,k]
    
    # So we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _verify_decomposition_fast(U, V, W, n, m, p, tol=1e-9):
    """Faster verification using numpy."""
    rank = len(U[0]) if U else 0
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    
    # For each (i,j,k), check sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] == 1
    # We can do this more efficiently:
    # For fixed i,k: sum_j ... no, we need all triples.
    
    # Reshape approach:
    # U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    # For each r, the outer product U[:,r] (n*m) x V[:,r] (m*p) x W[:,r] (p*n)
    # We want sum_r of this to be all ones.
    
    # Let's build it as: for each r, create a 3D tensor of shape (n,m,p)
    # U_r has shape (n*m,), reshape to (n,m)
    # V_r has shape (m*p,), reshape to (m,p)  
    # W_r has shape (p*n,), reshape to (p,n)
    # The contribution of rank-r term to T[i,j,k] is U_r[i,j] * V_r[j,k] * W_r[k,i]
    
    total = np.zeros((n, m, p), dtype=np.float64)
    for r in range(rank):
        Ur = Ua[:, r].reshape(n, m)
        Vr = Va[:, r].reshape(m, p)
        Wr = Wa[:, r].reshape(p, n)
        # T[i,j,k] += Ur[i,j] * Vr[j,k] * Wr[k,i]
        # This is like a 3-way product
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total[i,j,k] += Ur[i,j] * Vr[j,k] * Wr[k,i]
    
    return np.allclose(total, 1.0, atol=tol)

def _verify_decomposition_vectorized(U, V, W, n, m, p, tol=1e-9):
    """Even faster verification."""
    rank = len(U[0]) if U else 0
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    
    # For each (i,j,k), we need sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # Let's index: for fixed (i,j), U[i*m+j, :] is a vector of length rank
    # For fixed (j,k), V[j*p+k, :] is a vector of length rank
    # For fixed (k,i), W[k*n+i, :] is a vector of length rank
    # The dot product of these three... no, it's element-wise product then sum.
    
    # Actually: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # This is not a standard dot product of three vectors.
    # It's the sum of element-wise products.
    
    # Let me think: for fixed i, j, k:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # We can compute this as: for each pair (i,j), let a = U[i*m+j, :]
    # for each (j,k), let b = V[j*p+k, :]
    # for each (k,i), let c = W[k*n+i, :]
    # result[i,j,k] = sum_r a[r] * b[r] * c[r]
    
    # This is like a 3-tensor contraction.
    # Let me just do it with loops over i,j,k but vectorize over r.
    
    results = np.ones((n, m, p), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = Ua[i*m+j, :]
                v_val = Va[j*p+k, :]
                w_val = Wa[k*n+i, :]
                results[i,j,k] = np.dot(u_val * v_val, w_val)
    
    return np.allclose(results, 1.0, atol=tol)

def _als_cp_decomposition(T, rank, n_iters=200, tol=1e-12, seed=0):
    """ALS for CP decomposition of a 3D tensor T of shape (n*m, m*p, p*n).
    But our tensor is not directly available in that shape. Let me reconsider.
    
    The matrix multiplication tensor T has indices (i,j,k) with T[i,j,k]=1.
    We want to decompose T as sum_r U_r ⊗ V_r ⊗ W_r where:
    U_r[i,j] = U[i*m+j, r]
    V_r[j,k] = V[j*p+k, r]  
    W_r[k,i] = W[k*n+i, r]
    
    So T[i,j,k] = sum_r U_r[i,j] * V_r[j,k] * W_r[k,i]
    
    This is a CP decomposition of the 3D tensor T (shape n×m×p) with all entries 1.
    """
    n, m, p = DIMENSIONS
    T = np.ones((n, m, p), dtype=np.float64)
    
    rng = np.random.RandomState(seed)
    
    # Initialize
    U = rng.randn(n, m, rank)
    V = rng.randn(m, p, rank)
    W = rng.randn(p, n, rank)
    
    # Normalize
    for r in range(rank):
        norm = np.linalg.norm(U[:, :, r])
        if norm > 0:
            U[:, :, r] /= norm
            V[:, :, r] *= norm
            W[:, :, r] *= norm
    
    for it in range(n_iters):
        # Update U: for each (i,j), U[i,j,:] = (T[i,j,:] · (V ⊙ W)) / (V[i,j,:] ⊙ W[i,j,:] · U[i,j,:])
        # Actually in ALS for CP:
        # U[i,j,r] = sum_{k} T[i,j,k] * V[j,k,r] * W[k,i,r] / sum_{k'} V[j,k',r] * W[k',i,r] * U[i,j,r]
        # Wait, that's not right either. Let me use the standard formula.
        
        # For CP decomposition T ≈ sum_r U_r ⊗ V_r ⊗ W_r:
        # U[i,j,:] = mat((T ×_1 V ×_2 W)[i,j,:]) ... 
        # Actually: fix i,j. Then T[i,j,k] ≈ sum_r U[i,j,r] V[j,k,r] W[k,i,r]
        # For fixed (i,j) and r: U[i,j,r] = sum_k T[i,j,k] V[j,k,r] W[k,i,r] / sum_k V[j,k,r] W[k,i,r] U[i,j,r]
        # No wait, that's circular.
        
        # Standard ALS: to update U, fix V and W.
        # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # For fixed i,j: the vector T[i,j,:] (length p) ≈ sum_r U[i,j,r] * (V[j,:,r] * W[:,i,r])
        # So T[i,j,:] ≈ U[i,j,:] @ M where M[r,k] = V[j,k,r] * W[k,i,r]
        # Thus U[i,j,:] ≈ T[i,j,:] @ M^+ (pseudo-inverse)
        
        # Let me do it properly:
        # For each (i,j):
        #   Build matrix M of shape (rank, p): M[r, k] = V[j, k, r] * W[k, i, r]
        #   Solve: U[i,j,:] @ M^T = T[i,j,:]  (as a linear system)
        #   i.e., M @ U[i,j,:]^T = T[i,j,:]^T
        #   U[i,j,:] = lstsq(M, T[i,j,:])
        
        for i in range(n):
            for j in range(m):
                # M[r, k] = V[j, k, r] * W[k, i, r]
                M = np.zeros((rank, p))
                for r in range(rank):
                    M[r, :] = V[j, :, r] * W[:, i, r]
                # Solve M @ u = T[i,j,:]
                try:
                    U[i, j, :] = np.linalg.lstsq(M, T[i, j, :], rcond=None)[0]
                except:
                    U[i, j, :] = rng.randn(rank)
        
        # Update V similarly
        for j in range(m):
            for k in range(p):
                # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
                # For fixed (j,k): T[:,j,k] (length n) ≈ sum_r V[j,k,r] * (U[:,j,r] * W[k,:,r])
                # M[r, i] = U[i, j, r] * W[k, i, r]
                M = np.zeros((rank, n))
                for r in range(rank):
                    M[r, :] = U[:, j, r] * W[k, :, r]
                try:
                    V[j, k, :] = np.linalg.lstsq(M, T[:, j, k], rcond=None)[0]
                except:
                    V[j, k, :] = rng.randn(rank)
        
        # Update W similarly
        for k in range(p):
            for i in range(n):
                # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
                # For fixed (k,i): T[i,:,k] (length m) ≈ sum_r W[k,i,r] * (U[i,:,r] * V[:,k,r])
                # M[r, j] = U[i, j, r] * V[j, k, r]
                M = np.zeros((rank, m))
                for r in range(rank):
                    M[r, :] = U[i, :, r] * V[:, k, r]
                try:
                    W[k, i, :] = np.linalg.lstsq(M, T[i, :, k], rcond=None)[0]
                except:
                    W[k, i, :] = rng.randn(rank)
        
        # Compute error
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    pred = np.dot(U[i,j,:], V[j,k,:] * W[k,i,:])
                    err += (pred - 1.0)**2
        if err < tol:
            break
    
    # Convert to our format
    U_mat = [[0.0]*rank for _ in range(n*m)]
    V_mat = [[0.0]*rank for _ in range(m*p)]
    W_mat = [[0.0]*rank for _ in range(p*n)]
    for i in range(n):
        for j in range(m):
            U_mat[i*m+j] = U[i,j,:].tolist()
    for j in range(m):
        for k in range(p):
            V_mat[j*p+k] = V[j,k,:].tolist()
    for k in range(p):
        for i in range(n):
            W_mat[k*n+i] = W[k,i,:].tolist()
    
    return U_mat, V_mat, W_mat

def _try_rank(rank, n_trials=20, deadline=None):
    """Try to find a valid rank-r decomposition."""
    n, m, p = DIMENSIONS
    best = None
    best_err = float('inf')
    
    for trial in range(n_trials):
        if deadline is not None and time.time() > deadline:
            break
        
        seed = trial * 1000 + 42
        U, V, W = _als_cp_decomposition(None, rank, n_iters=100, tol=1e-14, seed=seed)
        
        # Check error
        Ua = np.array(U)
        Va = np.array(V)
        Wa = np.array(W)
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    pred = np.dot(Ua[i*m+j,:], Va[j*p+k,:] * Wa[k*n+i,:])
                    err += (pred - 1.0)**2
        err = np.sqrt(err)
        
        if err < best_err:
            best_err = err
            best = (U, V, W, err)
        
        if err < 1e-10:
            return best
    
    return best

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Safety margin
    
    # Schoolbook as fallback
    rank_school = n * m * p
    U_s = [[0]*rank_school for _ in range(n*m)]
    V_s = [[0]*rank_school for _ in range(m*p)]
    W_s = [[0]*rank_school for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_s[i*m+j][col] = 1
                V_s[j*p+k][col] = 1
                W_s[k*n+i][col] = 1
                col += 1
    best_result = {"U": U_s, "V": V_s, "W": W_s}
    best_rank = rank_school
    
    # Try ranks from low to high
    # Known: 4x4x4 has rank <= 20. For 4x4x6, let's try from 20 upward.
    for rank in range(20, 50):
        if time.time() > deadline - 20:
            break
        result = _try_rank(rank, n_trials=10, deadline=deadline - 20)
        if result is not None:
            U, V, W, err = result
            if err < 1e-10:
                # Verify exactly
                if _verify_decomposition_vectorized(U, V, W, n, m, p, tol=1e-9):
                    best_result = {"U": U, "V": V, "W": W}
                    best_rank = rank
                    break
            elif err < best_err_global if 'best_err_global' in dir() else True:
                if err < 1e-8:
                    best_result = {"U": U, "V": V, "W": W}
                    best_rank = rank
    
    return best_result
# EVOLVE-BLOCK-END
