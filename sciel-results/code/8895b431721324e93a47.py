"""Matrix multiplication 3,3,7 - rank-minimal exact tensor decomposition."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 7]
N, M, P = DIMENSIONS

def _verify(U, V, W, rank, n, m, p, tol=1e-10):
    """Verify that U @ V^T gives the correct multiplication tensor when contracted with W."""
    # T[i*m+j, k*n+l, c] should sum to delta(i,l) * (A[i,j] * B[j,k])
    # Actually: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] should equal 1 if the "output" index matches
    # The tensor for matrix multiplication: T[a,b,c] = 1 where a indexes (i,j), b indexes (j,k), c indexes (k,i)
    # and the decomposition is: T = sum_r u_r * v_r * w_r
    # where u_r[i*m+j], v_r[j*p+k], w_r[k*n+i]
    # For the product to be correct: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 if this represents the
    # identity mapping, which it does for the standard tensor.
    
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    # Compute the tensor via einsum
    # T[i*m+j, j'*p+k, k'*n+i'] = sum_r U[i*m+j,r]*V[j'*p+k,r]*W[k'*n+i',r]
    # For matrix mult, we need: when i'=i and k'=k (i.e., the output position matches), and j'=j,
    # the sum should be 1; otherwise 0.
    # Actually the standard form: the (i,j,k) entry of the multiplication tensor is 1 iff the indices
    # are consistent (which they always are in the flattened form). Let me just check directly.
    
    # Build full tensor
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for r in range(rank):
        np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r], out=T)
    
    # Expected: T[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k (with consistent indices)
    # More precisely, for the matrix multiplication tensor in the "trace" convention:
    # The output C[i,k] = sum_j A[i,j]*B[j,k]
    # In flattened form: C is indexed by k*n+i (per the task: W[k*n+i,r])
    # A is indexed by i*m+j
    # B is indexed by j*p+k
    # The tensor entry T[(i,j), (j',k), (k',i')] = delta_{i,i'} * delta_{k,k'} if j=j'
    # Actually it's: T[(i,j),(j',k),(k',i')] = 1 if i=i' AND k=k' AND j=j', else 0.
    # But since we're summing over j to get C[i,k], the tensor should have:
    # T[(i,j),(j,k),(k,i)] = 1 for all i,j,k, and 0 otherwise.
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = T[i*m+j, j*p+k, k*n+i]
                if abs(actual - expected) > tol:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    n, m, p = N, M, P
    rank = n * m * p  # 63 for schoolbook
    
    # Start with schoolbook
    U = np.zeros((n*m, rank), dtype=np.float64)
    V = np.zeros((m*p, rank), dtype=np.float64)
    W = np.zeros((p*n, rank), dtype=np.float64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1.0
                V[j*p+k, col] = 1.0
                W[k*n+i, col] = 1.0
                col += 1
    
    best_rank = rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try to reduce rank by finding linear dependencies
    # The idea: if we can express the tensor with fewer terms, we save rank.
    # 
    # Key insight for 3x3x7: We can try to use a 3x3x3 decomposition plus additional terms.
    # A 3x3 matrix multiplication has rank at most 27 (schoolbook) but may be less.
    # Actually, the rank of 3x3 matrix multiplication is known to be at most 23 (Strassen-like)
    # but the exact minimum is not known to be below 19.
    #
    # For 3x3x7, one approach: decompose B (3x7) into sum of 3x3 blocks.
    # B = [B1 | B2 | B3 | B4] where each Bj is 3x2 except the last which is 3x1.
    # Actually 7 = 2+2+2+1.
    #
    # Alternative: Use the fact that 3x3 matrix mult can be done with rank 23 (or less).
    # For each column of B, we multiply A (3x3) by a 3x1 vector, which is a 3x3x1 tensor.
    # The 3x3x1 tensor (matrix-vector mult) has rank 9 (schoolbook: 3*3*1=9).
    # But we can potentially share structure.
    #
    # Let me try a different approach: use known results.
    # The tensor [3,3,7] can be seen as 7 copies of [3,3,1] (matrix-vector) glued together.
    # Each [3,3,1] has rank 9. Total naive: 63.
    # But we can share the "A" factor across all 7 columns.
    #
    # Better approach: Try to find a decomposition of rank < 63 using algebraic structure.
    #
    # For now, let's try a greedy rank reduction:
    # 1. Form the full tensor T of shape (n*m, m*p, p*n)
    # 2. Try to find a low-rank decomposition
    
    # Build the target tensor
    T_target = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Try various rank values from below schoolbook
    # Use a simple iterative approach: try to fit a lower-rank decomposition
    
    # First, let's try rank = 50, 49, 48, ... down to some minimum
    # Use a randomized greedy approach
    
    def try_rank(r):
        if time.time() > deadline - 5.0:
            return None
        
        # Randomized SVD-like approach for tensors
        # Initialize with random vectors
        np.random.seed(42 + r)
        U_t = np.random.randn(n*m, r)
        V_t = np.random.randn(m*p, r)
        W_t = np.random.randn(p*n, r)
        
        # Iteratively refine
        for it in range(100):
            if time.time() > deadline - 5.0:
                return None
            
            # Update U: for each (i,j), find best u given V, W
            # sum_r u_r * v_r * w_r = T[i*m+j, :, :]
            # This is a linear system: U[i*m+j,:] @ (V[:,r] * W[:,r])^T = T[i*m+j, :, :]
            # Actually: sum_r U[i*m+j,r] * V[j'*p+k,r] * W[k'*n+i',r] = T[i*m+j, j'*p+k, k'*n+i']
            
            # For fixed (i,m,j), we need: sum_r U[i*m+j,r] * (V[:,r] * W[:,r]) = T[i*m+j, :, :]
            # where (V[:,r] * W[:,r]) is a vector of length m*p*p*n
            # This is underdetermined if r is small, overdetermined if r is large.
            
            # Better: alternate least squares
            # Update U[i*m+j, r] = <T[i*m+j,:,:], V[:,r]*W[:,r]> / <V[:,r]*V[:,r]> ... 
            # Actually it's not that simple. Let me use a different formulation.
            
            # The tensor is: T = sum_r u_r ⊗ v_r ⊗ w_r
            # So T[i*m+j, a, b] = sum_r U[i*m+j,r] * V[a,r] * W[b,r]
            
            # To update U[i*m+j, :]:
            # T[i*m+j, a, b] = sum_r U[i*m+j,r] * (V[a,r] * W[b,r])
            # Let M_ab_r = V[a,r] * W[b,r], then T[i*m+j] = sum_r U[i*m+j,r] * M_r
            # where M_r is a vector of length m*p*p*n
            # U[i*m+j, :] = argmin sum_{a,b} (T[i*m+j,a,b] - sum_r U[i*m+j,r]*V[a,r]*W[b,r])^2
            # = solve least squares
            
            flat_T = T_target.reshape(n*m, -1)
            
            for idx in range(n*m):
                # Build design matrix
                # Design[i,j,r] = V[i,r] * W[j,r]
                design = V * W  # shape (m*p, p*n, r) -> need to flatten first two dims
                design_flat = design.reshape(m*p*p*n, r)
                # Solve: design_flat @ U[idx, :] = flat_T[idx, :]
                try:
                    sol, _, _, _ = np.linalg.lstsq(design_flat, flat_T[idx], rcond=None)
                    U_t[idx, :] = sol
                except:
                    pass
            
            # Update V
            flat_T2 = T_target.transpose(1, 0, 2).reshape(m*p, -1)
            for idx in range(m*p):
                design = U * W  # (n*m, p*n, r)
                design_flat = design.reshape(n*m*p*n, r)
                try:
                    sol, _, _, _ = np.linalg.lstsq(design_flat, flat_T2[idx], rcond=None)
                    V_t[idx, :] = sol
                except:
                    pass
            
            # Update W
            flat_T3 = T_target.transpose(2, 0, 1).reshape(p*n, -1)
            for idx in range(p*n):
                design = U * V  # (n*m, m*p, r)
                design_flat = design.reshape(n*m*m*p, r)
                try:
                    sol, _, _, _ = np.linalg.lstsq(design_flat, flat_T3[idx], rcond=None)
                    W_t[idx, :] = sol
                except:
                    pass
        
        # Check accuracy
        T_recon = np.zeros((n*m, m*p, p*n), dtype=np.float64)
        for r in range(r):
            np.einsum('a,b,c->abc', U_t[:,r], V_t[:,r], W_t[:,r], out=T_recon)
        
        err = np.max(np.abs(T_recon - T_target))
        if err < 1e-8:
            return (U_t, V_t, W_t, r)
        return None
    
    # Try to find a lower rank
    # Start from rank-1 and go up, but that's too slow. Instead, try specific values.
    # Known bounds: rank of [3,3,p] for p=7
    # Let's try from rank 50 down to rank 40
    
    for r in range(55, 30, -1):
        if time.time() > deadline - 10.0:
            break
        result = try_rank(r)
        if result is not None:
            U_t, V_t, W_t, achieved_rank = result
            # Try to zero out small entries to get integer/half-integer values
            U_int = np.round(U_t * 2) / 2
            V_int = np.round(V_t * 2) / 2
            W_int = np.round(W_t * 2) / 2
            
            if _verify(U_int, V_int, W_int, achieved_rank, n, m, p, tol=1e-8):
                best_rank = achieved_rank
                best_U, best_V, best_W = U_int, V_int, W_int
                break
            else:
                # Keep the float version if it's close
                if err < 1e-6:
                    best_rank = achieved_rank
                    best_U, best_V, best_W = U_t, V_t, W_t
    
    if time.time() > deadline - 3.0:
        pass
    
    # Convert to lists
    def to_list(arr):
        return [[float(x) for x in row] for row in arr]
    
    return {
        "U": to_list(best_U),
        "V": to_list(best_V),
        "W": to_list(best_W)
    }
