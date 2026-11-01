"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x10 matrix.
    
    Returns dict with 'U', 'V', 'W' matrices.
    """
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 2, 3, 10
    
    # Build the multiplication tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j'*p+k', k''*n+i''] = 1 if j==j', k'==k'', i==i'', else 0
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    def verify(U, V, W):
        """Check if U, V, W form a valid decomposition."""
        U = np.array(U, dtype=np.float64)
        V = np.array(V, dtype=np.float64)
        W = np.array(W, dtype=np.float64)
        # Reconstruct tensor
        T_recon = np.einsum('ac,bc,dc->abd', U, V, W)
        return np.allclose(T_recon, T, atol=1e-10)
    
    def make_schoolbook():
        """Create the schoolbook rank-60 decomposition."""
        rank = n * m * p  # 60
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
        return U, V, W
    
    # Start with schoolbook
    U_best, V_best, W_best = make_schoolbook()
    best_rank = n * m * p
    
    # Try to find lower rank using numerical optimization
    # Strategy: try ranks from a lower bound upward
    # Lower bound: max(n*m, m*p, p*n) = max(6, 30, 20) = 30
    
    lower_bound = max(n*m, m*p, p*n)
    
    # For [2,3,p] matrix multiplication, let's try to find if we can beat 60
    # Use a structured approach: try to merge terms
    
    # Approach: Use the fact that for each k, we're computing a [2,3,1] matmul
    # The [2,3,1] tensor has rank 6 (can't do better)
    # So for 10 independent k's, we need at least 60 terms IF they're independent
    
    # But let's try a numerical search for rank 59, 58, etc.
    # This is computationally expensive, so we'll try a few ranks
    
    # Actually, let's try a smarter approach: use the structure to find a 
    # decomposition with rank < 60 if possible
    
    # Try rank values from lower_bound to best_rank-1
    # But this is very expensive. Let's try a few strategic ranks.
    
    # For this specific problem, let's try to see if we can achieve rank < 60
    # by using a different construction.
    
    # Key insight: For [n, m, p] with small n, we can sometimes use
    # "batching" tricks. Let's try to find a decomposition.
    
    # Try a structured approach: 
    # For each k, the computation is:
    # C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k] + A[0,2]*B[2,k]
    # C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k] + A[1,2]*B[2,k]
    
    # This is equivalent to multiplying a 2x3 matrix by a 3x1 matrix.
    # The rank of [2,3,1] is 6 (proven: you need 6 multiplications).
    
    # Since the B[*,k] for different k are independent, the total rank
    # should be 6*10 = 60.
    
    # Let's verify this is indeed the case by trying to find a rank-59
    # decomposition numerically.
    
    # Numerical approach: use alternating least squares or similar
    # to try to find a low-rank decomposition.
    
    def try_rank(r, max_iter=1000, tol=1e-8):
        """Try to find a rank-r decomposition using iterative optimization."""
        if time.time() > deadline:
            return None
        
        # Initialize with random matrices
        np.random.seed(42)
        U = np.random.randn(n*m, r)
        V = np.random.randn(m*p, r)
        W = np.random.randn(p*n, r)
        
        best_err = np.inf
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            
            # Update U: for each a, U[a,:] = argmin ||T[:,b,c] - sum_c' U[a,c']V[b,c']W[c,c']||
            # This is a linear system in U[a,:]
            for a in range(n*m):
                # We need: sum_c' U[a,c'] * V[b,c'] * W[c,c'] ≈ T[a,b,c] for all b,c
                # This is: (V * W^T) @ U[a,:] ≈ T[a,:,:].reshape(-1)
                # where (V * W^T) is (m*p*p*n) x r, with entry (b,c,c') = V[b,c']*W[c,c']
                # Actually: for fixed a, we want U[a,:] such that
                # sum_c' U[a,c'] * (V[:,c'] * W[:,c'].T) ≈ T[a,:,:]
                
                # Build the matrix: M[b, c, c'] = V[b, c'] * W[c, c']
                # Reshape to (m*p*p*n, r): M_flat[b*p*n+c, c'] = V[b,c']*W[c,c']
                M = np.einsum('bc,dc->bdc', V, W)  # shape (m*p, p*n, r)
                M_flat = M.reshape(m*p*p*n, r)
                T_flat = T[a, :, :].reshape(-1)
                
                # Solve: M_flat @ U[a,:] = T_flat
                try:
                    U[a, :], residuals, rank_sv, sv = np.linalg.lstsq(M_flat, T_flat, rcond=None)
                except:
                    pass
            
            # Update V
            for b in range(m*p):
                M = np.einsum('ac,dc->adc', U, W)  # shape (n*m, p*n, r)
                M_flat = M.reshape(n*m*p*n, r)
                T_flat = T[:, b, :].reshape(-1)
                try:
                    V[b, :], residuals, rank_sv, sv = np.linalg.lstsq(M_flat, T_flat, rcond=None)
                except:
                    pass
            
            # Update W
            for c in range(p*n):
                M = np.einsum('ac,bc->abc', U, V)  # shape (n*m, m*p, r)
                M_flat = M.reshape(n*m*m*p, r)
                T_flat = T[:, :, c].reshape(-1)
                try:
                    W[c, :], residuals, rank_sv, sv = np.linalg.lstsq(M_flat, T_flat, rcond=None)
                except:
                    pass
            
            # Check error
            T_recon = np.einsum('ac,bc,dc->abd', U, V, W)
            err = np.linalg.norm(T_recon - T)
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            if err < tol:
                break
        
        if best_err < tol:
            return best_U, best_V, best_W
        return None
    
    # Try ranks from lower_bound upward
    # But this is expensive. Let's try a few strategic ranks.
    # For [2,3,10], let's try ranks 55-59 to see if we can beat 60.
    
    for r in range(best_rank - 1, lower_bound - 1, -1):
        if time.time() > deadline:
            break
        
        result = try_rank(r, max_iter=200)
        if result is not None:
            U_best, V_best, W_best = result
            best_rank = r
            break  # Found a lower rank, stop
    
    # Convert to lists and return
    U_list = U_best.tolist()
    V_list = V_best.tolist()
    W_list = W_best.tolist()
    
    # Verify
    if not verify(U_list, V_list, W_list):
        # Fall back to schoolbook
        U_best, V_best, W_best = make_schoolbook()
        best_rank = n * m * p
        U_list = U_best.tolist()
        V_list = V_best.tolist()
        W_list = W_best.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
