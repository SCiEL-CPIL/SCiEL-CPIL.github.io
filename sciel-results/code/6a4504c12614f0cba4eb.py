import time
import numpy as np

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145.0
    
    nm = n * m  # 6
    mp = m * p  # 27
    pn = p * n  # 18
    
    # Build the target tensor: T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i
    # Otherwise 0
    target = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target[a, b, c] = 1.0
    
    # Trivial rank-54 decomposition as fallback
    def trivial_decomp():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Try to find a low-rank decomposition
    # The tensor has a specific structure. Let's try rank r for various r.
    
    # Key insight: The tensor can be written as sum over j of (sum_i e_i) ⊗ e_j ⊗ (sum_k e_k)
    # But in the flattened space, this doesn't directly work due to cross-terms.
    
    # Alternative: use the fact that for each j, the sub-tensor is rank 1 in (i,k) space.
    # Let's try to construct a decomposition.
    
    # For matrix multiplication, the tensor rank is known to be at most n*p + m*p + n*m - (n+m+p-2)
    # or similar bounds. For 2x3x9, let's try to find a good decomposition.
    
    # Let's try a direct approach: use SVD-like decomposition on the tensor.
    # Reshape the tensor to a matrix and find low-rank approximation.
    
    def try_rank(r, max_iters=100):
        """Try to find a rank-r decomposition using alternating least squares."""
        if r < 1:
            return None
        
        # Initialize randomly
        np.random.seed(42)
        U = np.random.randn(nm, r) * 0.1
        V = np.random.randn(mp, r) * 0.1
        W = np.random.randn(pn, r) * 0.1
        
        # Scale to get reasonable magnitudes
        U *= 2.0
        V *= 2.0
        W *= 2.0
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                return None
            
            # Update U: for each a, U[a,:] = argmin ||target[a,:,:] - sum_r U[a,r] V[:,r] W[:,r]^T||
            # This is: U[a,r] = <target[a,:,:], V[:,r] W[:,r]^T> / ||V[:,r] W[:,r]^T||^2
            # But this is coupled. Use alternating approach.
            
            # Update U
            for a in range(nm):
                # target[a,:,:] is a (mp, pn) matrix
                # We want: target[a,:,:] ≈ sum_r U[a,r] * outer(V[:,r], W[:,r])
                # So: U[a,r] ≈ <target[a,:,:], outer(V[:,r], W[:,r])> / ||outer(V[:,r], W[:,r])||^2
                # But we need to solve a linear system.
                
                # Build the system: for each r, the contribution is U[a,r] * outer(V[:,r], W[:,r])
                # Flatten: target[a,:,:].flatten() = sum_r U[a,r] * outer(V[:,r], W[:,r]).flatten()
                # This is a linear system in U[a,:]
                
                # A[a] is (mp*pn) x r
                A_mat = np.zeros((mp * pn, r))
                for r_idx in range(r):
                    A_mat[:, r_idx] = np.outer(V[:, r_idx], W[:, r_idx]).flatten()
                
                b_vec = target[a, :, :].flatten()
                
                # Solve A_mat @ U[a,:] = b_vec
                try:
                    U[a, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                except:
                    pass
            
            # Update V
            for b in range(mp):
                A_mat = np.zeros((nm * pn, r))
                for r_idx in range(r):
                    A_mat[:, r_idx] = np.outer(U[:, r_idx], W[:, r_idx]).flatten()
                
                # target[:, b, :] reshaped: for each (a,c), target[a,b,c]
                b_vec = np.zeros(nm * pn)
                for a in range(nm):
                    for c in range(pn):
                        b_vec[a * pn + c] = target[a, b, c]
                
                try:
                    V[b, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                except:
                    pass
            
            # Update W
            for c in range(pn):
                A_mat = np.zeros((nm * mp, r))
                for r_idx in range(r):
                    A_mat[:, r_idx] = np.outer(U[:, r_idx], V[:, r_idx]).flatten()
                
                b_vec = np.zeros(nm * mp)
                for a in range(nm):
                    for b in range(mp):
                        b_vec[a * mp + b] = target[a, b, c]
                
                try:
                    W[c, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                except:
                    pass
            
            # Check convergence
            recon = np.zeros((nm, mp, pn))
            for r_idx in range(r):
                recon += np.einsum('ar,br,cr->abc', U, V, W)
            err = np.linalg.norm(recon - target)
            if err < 1e-8:
                # Found a valid decomposition
                # Round to nearest half-integers if close
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                # Verify rounded version
                recon_r = np.zeros((nm, mp, pn))
                for r_idx in range(r):
                    recon_r += np.einsum('ar,br,cr->abc', U_rounded, V_rounded, W_rounded)
                err_r = np.linalg.norm(recon_r - target)
                
                if err_r < 1e-6:
                    return (U_rounded, V_rounded, W_rounded, r)
                else:
                    return (U, V, W, r)
        
        return None
    
    # Try ranks from low to high
    best_result = None
    for r in range(3, 20):
        if time.time() > deadline - 10:
            break
        
        result = try_rank(r, max_iters=50)
        if result is not None:
            best_result = result
            break
    
    if best_result is not None:
        U, V, W, r = best_result
        # Convert to lists
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback to trivial
    return trivial_decomp()
