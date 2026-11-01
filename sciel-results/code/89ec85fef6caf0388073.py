"""Find minimal rank tensor decomposition for 2x4 * 4x7 matrix multiplication."""
import numpy as np
import time
import itertools

DIMENSIONS = [2, 4, 7]

def construct():
    start_time = time.time()
    timeout = 145  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    dim_u, dim_v, dim_w = n*m, m*p, p*n
    
    # Build target tensor once
    T = np.zeros((dim_u, dim_v, dim_w))
    for i, j, k in itertools.product(range(n), range(m), range(p)):
        T[i*m+j, j*p+k, k*n+i] = 1.0
    
    def verify_exact(U, V, W):
        """Check if U⊗V⊗W == T with half-integer entries"""
        # Half-integer check
        for arr in (U, V, W):
            for val in arr.flatten():
                if abs(val*2 - round(val*2)) > 1e-9:
                    return False
        # Tensor product check
        recon = np.einsum('ur,vr,wr->uvw', U, V, W)
        return np.array_equal(recon, T) or np.allclose(recon, T, atol=1e-9)
    
    # Schoolbook baseline
    sb_rank = n*m*p
    U_sb = np.zeros((dim_u, sb_rank))
    V_sb = np.zeros((dim_v, sb_rank))
    W_sb = np.zeros((dim_w, sb_rank))
    for idx, (i,j,k) in enumerate(itertools.product(range(n), range(m), range(p))):
        U_sb[i*m+j, idx] = 1.0
        V_sb[j*p+k, idx] = 1.0
        W_sb[k*n+i, idx] = 1.0
    
    best = (U_sb, V_sb, W_sb)
    best_rank = sb_rank
    
    # Try lower ranks - start from reasonable lower bound
    # Theoretical lower bound is at least max(dim_u,dim_v,dim_w) = 28
    # But tensor rank for matmul is typically close to schoolbook
    
    lower_bound = 20
    for rank in range(lower_bound, sb_rank):
        if time.time() - start_time > timeout - 30:
            break
            
        found = search_rank(dim_u, dim_v, dim_w, T, rank, verify_exact)
        if found:
            best = found
            best_rank = rank
            # Try even lower? Break if we want first success
            break  # Get first working lower rank
    
    U_final, V_final, W_final = best
    return {
        "U": U_final.tolist(),
        "V": V_final.tolist(),
        "W": W_final.tolist()
    }

def search_rank(dim_u, dim_v, dim_w, T, rank, verify_fn):
    """Search for decomposition of given rank using multiple strategies"""
    # Strategy 1: Random initialization + coordinate descent
    for seed in range(50):
        np.random.seed(seed)
        U = np.random.randn(dim_u, rank) * 0.1
        V = np.random.randn(dim_v, rank) * 0.1
        W = np.random.randn(dim_w, rank) * 0.1
        
        for it in range(100):
            # Coordinate descent: update each factor
            # Update U: for each u, solve linear system
            for u in range(dim_u):
                # T[u,:,:] = sum_r U[u,r] * V[:,r] ⊗ W[:,r]
                # This is: T[u].flatten() = (V⊗W reshaped) @ U[u,:]
                VW = np.kron(V, W).reshape(dim_v*dim_w, rank)
                target = T[u].flatten()
                U[u] = np.linalg.lstsq(VW, target, rcond=None)[0]
            
            # Update V
            for v in range(dim_v):
                UW = np.kron(U, W).reshape(dim_u*dim_w, rank)
                target = T[:,v,:].flatten()
                V[v] = np.linalg.lstsq(UW, target, rcond=None)[0]
            
            # Update W
            for w in range(dim_w):
                UV = np.kron(U, V).reshape(dim_u*dim_v, rank)
                target = T[:,:,w].flatten()
                W[w] = np.linalg.lstsq(UV, target, rcond=None)[0]
            
            # Check convergence
            recon = np.einsum('ur,vr,wr->uvw', U, V, W)
            if np.allclose(recon, T, atol=1e-10):
                U_r, V_r, W_r = np.round(U*2)/2, np.round(V*2)/2, np.round(W*2)/2
                if verify_fn(U_r, V_r, W_r):
                    return U_r, V_r, W_r
    
    return None
