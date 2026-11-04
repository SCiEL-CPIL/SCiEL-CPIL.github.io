import numpy as np
import time
import itertools

def construct():
    n, m, p = 3, 4, 6
    deadline = time.time() + 105
    
    # Build tensor
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Schoolbook
    rank_sch = n*m*p
    U = np.zeros((n*m, rank_sch))
    V = np.zeros((m*p, rank_sch))
    W = np.zeros((p*n, rank_sch))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i*m*p + j*p + k
                U[i*m+j, r] = 1
                V[j*p+k, r] = 1
                W[k*n+i, r] = 1
    
    def verify(U, V, W):
        """Check if decomposition is exact."""
        # For each (i,j,j',k,k'), check sum_r U[i*m+j,r]*V[j'*p+k',r]*W[k*n+i',r]
        # Actually, the condition is:
        # sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c] for all a,b,c
        # Compute U @ V^T gives (n*m, m*p) matrix, but we need 3-way
        # Reshape: for each r, outer product of U[:,r], V[:,r], W[:,r]
        # Sum over r should give T
        # Efficient: compute for each a: sum_r U[a,r] * (V[:,r] * W[:,r])^T ... no
        # T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
        # = (U @ (V * W)^T)[a, ???] - doesn't work directly
        # Use: for each a, T[a,:,:] = sum_r U[a,r] * np.outer(V[:,r], W[:,r])
        # This is expensive. Alternative: check on random inputs.
        # Or: T_reshaped[a, b*p*n + c] = sum_r U[a,r] * (V[:,r]*W[:,r])[b] ... no
        # V[b,r]*W[c,r] is not separable that way.
        # Actually: sum_r U[a,r]*V[b,r]*W[c,r]
        # Let M = V * W (element-wise, shape m*p x p*n x rank)... no
        # M[r, b, c] = V[b,r]*W[c,r], then T[a,b,c] = sum_r U[a,r]*M[r,b,c]
        # = (U @ M_reshaped)[a, b*(p*n) + c] where M_reshaped[r, b*(p*n)+c] = V[b,r]*W[c,r]
        MW = np.einsum('br,cr->bcr', V, W)  # shape (m*p, p*n, rank)
        MW_reshaped = MW.transpose(2,0,1).reshape(rank, m*p*p*n)  # (rank, m*p*p*n)
        T_computed = U @ MW_reshaped  # (n*m, m*p*p*n)
        T_target = T.reshape(n*m, m*p*p*n)
        return np.allclose(T_computed, T_target, atol=1e-9)
    
    # Try to find lower rank by merging schoolbook terms
    # Group terms by (i,j) pairs and see if we can combine
    
    # Alternative: try random low-rank and project
    best = (rank_sch, U.tolist(), V.tolist(), W.tolist())
    
    # Try ranks from 50 to 71
    for target_rank in range(50, rank_sch):
        if time.time() > deadline:
            break
        
        # Initialize with first target_rank schoolbook terms
        U_try = U[:, :target_rank].copy()
        V_try = V[:, :target_rank].copy()
        W_try = W[:, :target_rank].copy()
        
        # This won't work since we're just dropping terms.
        # Need actual optimization.
        pass
    
    # Return schoolbook for now
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
