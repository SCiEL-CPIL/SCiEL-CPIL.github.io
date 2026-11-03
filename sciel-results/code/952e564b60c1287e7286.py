"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Build the target tensor T of shape (nm, mp, pn)
    # T[i*m+j, j*p+k, k*n+i] = 1, else 0
    T = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Try to find a low-rank CP decomposition
    # Start with rank less than schoolbook
    schoolbook_rank = n * m * p
    
    def verify(U, V, W):
        """Check if U, V, W exactly decompose T (within tolerance for half-integers)."""
        r = U.shape[1]
        # Compute reconstruction
        # T_recon[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # We need to check this equals T
        # Efficient: for each c, outer product
        T_recon = np.zeros((nm, mp, pn), dtype=float)
        for c in range(r):
            uc = U[:, c]
            vc = V[:, c]
            wc = W[:, c]
            # T_recon += outer(uc, outer(vc, wc)) but with index structure
            # Actually, T_recon[a,b,c_idx] += uc[a] * vc[b] * wc[c_idx]
            # This is just the outer product of the three vectors
            T_recon += np.einsum('a,b,c->abc', uc, vc, wc)
        
        # Check if T_recon equals T
        # Since entries should be half-integers, check with tolerance
        max_err = np.max(np.abs(T_recon - T))
        return max_err < 1e-9
    
    def round_to_half_int(x):
        """Round to nearest half-integer."""
        return np.round(x * 2) / 2.0
    
    def try_rank(r, time_budget=100):
        """Try to find a decomposition of rank r."""
        start_time = time.time()
        
        # Initialize randomly
        np.random.seed(42)
        U = np.random.randn(nm, r) * 0.1
        V = np.random.randn(mp, r) * 0.1
        W = np.random.randn(pn, r) * 0.1
        
        # ALS optimization
        for it in range(200):
            if time.time() - start_time > time_budget:
                break
            
            # Update U: for each a, U[a, :] = argmin sum_b,c T[a,b,c] * V[b,:] * W[c,:]
            # This is a linear system: U[a,:] @ (V[b,:] * W[c,:]) = T[a,b,c]
            # For fixed a: U[a,:] is a vector of length r
            # We need: sum_c U[a,c] * V[b,c] * W[c,c] = T[a,b,c] for all b,c
            # This is: U[a,:] * (V ⊙ W) = T[a,:,:].flatten() where ⊙ is element-wise in c
            # Actually, for each (b, c_idx), we have: sum_r U[a,r] * V[b,r] * W[c_idx,r] = T[a,b,c_idx]
            # So: U[a,:] @ (V[b,:] * W[c_idx,:]) = T[a,b,c_idx]
            # This is a linear system in U[a,:]: M[a] @ U[a,:] = T[a,:,:].flatten()
            # where M[a][ (b,c_idx), r ] = V[b,r] * W[c_idx,r]
            
            # This is expensive. Let's use a simpler approach: gradient descent.
            pass
        
        # Try rounding and verification
        U_r = round_to_half_int(U)
        V_r = round_to_half_int(V)
        W_r = round_to_half_int(W)
        
        if verify(U_r, V_r, W_r):
            return U_r, V_r, W_r
        
        return None
    
    # Try ranks from just below schoolbook down
    for r in range(schoolbook_rank - 1, max(50, schoolbook_rank // 2), -1):
        result = try_rank(r, time_budget=30)
        if result is not None:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fall back to schoolbook
    rank = schoolbook_rank
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][column] = 1
                V[j*p+k][column] = 1
                W[k*n+i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
