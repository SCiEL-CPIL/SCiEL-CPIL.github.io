"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a low-rank exact tensor decomposition for 
    multiplying a 3x4 matrix by a 4x4 matrix.
    
    Uses an iterative optimization approach to find a low-rank decomposition,
    then verifies and refines the result.
    
    Returns:
        dict: {'U': list of lists, 'V': list of lists, 'W': list of lists}
    """
    start_time = time.time()
    deadline = start_time + 108  # 108s limit, leave 42s margin
    
    n, m, p = DIMENSIONS
    dim_a = n * m  # 12
    dim_b = m * p  # 16
    dim_c = p * n  # 12
    
    # Build the multiplication tensor T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((dim_a, dim_b, dim_c), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
    
    # Schoolbook decomposition as fallback
    rank_school = n * m * p  # 48
    U_school = np.zeros((dim_a, rank_school))
    V_school = np.zeros((dim_b, rank_school))
    W_school = np.zeros((dim_c, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m + j, col] = 1.0
                V_school[j*p + k, col] = 1.0
                W_school[k*n + i, col] = 1.0
                col += 1
    
    best_rank = rank_school
    best_U = U_school.copy()
    best_V = V_school.copy()
    best_W = W_school.copy()
    
    # Try to find a lower rank decomposition
    # Use an alternating least squares approach
    
    for target_rank in range(35, rank_school):
        if time.time() > deadline:
            break
        
        found = False
        n_restarts = 50
        
        for restart in range(n_restarts):
            if time.time() > deadline:
                break
            
            # Initialize randomly
            U = np.random.randn(dim_a, target_rank) * 0.5
            V = np.random.randn(dim_b, target_rank) * 0.5
            W = np.random.randn(dim_c, target_rank) * 0.5
            
            # Alternating least squares
            for iteration in range(100):
                if time.time() > deadline:
                    break
                
                # Update U: for each column r, minimize ||T - sum_r U_r ⊗ V_r ⊗ W_r||
                # This is equivalent to: for each r, U_r = argmin ||T - U_r ⊗ V_r ⊗ W_r - rest||
                # Which can be solved by flattening
                
                # Actually, for ALS, we fix V and W, and solve for each U_r
                # The contribution of term r is U_r ⊗ V_r ⊗ W_r
                # We want to minimize ||T - sum_r U_r ⊗ V_r ⊗ W_r||_F^2
                
                # For a fixed r, with V_r and W_r fixed, the optimal U_r is:
                # U_r = (sum over (i,j,k) of T[i,j,k] * V_r[j] * W_r[k]) / (sum over (j,k) of V_r[j]^2 * W_r[k]^2)
                
                # More carefully: the term involving U_r is U_r[i] * V_r[j] * W_r[k]
                # So for each i, the optimal U_r[i] = (sum_{j,k} T[i,j,k] * V_r[j] * W_r[k]) / (sum_{j,k} V_r[j]^2 * W_r[k]^2)
                
                # But we also have the other terms, so we need to subtract them first
                
                # Let's do this properly
                # Reconstruct current decomposition
                recon = np.zeros((dim_a, dim_b, dim_c))
                for r in range(target_rank):
                    recon += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
                
                # For each r, update U_r
                for r in range(target_rank):
                    # Subtract current term r
                    recon_without_r = recon - np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
                    
                    # Residual
                    residual = T - recon_without_r
                    
                    # Optimal U_r[i] = sum_{j,k} residual[i,j,k] * V_r[j] * W_r[k] / sum_{j,k} V_r[j]^2 * W_r[k]^2
                    numerator = np.einsum('ijk,j,k->i', residual, V[:, r], W[:, r])
                    denominator = np.sum(V[:, r] ** 2) * np.sum(W[:, r] ** 2)
                    
                    if denominator > 1e-10:
                        U[:, r] = numerator / denominator
                    
                    # Update recon
                    recon = recon_without_r + np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
                
                # Similarly update V_r
                recon = np.zeros((dim_a, dim_b, dim_c))
                for r in range(target_rank):
                    recon += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
                
                for r in range(target_rank):
                    recon_without_r = recon - np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
                    residual = T - recon_without_r
                    
                    numerator = np.einsum('ijk,i,k->j', residual, U[:, r], W[:, r])
                    denominator = np.sum(U[:, r] ** 2) * np.sum(W[:, r] ** 2)
                    
                    if denominator > 1e-10:
                        V[:, r] = numerator / denominator
                    
                    recon = recon_without_r + np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
                
                # Similarly update W_r
                recon = np.zeros((dim_a, dim_b, dim_c))
                for r in range(target_rank):
                    recon += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
                
                for r in range(target_rank):
                    recon_without_r = recon - np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
                    residual = T - recon_without_r
                    
                    numerator = np.einsum('ijk,i,j->k', residual, U[:, r], V[:, r])
                    denominator = np.sum(U[:, r] ** 2) * np.sum(V[:, r] ** 2)
                    
                    if denominator > 1e-10:
                        W[:, r] = numerator / denominator
                    
                    recon = recon_without_r + np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
            
            # Check if we've achieved a good fit
            if time.time() > deadline:
                break
            
            # Verify the decomposition
            reconstructed = np.zeros((dim_a, dim_b, dim_c))
            for r in range(target_rank):
                reconstructed += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
            
            # Check if it matches T (within tolerance)
            if np.allclose(reconstructed, T, atol=1e-8):
                # Round to nearest 0.5 for cleaner output
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                # Verify rounded version
                recon_rounded = np.zeros((dim_a, dim_b, dim_c))
                for r in range(target_rank):
                    recon_rounded += np.einsum('i,j,k->ijk', U_rounded[:, r], V_rounded[:, r], W_rounded[:, r])
                
                if np.allclose(recon_rounded, T, atol=1e-9):
                    best_rank = target_rank
                    best_U = U_rounded
                    best_V = V_rounded
                    best_W = W_rounded
                    found = True
                    break
        
        if found:
            break
    
    # Convert to lists for output
    best_U_list = best_U.tolist()
    best_V_list = best_V.tolist()
    best_W_list = best_W.tolist()
    
    return {"U": best_U_list, "V": best_V_list, "W": best_W_list}
# EVOLVE-BLOCK-END
