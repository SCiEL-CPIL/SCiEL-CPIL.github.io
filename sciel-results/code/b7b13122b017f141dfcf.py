import numpy as np
from itertools import product
import time

DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Build the multiplication tensor T of shape (n*m, m*p, p*n)
    # T[i,j,k] = 1 if the term A[i,j]*B[j,k] contributes to C[i,k]
    # Index: A row index i (0..n-1), B col index k (0..p-1)
    # U index: i*m + j  (row of A, col of B that matches row of B)
    # V index: j*p + k  (row of B, col of B)
    # W index: k*n + i  (col of B, row of C) -> trace convention W[k*n+i]
    
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0
    
    deadline = time.time() + 130.0  # safety margin
    
    def snap_to_halfint(x):
        """Snap values to nearest half-integer."""
        return np.round(x * 2.0) / 2.0
    
    def verify(U, V, W):
        """Check if U, V, W exactly decompose T."""
        rank = U.shape[1]
        # Reconstruct tensor: T_recon[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
        # Use efficient einsum
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T_recon, T, atol=1e-10)
    
    def residual(U, V, W):
        rank = U.shape[1]
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        return np.linalg.norm(T_recon - T)
    
    best_rank = nm * mp  # schoolbook
    best_result = None
    
    # Schoolbook as fallback
    U_sb = np.zeros((nm, best_rank))
    V_sb = np.zeros((mp, best_rank))
    W_sb = np.zeros((pn, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    best_result = (U_sb.copy(), V_sb.copy(), W_sb.copy())
    
    # Try decreasing ranks
    for rank in range(nm*mp - 1, max(20, nm*mp - 30), -1):
        if time.time() > deadline - 30:
            break
        
        # Multiple random restarts for each rank
        n_restarts = 5
        for restart in range(n_restarts):
            if time.time() > deadline - 10:
                break
            
            # Random initialization
            U = np.random.randn(nm, rank)
            V = np.random.randn(mp, rank)
            W = np.random.randn(pn, rank)
            
            # Gradient-free iterative refinement (alternating least squares style)
            # For fixed V, W: U[a,r] = sum_{b,c} T[a,b,c] * V[b,r] * W[c,r] / (V[:,r]^2 * W[:,r]^2 + eps)
            # But this is for exact fit via least squares on the flattened tensor
            
            # Simpler: use coordinate descent / alternating minimization
            # Minimize ||T - sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r]||^2
            
            for it in range(100):
                if time.time() > deadline - 10:
                    break
                
                # Update U
                for r in range(rank):
                    # U[:,r] should satisfy: for each a, sum_{b,c} T[a,b,c]*V[b,r]*W[c,r] = sum_b sum_c T[a,b,c]*V[b,r]*W[c,r]
                    # This is a linear system in U[:,r] if we fix V[:,r] and W[:,r]
                    # Actually: T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
                    # For fixed V[:,r'], W[:,r'] for r' != r, and V[:,r], W[:,r]:
                    # U[a,r] * V[b,r] * W[c,r] = T[a,b,c] - sum_{r'!=r} U[a,r']*V[b,r']*W[c,r']
                    # This is still coupled. Let's use a different approach.
                    
                    # Use the fact that for a rank-1 component, we can do:
                    # Fix V[:,r] and W[:,r], then U[:,r] is determined by least squares
                    # on the residual tensor.
                    pass
                
                # Better approach: alternating least squares on the tensor
                # Flatten and use SVD-like updates
                T_recon = np.einsum('ar,br,cr->abc', U, V, W)
                R = T - T_recon  # residual
                
                # Update U: for each r, U[:,r] minimizes ||R + U[:,r]⊗V[:,r]⊗W[:,r]||
                # This is: U[:,r] = argmin sum_{a,b,c} (R[a,b,c] + U[a,r]*V[b,r]*W[c,r])^2
                # = argmin sum_a (sum_{b,c} R[a,b,c]*V[b,r]*W[c,r] + U[a,r]*sum_{b,c} V[b,r]^2*W[c,r]^2)^2
                # Linear in U[a,r]!
                # Let M[a] = sum_{b,c} R[a,b,c]*V[b,r]*W[c,r]
                # Let D = sum_{b,c} V[b,r]^2 * W[c,r]^2
                # Then U[a,r] = -M[a] / D
                for r in range(rank):
                    # M[a] = sum_{b,c} R[a,b,c]*V[b,r]*W[c,r]
                    # R[a,b,c] * V[b,r] * W[c,r], sum over b,c
                    # = sum_b sum_c R[a,b,c] * V[b,r] * W[c,r]
                    # = (R * V[:,r] * W[:,r]) summed over b,c
                    # Using einsum: M[a] = einsum('abc,br,cr->a', R, V[:,r], W[:,r])
                    M = np.einsum('abc,br,cr->a', R, V[:,r], W[:,r])
                    D = np.sum(V[:,r]**2) * np.sum(W[:,r]**2)
                    if D > 1e-15:
                        U[:,r] = -M / D
                
                # Update V similarly
                T_recon = np.einsum('ar,br,cr->abc', U, V, W)
                R = T - T_recon
                for r in range(rank):
                    M = np.einsum('abc,ar,cr->b', R, U[:,r], W[:,r])
                    D = np.sum(U[:,r]**2) * np.sum(W[:,r]**2)
                    if D > 1e-15:
                        V[:,r] = -M / D
                
                # Update W similarly
                T_recon = np.einsum('ar,br,cr->abc', U, V, W)
                R = T - T_recon
                for r in range(rank):
                    M = np.einsum('abc,ar,br->c', R, U[:,r], V[:,r])
                    D = np.sum(U[:,r]**2) * np.sum(V[:,r]**2)
                    if D > 1e-15:
                        W[:,r] = -M / D
                
                # Check convergence
                if it % 10 == 9:
                    res = residual(U, V, W)
                    if res < 1e-8:
                        # Try to snap to half-integers
                        U_s = snap_to_halfint(U)
                        V_s = snap_to_halfint(V)
                        W_s = snap_to_halfint(W)
                        if verify(U_s, V_s, W_s):
                            best_rank = rank
                            best_result = (U_s.copy(), V_s.copy(), W_s.copy())
                            break
                        # Also try without snapping (check if already close to half-integers)
                        if verify(U, V, W):
                            # Check if values are already half-integers
                            if np.allclose(U, snap_to_halfint(U), atol=1e-8) and \
                               np.allclose(V, snap_to_halfint(V), atol=1e-8) and \
                               np.allclose(W, snap_to_halfint(W), atol=1e-8):
                                best_rank = rank
                                best_result = (snap_to_halfint(U), snap_to_halfint(V), snap_to_halfint(W))
                                break
                if best_rank == rank:
                    break
            
            if best_rank == rank:
                break
        
        if best_rank == rank:
            break
    
    # If no better found, return schoolbook
    U, V, W = best_result
    
    # Convert to lists
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
