"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140  # safety margin before the 150s wall-clock limit
    rng = random.Random(42)
    np.random.seed(42)

    n, m, p = DIMENSIONS  # n=2, m=2, p=8
    nm, mp, pn = n * m, m * p, p * n

    # Build the full tensor T[a,b,c] = 1 iff a=i*m+j, b=j*p+k, c=k*n+i, else 0.
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    # Schoolbook decomposition of rank n*m*p (guaranteed valid baseline).
    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank), dtype=np.float64)
        V = np.zeros((mp, rank), dtype=np.float64)
        W = np.zeros((pn, rank), dtype=np.float64)
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, c] = 1.0
                    V[j * p + k, c] = 1.0
                    W[k * n + i, c] = 1.0
                    c += 1
        return U, V, W

    best_U, best_V, best_W = schoolbook()
    best_rank = n * m * p  # 32

    # Verify a decomposition: sum_r U[a,r]*V[b,r]*W[c,r] == T[a,b,c]
    def verify(U, V, W, tol=1e-8):
        rank = U.shape[1]
        # Compute tensor from decomposition
        T_decomp = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T_decomp, T, atol=tol)

    # Alternating Least Squares for tensor decomposition
    def als_decompose(T_tensor, rank, max_iter=50, tol=1e-10):
        """
        ALS decomposition: T ≈ sum_r U_r ⊗ V_r ⊗ W_r
        """
        nm, mp, pn = T_tensor.shape
        
        # Initialize randomly
        U = np.random.randn(nm, rank) * 0.1
        V = np.random.randn(mp, rank) * 0.1
        W = np.random.randn(pn, rank) * 0.1
        
        prev_residual = np.inf
        for it in range(max_iter):
            if time.time() > deadline:
                break
            
            # Update U: for each a, minimize ||T[a,:,:] - U[a,:] * (V⊗W)||^2
            # T[a,:,:] = sum_r U[a,r] V[:,r] W[:,r]^T
            # For fixed a: T[a,:,:] = U[a,:] @ (V⊗W) where V⊗W is (mp*pn, rank)
            VW = np.einsum('br,cr->bcr', V, W).reshape(mp * pn, rank)
            for a in range(nm):
                if np.any(U[a, :] != 0) or np.any(T_tensor[a, :, :] != 0):
                    # Solve: U[a,:] @ VW ≈ T[a,:,:].flatten()
                    # This is a least squares problem: min ||U[a,:] @ VW - target||
                    # Equivalent to: VW^T @ U[a,:]^T ≈ target
                    # So: U[a,:] = target @ (VW @ VW^T)^{-1} @ VW^T
                    target = T_tensor[a, :, :].flatten()
                    A = VW.T @ VW  # (rank, rank)
                    b = VW.T @ target  # (rank,)
                    try:
                        U[a, :] = np.linalg.solve(A + 1e-12 * np.eye(rank), b)
                    except np.linalg.LinAlgError:
                        U[a, :] = np.linalg.lstsq(A, b, rcond=None)[0]
            
            # Update V: for each b, minimize ||T[:,b,:] - (U⊗W) @ V[b,:]||^2
            UW = np.einsum('ar,cr->acr', U, W).reshape(nm * pn, rank)
            for b in range(mp):
                if np.any(V[b, :] != 0) or np.any(T_tensor[:, b, :] != 0):
                    target = T_tensor[:, b, :].flatten()
                    A = UW.T @ UW
                    b_vec = UW.T @ target
                    try:
                        V[b, :] = np.linalg.solve(A + 1e-12 * np.eye(rank), b_vec)
                    except np.linalg.LinAlgError:
                        V[b, :] = np.linalg.lstsq(A, b_vec, rcond=None)[0]
            
            # Update W: for each c, minimize ||T[:,:,c] - (U⊗V) @ W[c,:]||^2
            UV = np.einsum('ar,br->abr', U, V).reshape(nm * mp, rank)
            for c in range(pn):
                if np.any(W[c, :] != 0) or np.any(T_tensor[:, :, c] != 0):
                    target = T_tensor[:, :, c].flatten()
                    A = UV.T @ UV
                    b_vec = UV.T @ target
                    try:
                        W[c, :] = np.linalg.solve(A + 1e-12 * np.eye(rank), b_vec)
                    except np.linalg.LinAlgError:
                        W[c, :] = np.linalg.lstsq(A, b_vec, rcond=None)[0]
            
            # Compute residual
            T_decomp = np.einsum('ar,br,cr->abc', U, V, W)
            residual = np.sum((T_decomp - T_tensor) ** 2)
            
            if abs(prev_residual - residual) < tol:
                break
            prev_residual = residual
        
        return U, V, W, residual

    # Try various ranks from low to high
    # Theoretical lower bound for matrix multiplication tensor rank is at least max(nm, mp, pn) = 16
    # But for this specific tensor, the rank might be higher
    # Let's try ranks from 16 to 31
    
    for target_rank in range(16, best_rank + 1):
        if time.time() > deadline:
            break
        
        rank = target_rank
        found = False
        
        # Multiple restarts with different initializations
        num_restarts = min(10, max(3, int((deadline - time.time()) / 5)))
        
        for restart in range(num_restarts):
            if time.time() > deadline:
                break
            
            # Try different initialization strategies
            if restart == 0:
                # Random initialization
                U_init = np.random.randn(nm, rank) * 0.1
                V_init = np.random.randn(mp, rank) * 0.1
                W_init = np.random.randn(pn, rank) * 0.1
            elif restart == 1:
                # Structured initialization based on tensor structure
                U_init = np.zeros((nm, rank))
                V_init = np.zeros((mp, rank))
                W_init = np.zeros((pn, rank))
                # Initialize based on the sparsity pattern
                for r in range(rank):
                    i = r % n
                    j = (r // n) % m
                    k = (r // (n * m)) % p
                    U_init[i * m + j, r] = 1.0
                    V_init[j * p + k, r] = 1.0
                    W_init[k * n + i, r] = 1.0
            else:
                # Perturbed structured initialization
                U_init = np.zeros((nm, rank))
                V_init = np.zeros((mp, rank))
                W_init = np.zeros((pn, rank))
                for r in range(rank):
                    i = r % n
                    j = (r // n) % m
                    k = (r // (n * m)) % p
                    U_init[i * m + j, r] = 1.0 + np.random.randn() * 0.1
                    V_init[j * p + k, r] = 1.0 + np.random.randn() * 0.1
                    W_init[k * n + i, r] = 1.0 + np.random.randn() * 0.1
            
            # Use ALS
            U_opt, V_opt, W_opt, residual = als_decompose(T, rank, max_iter=30, tol=1e-12)
            
            if residual < 1e-8:
                # Verify
                if verify(U_opt, V_opt, W_opt, tol=1e-7):
                    # Try to round to half-integers
                    U_rounded = np.round(U_opt * 2) / 2
                    V_rounded = np.round(V_opt * 2) / 2
                    W_rounded = np.round(W_opt * 2) / 2
                    
                    if verify(U_rounded, V_rounded, W_rounded, tol=1e-6):
                        best_U = U_rounded
                        best_V = V_rounded
                        best_W = W_rounded
                        best_rank = target_rank
                        found = True
                        break
                    else:
                        # Use unrounded if it verifies
                        best_U = U_opt
                        best_V = V_opt
                        best_W = W_opt
                        best_rank = target_rank
                        found = True
                        break
        
        if found:
            # Found a valid decomposition at this rank
            # Continue to see if we can find a lower rank
            pass
    
    # If no better rank was found, return schoolbook
    if best_rank == n * m * p:
        U, V, W = schoolbook()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
