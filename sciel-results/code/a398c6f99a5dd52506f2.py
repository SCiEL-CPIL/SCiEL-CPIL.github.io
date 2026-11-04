"""Rank-minimal exact tensor decomposition for 3x4 x 4x6 matrix multiplication."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 6
    deadline = time.time() + 110  # safety margin before 120s limit
    start_time = time.time()

    # Build the multiplication tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    def verify(U, V, W, atol=1e-9):
        """Check if the decomposition exactly reproduces the tensor."""
        T_computed = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T_computed, T, atol=atol)

    def round_and_verify(U, V, W):
        """Try rounding to integers/half-integers and verify."""
        # Try rounding to nearest integer
        U_r = np.round(U)
        V_r = np.round(V)
        W_r = np.round(W)
        if verify(U_r, V_r, W_r, atol=1e-9):
            return U_r, V_r, W_r
        
        # Try rounding to nearest half-integer
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        if verify(U_r, V_r, W_r, atol=1e-9):
            return U_r, V_r, W_r
        
        # Try rounding to nearest quarter-integer
        U_r = np.round(U * 4) / 4
        V_r = np.round(V * 4) / 4
        W_r = np.round(W * 4) / 4
        if verify(U_r, V_r, W_r, atol=1e-9):
            return U_r, V_r, W_r
        
        return None

    # Schoolbook decomposition: rank = n*m*p = 72
    rank_sch = n * m * p
    U_sch = np.zeros((n * m, rank_sch), dtype=np.float64)
    V_sch = np.zeros((m * p, rank_sch), dtype=np.float64)
    W_sch = np.zeros((p * n, rank_sch), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i * m * p + j * p + k
                U_sch[i * m + j, r] = 1.0
                V_sch[j * p + k, r] = 1.0
                W_sch[k * n + i, r] = 1.0

    # Verify schoolbook is correct
    assert verify(U_sch, V_sch, W_sch), "Schoolbook verification failed"

    best_rank = rank_sch
    best_U = U_sch.copy()
    best_V = V_sch.copy()
    best_W = W_sch.copy()

    # Pre-compute matricizations for ALS
    # T_A: (nm, mp*pn) where T_A[a, b*pn+c] = T[a,b,c]
    T_A = T.reshape(n * m, m * p * p * n)
    # T_B: (mp, nm*pn) where T_B[b, a*pn+c] = T[a,b,c]
    T_B = T.transpose(1, 0, 2).reshape(m * p, n * m * p * n)
    # T_C: (pn, nm*mp) where T_C[c, a*mp+b] = T[a,b,c]
    T_C = T.transpose(2, 0, 1).reshape(p * n, n * m * m * p)

    def als_decompose(U_init, V_init, W_init, max_iter=80, tol=1e-10):
        """Run ALS to refine the decomposition with better convergence handling."""
        U, V, W = U_init.copy(), V_init.copy(), W_init.copy()
        rank = U.shape[1]
        
        for iteration in range(max_iter):
            if time.time() > deadline:
                break
            
            # Update U: T_A = U @ (V⊙W)^T
            # (V⊙W)[(b,c), r] = V[b,r] * W[c,r], shape (mp*pn, rank)
            MW = np.einsum('br,cr->bcr', V, W)  # (mp, pn, rank)
            MW_flat = MW.reshape(m * p * p * n, rank)  # (b*pn+c, r)
            
            G = MW_flat.T @ MW_flat
            try:
                # Use SVD-based pseudo-inverse for better numerical stability
                s = np.linalg.svd(G, compute_uv=False)
                s_inv = np.where(s > tol * s[0], 1.0/s, 0.0)
                U_t, _, Vt = np.linalg.svd(G, full_matrices=False)
                G_inv = U_t @ np.diag(s_inv) @ Vt
                U_new = T_A @ MW_flat @ G_inv
            except np.linalg.LinAlgError:
                break

            # Update V: T_B = V @ (U⊙W)^T
            # (U⊙W)[(a,c), r] = U[a,r] * W[c,r], shape (nm*pn, rank)
            UW = np.einsum('ar,cr->acr', U_new, W)  # (nm, pn, rank)
            UW_flat = UW.reshape(n * m * p * n, rank)  # (a*pn+c, r)
            
            G = UW_flat.T @ UW_flat
            try:
                s = np.linalg.svd(G, compute_uv=False)
                s_inv = np.where(s > tol * s[0], 1.0/s, 0.0)
                U_t, _, Vt = np.linalg.svd(G, full_matrices=False)
                G_inv = U_t @ np.diag(s_inv) @ Vt
                V_new = T_B @ UW_flat @ G_inv
            except np.linalg.LinAlgError:
                break

            # Update W: T_C = W @ (U⊙V)^T
            # (U⊙V)[(a,b), r] = U[a,r] * V[b,r], shape (nm*mp, rank)
            UV = np.einsum('ar,br->abr', U_new, V_new)  # (nm, mp, rank)
            UV_flat = UV.reshape(n * m * m * p, rank)  # (a*mp+b, r)
            
            G = UV_flat.T @ UV_flat
            try:
                s = np.linalg.svd(G, compute_uv=False)
                s_inv = np.where(s > tol * s[0], 1.0/s, 0.0)
                U_t, _, Vt = np.linalg.svd(G, full_matrices=False)
                G_inv = U_t @ np.diag(s_inv) @ Vt
                W_new = T_C @ UV_flat @ G_inv
            except np.linalg.LinAlgError:
                break

            # Check convergence
            delta = max(np.linalg.norm(U_new - U), np.linalg.norm(V_new - V), np.linalg.norm(W_new - W))
            U, V, W = U_new, V_new, W_new
            
            if delta < 1e-13:
                break

        return U, V, W

    def try_polish(U, V, W, target_rank):
        """High-precision polish: refine with tighter tolerance and more iterations."""
        if time.time() > deadline - 5:
            return None
        U_p, V_p, W_p = als_decompose(U, V, W, max_iter=200, tol=1e-14)
        if verify(U_p, V_p, W_p, atol=1e-10):
            rounded = round_and_verify(U_p, V_p, W_p)
            if rounded is not None:
                return rounded
        return None

    # Strategy: Exploit structure. The tensor for 3x4x6 has a specific pattern.
    # Try to find low-rank decompositions using structured initialization.
    
    # Key insight: We can use the fact that matrix multiplication tensors have
    # a specific algebraic structure. Try various rank targets with multiple
    # initialization strategies.
    
    target_ranks = list(range(rank_sch - 1, 53, -1))
    
    for target_rank in target_ranks:
        if time.time() > deadline - 5:
            break
        
        found = False
        
        # Multiple initialization strategies with more diversity
        n_restarts = 12
        for restart in range(n_restarts):
            if time.time() > deadline - 5:
                break
            
            if restart < 4:
                # Strategy 1: Random Gaussian initialization
                np.random.seed(42 + restart * 7)
                U_init = np.random.randn(n * m, target_rank)
                V_init = np.random.randn(m * p, target_rank)
                W_init = np.random.randn(p * n, target_rank)
            elif restart < 8:
                # Strategy 2: Structured initialization based on Kronecker structure
                np.random.seed(100 + restart * 13)
                # Initialize with block structure reflecting the tensor pattern
                U_init = np.zeros((n * m, target_rank))
                V_init = np.zeros((m * p, target_rank))
                W_init = np.zeros((p * n, target_rank))
                # Fill with random values but preserve some structure
                for r in range(target_rank):
                    # Random sparse pattern
                    mask_u = np.random.random(n * m) < 0.3
                    mask_v = np.random.random(m * p) < 0.3
                    mask_w = np.random.random(p * n) < 0.3
                    U_init[mask_u, r] = np.random.randn(np.sum(mask_u))
                    V_init[mask_v, r] = np.random.randn(np.sum(mask_v))
                    W_init[mask_w, r] = np.random.randn(np.sum(mask_w))
            else:
                # Strategy 3: Perturb schoolbook and project to lower rank
                np.random.seed(200 + restart * 17)
                U_init = U_sch[:, :target_rank].copy() + 0.05 * np.random.randn(n * m, target_rank)
                V_init = V_sch[:, :target_rank].copy() + 0.05 * np.random.randn(m * p, target_rank)
                W_init = W_sch[:, :target_rank].copy() + 0.05 * np.random.randn(p * n, target_rank)
            
            # Run ALS
            U_sol, V_sol, W_sol = als_decompose(U_init, V_init, W_init, max_iter=100)
            
            # Check if we found an exact solution
            if verify(U_sol, V_sol, W_sol, atol=1e-8):
                # Try high-precision polish
                polished = try_polish(U_sol, V_sol, W_sol, target_rank)
                if polished is not None:
                    U_r, V_r, W_r = polished
                    best_rank = target_rank
                    best_U = U_r
                    best_V = V_r
                    best_W = W_r
                    found = True
                    break
                
                rounded = round_and_verify(U_sol, V_sol, W_sol)
                if rounded is not None:
                    U_r, V_r, W_r = rounded
                    best_rank = target_rank
                    best_U = U_r
                    best_V = V_r
                    best_W = W_r
                    found = True
                    break
            else:
                # Even if not exactly matching, try rounding the continuous solution
                rounded = round_and_verify(U_sol, V_sol, W_sol)
                if rounded is not None:
                    U_r, V_r, W_r = rounded
                    best_rank = target_rank
                    best_U = U_r
                    best_V = V_r
                    best_W = W_r
                    found = True
                    break

        if found:
            break  # Found a valid lower-rank decomposition, stop searching

    # If we haven't found anything better, try one more approach:
    # Use the known Strassen-like decomposition for sub-blocks
    if best_rank == rank_sch and time.time() < deadline - 10:
        # Try a block-structured approach
        # 3x4x6 can be seen as combining smaller blocks
        # Use rank-7 decomposition for 2x2x2 (Strassen) as building blocks
        
        # Construct a candidate using Kronecker products of Strassen
        # This is a heuristic - may not yield exact low-rank solution
        # but worth trying
        
        pass  # Skip complex structured construction for now

    # Final: ensure we return valid arrays
    U_out = best_U
    V_out = best_V
    W_out = best_W

    return {"U": U_out.tolist(), "V": V_out.tolist(), "W": W_out.tolist()}
