import time
import numpy as np

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115  # safety margin before 150s limit

    # Known theoretical lower bound for 2x3 * 3x4 matrix multiplication is 20.
    # The schoolbook algorithm has rank 24.
    # We attempt to find a rank-20 decomposition via Alternating Least Squares (ALS)
    # with random restarts. If we find an exact (or near-exact) solution, we round
    # to integers or half-integers and verify.

    # Build the target tensor T of shape (n*m, m*p, p*n)
    # T[a, b, c] = 1 if the corresponding indices (i,j), (j,k), (k,i) are consistent
    # a = i*m + j, b = j*p + k, c = k*n + i
    nm = n * m
    mp = m * p
    pn = p * n

    T = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    T_flat = T.reshape(-1)
    total_dim = nm * mp * pn

    # Try ranks from 20 up to 23. If we find a solution at 20, we're done.
    # We focus on 20 first since it's the theoretical minimum.
    best_solution = None
    best_rank = nm * mp * pn  # schoolbook rank as fallback

    # Pre-compute schoolbook as fallback
    U_sb = np.zeros((nm, best_rank), dtype=float)
    V_sb = np.zeros((mp, best_rank), dtype=float)
    W_sb = np.zeros((pn, best_rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1

    def verify_and_round(U, V, W, tol=1e-8):
        """Check if the decomposition is exact within tolerance and round entries."""
        # Reconstruct tensor
        U_r, V_r, W_r = U, V, W
        T_recon = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
        diff = np.max(np.abs(T_recon - T))
        if diff > tol:
            return None
        
        # Round to nearest 0.5
        U_round = np.round(U * 2) / 2
        V_round = np.round(V * 2) / 2
        W_round = np.round(W * 2) / 2
        
        # Verify rounded version
        T_recon2 = np.einsum('ar,br,cr->abc', U_round, V_round, W_round)
        diff2 = np.max(np.abs(T_recon2 - T))
        if diff2 > tol:
            return None
        
        return (U_round, V_round, W_round)

    def als_decompose(rank, init_iters=50, max_outer_iters=200, tol=1e-10):
        """Run ALS to find a rank decomposition. Returns (U, V, W) or None if failed."""
        np.random.seed(42)  # Fixed seed for reproducibility within a run
        
        U = np.random.randn(nm, rank)
        V = np.random.randn(mp, rank)
        W = np.random.randn(pn, rank)
        
        # Normalize columns to avoid scaling issues
        for r in range(rank):
            norm_u = np.linalg.norm(U[:, r])
            norm_v = np.linalg.norm(V[:, r])
            norm_w = np.linalg.norm(W[:, r])
            total = norm_u * norm_v * norm_w
            if total > 1e-12:
                scale = total ** (1/3)
                U[:, r] /= scale
                V[:, r] /= scale
                W[:, r] /= scale
        
        for it in range(max_outer_iters):
            # Update U
            # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
            # For each a, T[a,:,:].flat = sum_r U[a,r] (V[:,r] * W[:,r])
            # Let S be mp*pn x rank, S[bc, r] = V[b,r] * W[c,r]
            # Then T[a].flat @ S.T @ inv(S.S.T) = U[a,:]
            # More efficiently, solve least squares.
            
            if time.time() > deadline - 5:
                break
            
            # Update U
            # For each a: T[a,:,:].flat is a vector of length mp*pn
            # We want U[a,:] such that U[a,:] @ S = T[a].flat
            # where S[:,r] = V[:,r] * W[:,r] (flattened)
            S_vw = (V[:, :, np.newaxis] * W[:, np.newaxis, :]).reshape(mp * pn, rank)
            # S_vw is (mp*pn, rank). We want U[a,:] @ S_vw = T[a].flat
            # This is U[a,:] = T[a].flat @ S_vw.T @ inv(S_vw @ S_vw.T)
            # But S_vw might be rank deficient. Use lstsq or pseudo-inverse.
            # Better: solve S_vw.T @ U[a,:].T = T[a].flat.T ? No.
            # Let's use: U[a,:] = argmin || T[a].flat - U[a,:] @ S_vw ||
            # This is a linear system: S_vw.T @ U[a,:].T = T[a,:].flat.T
            # Actually, it's simpler to just use np.linalg.lstsq
            # But doing it for each a is slow. 
            # Alternative: Vectorize.
            # T_flat shape: (nm*mp*pn,)
            # Let's reshape T to (nm, mp*pn)
            T_nm = T.reshape(nm, mp * pn)
            # For each a, solve: S_vw.T @ u_a = T_nm[a,:]
            # Where u_a is a vector of length rank.
            # This is a batched least squares problem.
            # We can form a big matrix A of shape (nm, mp*pn) and B of shape (nm, rank)?
            # No, the system is A_a^T x = b_a for each a.
            # A = S_vw (same for all a). B = T_nm.T (shape mp*pn x nm).
            # We want X (shape rank x nm) such that S_vw.T @ X = B.
            # So X = lstsq(S_vw.T, B)
            try:
                U_new, _, _, _ = np.linalg.lstsq(S_vw.T, T_nm.T, rcond=None)
                U = U_new.T  # shape (nm, rank)
            except:
                U = U  # keep old if fails
            
            # Update V
            # T[a,b,c] = sum_r V[b,r] (U[a,r] * W[c,r])
            # For each b, T[:,b,:].flat = sum_r V[b,r] (U[:,r] * W[:,r])
            S_uw = (U[:, :, np.newaxis] * W[:, np.newaxis, :]).reshape(nm * pn, rank)
            # T reshaped to (mp, nm*pn)
            T_mp = T.reshape(mp, nm * pn).T  # shape (nm*pn, mp)
            try:
                V_new, _, _, _ = np.linalg.lstsq(S_uw.T, T_mp, rcond=None)
                V = V_new.T
            except:
                V = V
            
            # Update W
            # T[a,b,c] = sum_r W[c,r] (U[a,r] * V[b,r])
            # For each c, T[:,:,c].flat = sum_r W[c,r] (U[:,r] * V[:,r])
            S_uv = (U[:, :, np.newaxis] * V[:, np.newaxis, :]).reshape(nm * mp, rank)
            # T reshaped to (pn, nm*mp)
            T_pn = T.reshape(pn, nm * mp).T  # shape (nm*mp, pn)
            try:
                W_new, _, _, _ = np.linalg.lstsq(S_uv.T, T_pn, rcond=None)
                W = W_new.T
            except:
                W = W
            
            # Check convergence
            T_recon = np.einsum('ar,br,cr->abc', U, V, W)
            diff = np.linalg.norm(T_recon - T)
            if diff < tol:
                break
        
        return U, V, W

    # Try rank 20 first
    for rank in range(20, 24):
        if time.time() > deadline - 10:
            break
            
        for restart in range(50):
            if time.time() > deadline - 10:
                break
                
            np.random.seed(restart)
            U, V, W = als_decompose(rank, max_outer_iters=100)
            
            if U is not None:
                result = verify_and_round(U, V, W, tol=1e-6)
                if result is not None:
                    best_solution = result
                    best_rank = rank
                    break
        
        if best_solution is not None and best_rank == rank:
            break
    
    # If no better solution found, use schoolbook
    if best_solution is not None:
        U_final, V_final, W_final = best_solution
    else:
        U_final, V_final, W_final = U_sb, V_sb, W_sb

    return {
        "U": U_final.tolist(),
        "V": V_final.tolist(),
        "W": W_final.tolist()
    }
# EVOLVE-BLOCK-END
