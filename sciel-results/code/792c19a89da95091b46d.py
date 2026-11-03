import numpy as np
import time

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Build the target tensor
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    start_time = time.time()
    deadline = 120.0
    safety = 5.0
    
    # Schoolbook fallback (rank = n*m*p = 42)
    rank_sb = n * m * p
    U_sb = np.zeros((nm, rank_sb))
    V_sb = np.zeros((mp, rank_sb))
    W_sb = np.zeros((pn, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1.0
                V_sb[j * p + k, col] = 1.0
                W_sb[k * n + i, col] = 1.0
                col += 1
    
    def verify(U, V, W, tol=1e-9):
        """Check if U, V, W exactly decompose T."""
        P = np.einsum('ur,vr,wr->uvw', U, V, W)
        return np.allclose(P, T, atol=tol)
    
    def is_half_integer(M, tol=1e-9):
        """Check if entries are half-integers."""
        doubled = 2.0 * np.array(M)
        return np.allclose(doubled, np.round(doubled), atol=tol)
    
    def project_half_integer(U, V, W):
        """Project to nearest half-integers."""
        return (np.round(2.0 * U) / 2.0, 
                np.round(2.0 * V) / 2.0, 
                np.round(2.0 * W) / 2.0)
    
    def alternating_minimization(target_rank, n_restarts=10, max_iter=200):
        """
        Try to find a rank-1 decomposition of T using alternating minimization.
        Returns (U, V, W) if successful, else None.
        """
        for restart in range(n_restarts):
            if time.time() > deadline - safety - 5:
                return None
            
            rng = np.random.RandomState(restart * 1000 + 42)
            
            # Initialize with random vectors
            U = rng.randn(nm, target_rank) * 0.1
            V = rng.randn(mp, target_rank) * 0.1
            W = rng.randn(pn, target_rank) * 0.1
            
            # Alternating minimization
            for it in range(max_iter):
                if time.time() > deadline - safety - 3:
                    return None
                
                # Update U: for each column r, U[:,r] should be proportional to T contracted with V[:,r] and W[:,r]
                for r in range(target_rank):
                    # S_u[i] = sum_{j,k} T[i,j,k] * V[j,r] * W[k,r]
                    S_u = np.einsum('ijk,j,k->i', T, V[:, r], W[:, r])
                    norm = np.linalg.norm(S_u)
                    if norm > 1e-12:
                        U[:, r] = S_u / norm
                
                # Update V
                for r in range(target_rank):
                    S_v = np.einsum('ijk,i,k->j', T, U[:, r], W[:, r])
                    norm = np.linalg.norm(S_v)
                    if norm > 1e-12:
                        V[:, r] = S_v / norm
                
                # Update W
                for r in range(target_rank):
                    S_w = np.einsum('ijk,i,j->k', T, U[:, r], V[:, r])
                    norm = np.linalg.norm(S_w)
                    if norm > 1e-12:
                        W[:, r] = S_w / norm
                
                # Check convergence
                if it % 10 == 9:
                    P = np.einsum('ur,vr,wr->uvw', U, V, W)
                    err = np.linalg.norm(P - T)
                    if err < 1e-8:
                        break
            
            # Verify
            P = np.einsum('ur,vr,wr->uvw', U, V, W)
            if np.allclose(P, T, atol=1e-8):
                # Try to project to half-integers
                U_h, V_h, W_h = project_half_integer(U, V, W)
                if verify(U_h, V_h, W_h, tol=1e-9) and is_half_integer(U_h) and is_half_integer(V_h) and is_half_integer(W_h):
                    return (U_h, V_h, W_h)
                # If projection doesn't work, check if original is valid
                if verify(U, V, W, tol=1e-9):
                    return (U, V, W)
        
        return None
    
    # Try ranks from the theoretical lower bound up to schoolbook
    # Theoretical lower bound: max(n*m, m*p, p*n) = max(6, 21, 14) = 21
    lower_bound = max(n * m, m * p, p * n)
    
    best_rank = rank_sb
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # Try ranks from lower_bound to rank_sb-1
    for rank in range(lower_bound, rank_sb):
        if time.time() > deadline - safety - 10:
            break
        
        result = alternating_minimization(rank, n_restarts=5, max_iter=100)
        if result is not None:
            U_opt, V_opt, W_opt = result
            best_rank = rank
            best_U, best_V, best_W = U_opt, V_opt, W_opt
            break  # Found a valid decomposition, can stop
    
    # Return the best result
    if best_rank < rank_sb:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
