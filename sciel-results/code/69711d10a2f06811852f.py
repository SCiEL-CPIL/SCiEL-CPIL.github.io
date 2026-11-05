import numpy as np
import time

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    # Target Tensor T of shape (dim_u, dim_v, dim_w)
    T = np.zeros((dim_u, dim_v, dim_w), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0

    start_time = time.time()
    timeout = 140.0  # Leave 10s margin before 150s kill

    def check_solution(U, V, W):
        """Check if U, V, W (integer or half-integer) exactly decompose T."""
        Uf = np.asarray(U, dtype=np.float64)
        Vf = np.asarray(V, dtype=np.float64)
        Wf = np.asarray(W, dtype=np.float64)
        
        # Reconstruct tensor
        T_recon = np.einsum('uc,vc,wc->uvw', Uf, Vf, Wf)
        
        # Check if it matches T exactly
        if np.allclose(T_recon, T, atol=1e-9):
            return True
        return False

    def als_decompose(rank, max_iters=500, tol=1e-10):
        """Alternating Least Squares for tensor decomposition."""
        # Initialize randomly
        U = np.random.randn(dim_u, rank)
        V = np.random.randn(dim_v, rank)
        W = np.random.randn(dim_w, rank)
        
        prev_loss = np.inf
        for it in range(max_iters):
            if time.time() - start_time > timeout:
                return None
            
            # Update U: for each c, solve least squares
            for c in range(rank):
                # Fix V[:, c] and W[:, c], solve for U[:, c]
                # T[u, v, w] = sum_c U[u,c] V[v,c] W[w,c]
                # For fixed c, we want to minimize sum_{u,v,w} (T[u,v,w] - U[u,c]*V[v,c]*W[w,c] - sum_{c'!=c} U[u,c']*V[v,c']*W[w,c'])^2
                # This is equivalent to: for each u, sum_v,w (T[u,v,w] - residual[u,v,w]) * V[v,c] * W[w,c] = 0
                # where residual = T - sum_{c'!=c} U[u,c']*V[v,c']*W[w,c]
                
                # Compute residual excluding column c
                T_recon = np.einsum('uc,vc,wc->uvw', U, V, W)
                T_recon -= np.einsum('uc,vc,wc->uvw', 
                    np.delete(U, c, axis=1), 
                    np.delete(V, c, axis=1), 
                    np.delete(W, c, axis=1))
                # Now T_recon = T - sum_{c'!=c} U[u,c']*V[v,c']*W[w,c]
                # We want U[:,c] to minimize sum_u (T_recon[u,:,:] - U[u,c]*outer(V[:,c], W[:,c]))^2
                # This is a linear least squares problem for each u
                
                # For efficiency, compute the target for U[:,c]
                # target[u] = sum_{v,w} T_recon[u,v,w] * V[v,c] * W[w,c] / sum_{v,w} (V[v,c]*W[w,c])^2
                # But this assumes V[:,c] and W[:,c] are fixed
                
                # Actually, simpler: use matrix form
                # Let A = outer(V[:,c], W[:,c]) of shape (dim_v, dim_w)
                # Then for each u, we want to find U[u,c] to minimize sum_{v,w} (T_recon[u,v,w] - U[u,c]*A[v,w])^2
                # Solution: U[u,c] = sum_{v,w} T_recon[u,v,w]*A[v,w] / sum_{v,w} A[v,w]^2
                
                A = np.outer(V[:, c], W[:, c])
                denom = np.sum(A * A)
                if denom < 1e-12:
                    U[:, c] = 0
                    continue
                # numerator[u] = sum_{v,w} T_recon[u,v,w] * A[v,w]
                numerator = np.einsum('uvw,vw->u', T_recon, A)
                U[:, c] = numerator / denom
            
            # Similarly update V and W
            for c in range(rank):
                T_recon = np.einsum('uc,vc,wc->uvw', U, V, W)
                T_recon -= np.einsum('uc,vc,wc->uvw', 
                    np.delete(U, c, axis=1), 
                    np.delete(V, c, axis=1), 
                    np.delete(W, c, axis=1))
                
                A = np.outer(U[:, c], W[:, c])
                denom = np.sum(A * A)
                if denom < 1e-12:
                    V[:, c] = 0
                    continue
                numerator = np.einsum('uvw,uw->v', T_recon, A)
                V[:, c] = numerator / denom
            
            for c in range(rank):
                T_recon = np.einsum('uc,vc,wc->uvw', U, V, W)
                T_recon -= np.einsum('uc,vc,wc->uvw', 
                    np.delete(U, c, axis=1), 
                    np.delete(V, c, axis=1), 
                    np.delete(W, c, axis=1))
                
                A = np.outer(U[:, c], V[:, c])
                denom = np.sum(A * A)
                if denom < 1e-12:
                    W[:, c] = 0
                    continue
                numerator = np.einsum('uvw,uv->w', T_recon, A)
                W[:, c] = numerator / denom
            
            # Compute loss
            T_recon = np.einsum('uc,vc,wc->uvw', U, V, W)
            loss = np.sum((T - T_recon) ** 2)
            
            if abs(prev_loss - loss) < tol:
                break
            prev_loss = loss
        
        # Check convergence
        T_recon = np.einsum('uc,vc,wc->uvw', U, V, W)
        error = np.linalg.norm(T - T_recon)
        if error < 1e-6:
            return U, V, W
        return None

    def try_rank_decomposition(rank, n_restarts=10):
        """Try to find a decomposition of given rank using multiple restarts."""
        for restart in range(n_restarts):
            if time.time() - start_time > timeout - 5:
                return None
            
            np.random.seed(restart * 1000 + rank)
            result = als_decompose(rank)
            if result is not None:
                U, V, W = result
                
                # Try to snap to half-integers
                U_snap = np.round(U * 2) / 2
                V_snap = np.round(V * 2) / 2
                W_snap = np.round(W * 2) / 2
                
                if check_solution(U_snap, V_snap, W_snap):
                    return U_snap, V_snap, W_snap
                
                # Try integer snap
                U_int = np.round(U)
                V_int = np.round(V)
                W_int = np.round(W)
                if check_solution(U_int, V_int, W_int):
                    return U_int, V_int, W_int
        
        return None

    # Initialize with schoolbook as fallback
    schoolbook_rank = n * m * p
    U_school = np.zeros((dim_u, schoolbook_rank))
    V_school = np.zeros((dim_v, schoolbook_rank))
    W_school = np.zeros((dim_w, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1

    # Search from low rank upward
    # Known lower bounds: rank >= max(n*m, m*p, p*n) = max(12, 24, 18) = 24
    # Also rank >= (n*m*p) / min(n, m, p) = 72 / 3 = 24
    # Try ranks from 24 to 71
    for rank in range(24, schoolbook_rank):
        if time.time() - start_time > timeout - 10:
            break
        
        result = try_rank_decomposition(rank, n_restarts=5)
        if result is not None:
            U, V, W = result
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
    
    # If no better solution found, return schoolbook
    return {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
# EVOLVE-BLOCK-END
