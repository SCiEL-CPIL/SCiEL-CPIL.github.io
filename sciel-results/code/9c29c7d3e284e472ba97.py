import numpy as np
import time

DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110
    
    # Build target tensor
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Schoolbook fallback
    rank_school = n*m*p
    U_school = np.zeros((n*m, rank_school))
    V_school = np.zeros((m*p, rank_school))
    W_school = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    
    best_rank = rank_school
    best_U, best_V, best_W = U_school, V_school, W_school
    
    def verify(U, V, W):
        T_approx = np.einsum('ic,jc,kc->ijk', U, V, W)
        return np.allclose(T_approx, T, atol=1e-10)
    
    def round_half(X):
        return np.round(X * 2) / 2
    
    def als_solve(T, U, V, W, rank, max_iters=200):
        nr, nc, nk = T.shape
        prev_err = np.inf
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
            
            # Update U: T[i,j,k] = sum_c U[i,c] * V[j,c] * W[k,c]
            # For each i: T[i,:,:].flatten() = U[i,:] @ (V[:,c]*W[:,c] for c).T
            M_VW = np.einsum('jc,kc->ckj', V, W).reshape(rank, nc * nk)
            T_i = T.reshape(nr, nc * nk)
            try:
                U = T_i @ np.linalg.pinv(M_VW.T)
            except:
                break
            
            # Update V: For each j: T[:,j,:].flatten() = V[j,:] @ (U[:,c]*W[:,c] for c).T
            M_UW = np.einsum('ic,kc->cik', U, W).reshape(rank, nr * nk)
            T_j = np.zeros((nc, nr * nk))
            for j in range(nc):
                T_j[j, :] = T[:, j, :].flatten()
            try:
                V = T_j @ np.linalg.pinv(M_UW.T)
            except:
                break
            
            # Update W: For each k: T[:,:,k].flatten() = W[k,:] @ (U[:,c]*V[:,c] for c).T
            M_UV = np.einsum('ic,jc->cij', U, V).reshape(rank, nr * nc)
            T_k = np.zeros((nk, nr * nc))
            for k in range(nk):
                T_k[k, :] = T[:, :, k].flatten()
            try:
                W = T_k @ np.linalg.pinv(M_UV.T)
            except:
                break
            
            T_approx = np.einsum('ic,jc,kc->ijk', U, V, W)
            err = np.linalg.norm(T_approx - T)
            if abs(err - prev_err) < 1e-14:
                break
            prev_err = err
        
        return U, V, W
    
    # Try various ranks with multiple initializations
    for rank in [64, 56, 48, 40, 36]:
        if time.time() > deadline - 30:
            break
        
        found = False
        n_attempts = 10
        for attempt in range(n_attempts):
            if time.time() > deadline - 25:
                break
            
            # Different initialization strategies
            if attempt < 3:
                # Random Gaussian
                np.random.seed(attempt * 7919 + rank)
                U = np.random.randn(n*m, rank) * 0.3
                V = np.random.randn(m*p, rank) * 0.3
                W = np.random.randn(p*n, rank) * 0.3
            elif attempt < 6:
                # Sparse random with +/-1
                np.random.seed(attempt * 104729 + rank)
                U = np.zeros((n*m, rank))
                V = np.zeros((m*p, rank))
                W = np.zeros((p*n, rank))
                for c in range(rank):
                    U[np.random.randint(n*m), c] = np.random.choice([-1.0, 1.0])
                    V[np.random.randint(m*p), c] = np.random.choice([-1.0, 1.0])
                    W[np.random.randint(p*n), c] = np.random.choice([-1.0, 1.0])
            else:
                # Based on schoolbook subset
                np.random.seed(attempt * 15485863 + rank)
                idx = np.random.choice(rank_school, rank, replace=False)
                U = U_school[:, idx].copy()
                V = V_school[:, idx].copy()
                W = W_school[:, idx].copy()
            
            U, V, W = als_solve(T, U, V, W, rank, max_iters=200)
            
            U_r = round_half(U)
            V_r = round_half(V)
            W_r = round_half(W)
            
            if verify(U_r, V_r, W_r):
                found = True
                best_rank = rank
                best_U, best_V, best_W = U_r, V_r, W_r
                break
        
        if found:
            break
    
    if best_rank == rank_school:
        best_U, best_V, best_W = U_school, V_school, W_school
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
