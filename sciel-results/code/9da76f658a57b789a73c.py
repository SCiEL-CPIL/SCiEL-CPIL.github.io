import numpy as np
import time
import itertools

DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    # Build target tensor T[i*m+j, j*p+k, k*n+i] = 1
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
    best_U, best_V, best_W = U_school.copy(), V_school.copy(), W_school.copy()
    
    def verify(U, V, W):
        T_approx = np.einsum('ic,jc,kc->ijk', U, V, W)
        return np.allclose(T_approx, T, atol=1e-10)
    
    def round_half(X):
        return np.round(X * 2) / 2
    
    def als_solve(T, U, V, W, rank, max_iters=300):
        nr, nc, nk = T.shape
        prev_err = np.inf
        for it in range(max_iters):
            if time.time() > deadline - 15:
                break
            
            # Update U: for each i, T[i,:,:].flatten() = U[i,:] @ (V[:,c]*W[:,c]).T
            M_VW = np.einsum('jc,kc->ckj', V, W).reshape(rank, nc * nk)
            T_i = T.reshape(nr, nc * nk)
            try:
                U = T_i @ np.linalg.pinv(M_VW.T)
            except:
                break
            
            # Update V: for each j, T[:,j,:].flatten() = V[j,:] @ (U[:,c]*W[:,c]).T
            M_UW = np.einsum('ic,kc->cik', U, W).reshape(rank, nr * nk)
            T_j = np.zeros((nc, nr * nk))
            for j in range(nc):
                T_j[j, :] = T[:, j, :].flatten()
            try:
                V = T_j @ np.linalg.pinv(M_UW.T)
            except:
                break
            
            # Update W: for each k, T[:,:,k].flatten() = W[k,:] @ (U[:,c]*V[:,c]).T
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
    
    def polish(U, V, W, max_iters=100):
        """High-precision polish: minimize ||sum_c U[:,c]V[:,c]W[:,c] - T||_F^2
        with tighter tolerance and more iterations."""
        nr, nc, nk = T.shape
        rank = U.shape[1]
        prev_err = np.inf
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
            
            M_VW = np.einsum('jc,kc->ckj', V, W).reshape(rank, nc * nk)
            T_i = T.reshape(nr, nc * nk)
            try:
                U_new = T_i @ np.linalg.pinv(M_VW.T)
            except:
                break
            
            M_UW = np.einsum('ic,kc->cik', U_new, W).reshape(rank, nr * nk)
            T_j = np.zeros((nc, nr * nk))
            for j in range(nc):
                T_j[j, :] = T[:, j, :].flatten()
            try:
                V_new = T_j @ np.linalg.pinv(M_UW.T)
            except:
                break
            
            M_UV = np.einsum('ic,jc->cij', U_new, V_new).reshape(rank, nr * nc)
            T_k = np.zeros((nk, nr * nc))
            for k in range(nk):
                T_k[k, :] = T[:, :, k].flatten()
            try:
                W_new = T_k @ np.linalg.pinv(M_UV.T)
            except:
                break
            
            T_approx = np.einsum('ic,jc,kc->ijk', U_new, V_new, W_new)
            err = np.linalg.norm(T_approx - T)
            if abs(err - prev_err) < 1e-16:
                U, V, W = U_new, V_new, W_new
                break
            prev_err = err
            U, V, W = U_new, V_new, W_new
        
        return U, V, W
    
    def try_rank(rank, n_attempts=15, use_polish=True):
        """Try to find a valid decomposition at given rank."""
        nonlocal best_rank, best_U, best_V, best_W
        
        for attempt in range(n_attempts):
            if time.time() > deadline - 20:
                break
            
            # Different initialization strategies
            if attempt < 4:
                # Random Gaussian with various scales
                np.random.seed(attempt * 7919 + rank * 131)
                scale = 0.2 + 0.1 * (attempt % 3)
                U = np.random.randn(n*m, rank) * scale
                V = np.random.randn(m*p, rank) * scale
                W = np.random.randn(p*n, rank) * scale
            elif attempt < 8:
                # Sparse random with +/-1
                np.random.seed(attempt * 104729 + rank)
                U = np.zeros((n*m, rank))
                V = np.zeros((m*p, rank))
                W = np.zeros((p*n, rank))
                for c in range(rank):
                    U[np.random.randint(n*m), c] = np.random.choice([-1.0, 1.0])
                    V[np.random.randint(m*p), c] = np.random.choice([-1.0, 1.0])
                    W[np.random.randint(p*n), c] = np.random.choice([-1.0, 1.0])
            elif attempt < 12:
                # Based on schoolbook subset
                np.random.seed(attempt * 15485863 + rank)
                idx = np.random.choice(rank_school, min(rank, rank_school), replace=False)
                U = U_school[:, idx].copy()
                V = V_school[:, idx].copy()
                W = W_school[:, idx].copy()
            else:
                # Structured: exploit the block structure
                # A is 3x4, B is 4x8, C is 3x8
                # Split B into two 4x4 blocks: C = A*B1 + A*B2
                # Try to share some terms between the two blocks
                np.random.seed(attempt * 31337 + rank)
                U = np.random.randn(n*m, rank) * 0.3
                V = np.zeros((m*p, rank))
                W = np.zeros((p*n, rank))
                # First half of rank for B1, second half for B2
                half = rank // 2
                for c in range(half):
                    U[np.random.randint(n*m), c] = np.random.choice([-1.0, 1.0])
                    V[np.random.randint(0, m*4), c] = np.random.choice([-1.0, 1.0])
                    W[np.random.randint(0, p*n//2), c] = np.random.choice([-1.0, 1.0])
                for c in range(half, rank):
                    U[np.random.randint(n*m), c] = np.random.choice([-1.0, 1.0])
                    V[np.random.randint(m*4, m*p), c] = np.random.choice([-1.0, 1.0])
                    W[np.random.randint(p*n//2, p*n), c] = np.random.choice([-1.0, 1.0])
            
            U, V, W = als_solve(T, U, V, W, rank, max_iters=200)
            
            if use_polish:
                U, V, W = polish(U, V, W, max_iters=100)
            
            U_r = round_half(U)
            V_r = round_half(V)
            W_r = round_half(W)
            
            if verify(U_r, V_r, W_r):
                best_rank = rank
                best_U, best_V, best_W = U_r, V_r, W_r
                return True
        
        return False
    
    # Try ranks in decreasing order of likelihood
    # M(3,4,8) schoolbook = 96
    # Known bounds suggest M(3,4,8) <= 2*M(3,4,4) and M(3,4,4) <= 36
    # So M(3,4,8) <= 72, but likely much lower
    # Try a range of ranks
    ranks_to_try = [64, 56, 48, 42, 36, 32, 28, 24]
    
    for rank in ranks_to_try:
        if time.time() > deadline - 30:
            break
        if rank >= best_rank:
            continue
        
        found = try_rank(rank, n_attempts=12, use_polish=True)
        if found:
            # Try to go lower
            continue
    
    if best_rank == rank_school:
        best_U, best_V, best_W = U_school.copy(), V_school.copy(), W_school.copy()
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
