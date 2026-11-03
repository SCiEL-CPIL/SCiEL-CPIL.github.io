import time
import numpy as np

DIMENSIONS = [5, 5, 5]

def construct():
    start_time = time.time()
    deadline = 145.0  # 5s safety margin
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 125
    
    def verify_fast(U, V, W):
        """Fast vectorized verification using einsum."""
        U_mat = U.reshape(n, m, -1)
        V_mat = V.reshape(m, p, -1)
        W_mat = W.reshape(p, n, -1)
        T = np.einsum('ijc,jkc,kic->ijk', U_mat, V_mat, W_mat)
        max_err = np.max(np.abs(T - 1.0))
        return float(max_err)
    
    def vec_update_U(U, V, W, rank):
        """Vectorized update for U."""
        for i in range(n):
            for j in range(m):
                # B[c, k] = V[j*p+k, c] * W[k*n+i, c]
                B = V[j*p:(j+1)*p, :].T * W[:, None, :]  # wrong, need per-k
                # Actually: B[c,k] = V[j*p+k, c] * W[k*n+i, c]
                B = np.zeros((rank, p))
                for k in range(p):
                    B[:, k] = V[j*p+k, :] * W[k*n+i, :]
                try:
                    U[i*m+j, :] = np.linalg.lstsq(B.T, np.ones(p), rcond=None)[0]
                except:
                    pass
    
    def vec_update_V(U, V, W, rank):
        """Vectorized update for V."""
        for j in range(m):
            for k in range(p):
                A = np.zeros((rank, n))
                for i in range(n):
                    A[:, i] = U[i*m+j, :] * W[k*n+i, :]
                try:
                    V[j*p+k, :] = np.linalg.lstsq(A.T, np.ones(n), rcond=None)[0]
                except:
                    pass
    
    def vec_update_W(U, V, W, rank):
        """Vectorized update for W."""
        for k in range(p):
            for i in range(n):
                C = np.zeros((rank, m))
                for j in range(m):
                    C[:, j] = U[i*m+j, :] * V[j*p+k, :]
                try:
                    W[k*n+i, :] = np.linalg.lstsq(C.T, np.ones(m), rcond=None)[0]
                except:
                    pass
    
    def try_decomposition(rank, max_iters=30, seed=0, init_U=None, init_V=None, init_W=None, tol=1e-7):
        """Alternating minimization for rank decomposition."""
        if time.time() > deadline - 10:
            return None
        
        np.random.seed(seed)
        
        if init_U is not None and init_V is not None and init_W is not None:
            U = init_U.copy()
            V = init_V.copy()
            W = init_W.copy()
        else:
            U = np.random.randn(n*m, rank) * 0.3
            V = np.random.randn(m*p, rank) * 0.3
            W = np.random.randn(p*n, rank) * 0.3
        
        prev_err = float('inf')
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                return None
            
            vec_update_U(U, V, W, rank)
            vec_update_V(U, V, W, rank)
            vec_update_W(U, V, W, rank)
            
            max_err = verify_fast(U, V, W)
            
            if max_err < tol:
                return U, V, W, max_err
            
            if max_err > prev_err * 1.2 and it > 5:
                break
            prev_err = max_err
        
        return None
    
    def polish_decomposition(U, V, W, rank, max_iters=50):
        """High-precision polish pass."""
        if time.time() > deadline - 5:
            return None
        
        prev_err = float('inf')
        
        for it in range(max_iters):
            if time.time() > deadline - 3:
                break
            
            vec_update_U(U, V, W, rank)
            vec_update_V(U, V, W, rank)
            vec_update_W(U, V, W, rank)
            
            max_err = verify_fast(U, V, W)
            
            if max_err < 1e-12:
                break
            
            if max_err > prev_err * 1.02 and it > 20:
                break
            prev_err = max_err
        
        return U, V, W, verify_fast(U, V, W)
    
    def rational_reconstruct(U, V, W, max_denom=100):
        """Try to find exact rational/half-integer values."""
        # Round to nearest half-integer first
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        err = verify_fast(U_r, V_r, W_r)
        if err < 1e-10:
            return U_r, V_r, W_r, err
        
        # Try integer rounding
        U_i = np.round(U)
        V_i = np.round(V)
        W_i = np.round(W)
        err = verify_fast(U_i, V_i, W_i)
        if err < 1e-10:
            return U_i, V_i, W_i, err
        
        # Try with denominator 2
        for denom in [2, 4, 8, 16]:
            U_d = np.round(U * denom) / denom
            V_d = np.round(V * denom) / denom
            W_d = np.round(W * denom) / denom
            err = verify_fast(U_d, V_d, W_d)
            if err < 1e-10:
                return U_d, V_d, W_d, err
        
        return U_r, V_r, W_r, err
    
    # Schoolbook fallback
    U_school = np.zeros((n*m, rank_schoolbook))
    V_school = np.zeros((m*p, rank_schoolbook))
    W_school = np.zeros((p*n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    
    best_rank = rank_schoolbook
    best_result = (U_school, V_school, W_school)
    
    # Try ranks from known lower bounds upward
    # Known: rank >= 105 for 5x5x5 (Landsberg bound is ~105)
    # Try from 105 up to 124
    candidate_ranks = list(range(105, min(125, rank_schoolbook)))
    
    for rank in candidate_ranks:
        if time.time() > deadline - 15:
            break
        
        found_valid = False
        
        # Try multiple seeds
        for seed in range(5):
            if time.time() > deadline - 10:
                break
            
            result = try_decomposition(rank, max_iters=25, seed=seed + rank*100, tol=1e-6)
            if result is not None:
                U, V, W, err = result
                U_r, V_r, W_r, err_r = rational_reconstruct(U, V, W)
                if err_r < 1e-8:
                    best_rank = rank
                    best_result = (U_r, V_r, W_r)
                    found_valid = True
                    break
        
        if found_valid and best_rank <= 108:
            break
    
    # Polish if we found something
    if best_rank < rank_schoolbook:
        U, V, W = best_result
        polished = polish_decomposition(U.copy(), V.copy(), W.copy(), best_rank)
        if polished is not None:
            U_p, V_p, W_p, err_p = polished
            U_pr, V_pr, W_pr, err_pr = rational_reconstruct(U_p, V_p, W_p)
            if err_pr < 1e-8:
                best_result = (U_pr, V_pr, W_pr)
    
    U, V, W = best_result
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
