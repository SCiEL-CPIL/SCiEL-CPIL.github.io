"""Matrix multiplication 3x3x5 rank-minimal decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill

    n, m, p = DIMENSIONS  # 3, 3, 5

    # First, build the rank-45 schoolbook solution as fallback
    rank_fallback = n * m * p  # 45
    U_fb = np.zeros((n * m, rank_fallback))
    V_fb = np.zeros((m * p, rank_fallback))
    W_fb = np.zeros((p * n, rank_fallback))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j, col] = 1.0
                V_fb[j * p + k, col] = 1.0
                W_fb[k * n + i, col] = 1.0
                col += 1
    
    # Vectorized verification
    def verify(U, V, W, n, m, p):
        try:
            U_t = U.T.reshape(n, m, -1)
            V_t = V.T.reshape(m, p, -1)
            W_t = W.T.reshape(p, n, -1)
            res = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
            return np.allclose(res, 1.0, atol=1e-9)
        except Exception:
            return False

    if not verify(U_fb, V_fb, W_fb, n, m, p):
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    def trim(U, V, W):
        active = np.ones(U.shape[1], dtype=bool)
        u_norms = np.linalg.norm(U, axis=0)
        active &= (u_norms > 1e-12)
        v_norms = np.linalg.norm(V, axis=0)
        active &= (v_norms > 1e-12)
        w_norms = np.linalg.norm(W, axis=0)
        active &= (w_norms > 1e-12)
        
        if not np.all(active):
            U = U[:, active]
            V = V[:, active]
            W = W[:, active]
        return U, V, W

    best_rank = rank_fallback
    best_U, best_V, best_W = U_fb, V_fb, W_fb
    
    def solve_rank(target_rank, max_steps=1000, lr=0.05, seed=0):
        np.random.seed(seed)
        U_try = np.random.randn(n * m, target_rank) * 0.1
        V_try = np.random.randn(m * p, target_rank) * 0.1
        W_try = np.random.randn(p * n, target_rank) * 0.1
        
        for step in range(max_steps):
            if time.time() > deadline - 10:
                break
                
            U_t = U_try.T.reshape(n, m, target_rank)
            V_t = V_try.T.reshape(m, p, target_rank)
            W_t = W_try.T.reshape(p, n, target_rank)
            
            res_sum = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
            R = res_sum - 1.0
            
            if np.max(np.abs(R)) < 1e-9:
                return U_try, V_try, W_try
            
            grad_U_t = 2 * np.einsum('ijk,jkr,kir->ijr', R, V_t, W_t)
            grad_V_t = 2 * np.einsum('ijk,ijr,kir->jkr', R, U_t, W_t)
            grad_W_t = 2 * np.einsum('ijk,ijr,jkr->kir', R, U_t, V_t)
            
            grad_U = grad_U_t.T.reshape(n * m, target_rank)
            grad_V = grad_V_t.T.reshape(m * p, target_rank)
            grad_W = grad_W_t.T.reshape(p * n, target_rank)
            
            U_try -= lr * grad_U
            V_try -= lr * grad_V
            W_try -= lr * grad_W
            
            norm_U = np.linalg.norm(U_try)
            norm_V = np.linalg.norm(V_try)
            norm_W = np.linalg.norm(W_try)
            max_norm = max(norm_U, norm_V, norm_W)
            if max_norm > 100:
                scale = 100 / max_norm
                U_try *= scale
                V_try *= scale
                W_try *= scale
                lr *= 0.5
                
            if step % 50 == 0:
                lr *= 0.95

        return U_try, V_try, W_try

    for target_rank in range(44, 29, -1):
        if time.time() > deadline - 20:
            break
        
        for attempt in range(10):
            if time.time() > deadline - 20:
                break
            
            U_try, V_try, W_try = solve_rank(target_rank, max_steps=500, lr=0.05, seed=attempt)
            
            if verify(U_try, V_try, W_try, n, m, p):
                U_try, V_try, W_try = trim(U_try, V_try, W_try)
                new_rank = U_try.shape[1]
                if new_rank < best_rank:
                    best_rank = new_rank
                    best_U, best_V, best_W = U_try, V_try, W_try
                break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
