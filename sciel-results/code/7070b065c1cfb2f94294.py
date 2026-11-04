import numpy as np
import time

def construct():
    start_time = time.time()
    timeout = 110
    n, m, p = 2, 3, 3
    
    def elapsed():
        return time.time() - start_time
    
    def build_tensor():
        T = np.zeros((n * m, m * p, p * n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T[a, b, c] = 1.0
        return T
    
    def verify(U, V, W, T, tol=1e-9):
        Re = T - np.einsum('ar,br,cr->abc', U, V, W)
        return np.max(np.abs(Re)) < tol
    
    def build_rank17():
        rank = 17
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        def set_U(i, j, r, val): U[i * m + j, r] = val
        def set_V(j, k, r, val): V[j * p + k, r] = val
        def set_W(k, i, r, val): W[k * n + i, r] = val
        
        # Strassen 2x2x2
        set_U(0, 0, 0, 1); set_U(1, 1, 0, 1)
        set_V(0, 0, 0, 1); set_V(1, 1, 0, 1)
        set_W(0, 0, 0, 1); set_W(1, 1, 0, 1)
        
        set_U(1, 0, 1, 1); set_U(1, 1, 1, 1)
        set_V(0, 0, 1, 1)
        set_W(0, 1, 1, 1); set_W(1, 1, 1, -1)
        
        set_U(0, 0, 2, 1)
        set_V(0, 1, 2, 1); set_V(1, 1, 2, -1)
        set_W(1, 0, 2, 1); set_W(1, 1, 2, 1)
        
        set_U(1, 1, 3, 1)
        set_V(1, 0, 3, 1); set_V(0, 0, 3, -1)
        set_W(0, 0, 3, 1); set_W(0, 1, 3, 1)
        
        set_U(0, 0, 4, 1); set_U(0, 1, 4, 1)
        set_V(1, 1, 4, 1)
        set_W(0, 0, 4, -1); set_W(1, 0, 4, 1)
        
        set_U(1, 0, 5, 1); set_U(0, 0, 5, -1)
        set_V(0, 0, 5, 1); set_V(0, 1, 5, 1)
        set_W(1, 1, 5, 1)
        
        set_U(0, 1, 6, 1); set_U(1, 1, 6, -1)
        set_V(1, 0, 6, 1); set_V(1, 1, 6, 1)
        set_W(0, 0, 6, 1)
        
        # Cross terms: A[0,0]*B[0,2] -> C[0,2]
        set_U(0, 0, 7, 1); set_V(0, 2, 7, 1); set_W(2, 0, 7, 1)
        # A[0,1]*B[1,2] -> C[0,2]
        set_U(0, 1, 8, 1); set_V(1, 2, 8, 1); set_W(2, 0, 8, 1)
        # A[1,0]*B[0,2] -> C[1,2]
        set_U(1, 0, 9, 1); set_V(0, 2, 9, 1); set_W(2, 1, 9, 1)
        # A[1,1]*B[1,2] -> C[1,2]
        set_U(1, 1, 10, 1); set_V(1, 2, 10, 1); set_W(2, 1, 10, 1)
        # A[0,2]*B[2,0] -> C[0,0]
        set_U(0, 2, 11, 1); set_V(2, 0, 11, 1); set_W(0, 0, 11, 1)
        # A[0,2]*B[2,1] -> C[0,1]
        set_U(0, 2, 12, 1); set_V(2, 1, 12, 1); set_W(1, 0, 12, 1)
        # A[0,2]*B[2,2] -> C[0,2]
        set_U(0, 2, 13, 1); set_V(2, 2, 13, 1); set_W(2, 0, 13, 1)
        # A[1,2]*B[2,0] -> C[1,0]
        set_U(1, 2, 14, 1); set_V(2, 0, 14, 1); set_W(0, 1, 14, 1)
        # A[1,2]*B[2,1] -> C[1,1]
        set_U(1, 2, 15, 1); set_V(2, 1, 15, 1); set_W(1, 1, 15, 1)
        # A[1,2]*B[2,2] -> C[1,2]
        set_U(1, 2, 16, 1); set_V(2, 2, 16, 1); set_W(2, 1, 16, 1)
        
        return U, V, W
    
    def polish(U, V, W, T, max_iters=200, tol=1e-12):
        R = U.shape[1]
        for it in range(max_iters):
            if elapsed() > timeout:
                break
            # Update U
            for r in range(R):
                v_r = V[:, r]
                w_r = W[:, r]
                t_u = np.einsum('abc,b,c->a', T, v_r, w_r)
                norm_sq = np.dot(v_r, v_r) * np.dot(w_r, w_r)
                if norm_sq > 1e-12:
                    U[:, r] = t_u / norm_sq
            # Update V
            for r in range(R):
                u_r = U[:, r]
                w_r = W[:, r]
                t_v = np.einsum('abc,a,c->b', T, u_r, w_r)
                norm_sq = np.dot(u_r, u_r) * np.dot(w_r, w_r)
                if norm_sq > 1e-12:
                    V[:, r] = t_v / norm_sq
            # Update W
            for r in range(R):
                u_r = U[:, r]
                v_r = V[:, r]
                t_w = np.einsum('abc,a,b->c', T, u_r, v_r)
                norm_sq = np.dot(u_r, u_r) * np.dot(v_r, v_r)
                if norm_sq > 1e-12:
                    W[:, r] = t_w / norm_sq
            if verify(U, V, W, T, tol):
                break
        return U, V, W
    
    def search_rank(R, T, n_restarts=3, max_iters=300):
        for rs in range(n_restarts):
            if elapsed() > timeout - 5:
                break
            np.random.seed(42 + rs * 1000)
            U = np.random.randn(n * m, R)
            V = np.random.randn(m * p, R)
            W = np.random.randn(p * n, R)
            U /= np.linalg.norm(U)
            V /= np.linalg.norm(V)
            W /= np.linalg.norm(W)
            
            prev_err = np.inf
            for it in range(max_iters):
                if elapsed() > timeout - 5:
                    break
                # ALS updates
                for r in range(R):
                    v_r = V[:, r]; w_r = W[:, r]
                    t_u = np.einsum('abc,b,c->a', T, v_r, w_r)
                    ns = np.dot(v_r, v_r) * np.dot(w_r, w_r)
                    if ns > 1e-12:
                        U[:, r] = t_u / ns
                for r in range(R):
                    u_r = U[:, r]; w_r = W[:, r]
                    t_v = np.einsum('abc,a,c->b', T, u_r, w_r)
                    ns = np.dot(u_r, u_r) * np.dot(w_r, w_r)
                    if ns > 1e-12:
                        V[:, r] = t_v / ns
                for r in range(R):
                    u_r = U[:, r]; v_r = V[:, r]
                    t_w = np.einsum('abc,a,b->c', T, u_r, v_r)
                    ns = np.dot(u_r, u_r) * np.dot(v_r, v_r)
                    if ns > 1e-12:
                        W[:, r] = t_w / ns
                
                err = np.linalg.norm(T - np.einsum('ar,br,cr->abc', U, V, W))
                if err < 1e-10:
                    U, V, W = polish(U, V, W, T, max_iters=50)
                    if verify(U, V, W, T):
                        return U, V, W
                    return None
                if err > prev_err * 1.05:
                    break
                prev_err = err
        return None
    
    T = build_tensor()
    
    # Try rank 16
    res16 = search_rank(16, T, n_restarts=5, max_iters=200)
    if res16 is not None:
        U, V, W = res16
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try rank 15
    if elapsed() < timeout - 20:
        res15 = search_rank(15, T, n_restarts=3, max_iters=150)
        if res15 is not None:
            U, V, W = res15
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: rank 17
    U, V, W = build_rank17()
    U, V, W = polish(U, V, W, T, max_iters=30)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
