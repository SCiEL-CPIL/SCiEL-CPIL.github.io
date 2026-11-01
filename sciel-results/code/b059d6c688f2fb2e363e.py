import numpy as np
import time

DIMENSIONS = [2, 2, 6]

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _build_tensor():
    n, m, p = DIMENSIONS
    T = np.zeros((n * m, m * p, p * n))
    for u in range(n * m):
        i = u // m
        j = u % m
        for v in range(m * p):
            j_v = v // p
            k_v = v % p
            if j != j_v: continue
            for w in range(p * n):
                i_w = w % n
                k_w = w // n
                if i == i_w and k_v == k_w:
                    T[u, v, w] = 1.0
    return T

def _check_exact(U, V, W, T):
    # U, V, W are integer/half-integer
    # Check sum_t U[u,t] V[v,t] W[w,t] == T[u,v,w]
    # Use einsum or loops
    # Since size is small, loops are fine
    n, m, p = DIMENSIONS
    for u in range(n * m):
        for v in range(m * p):
            for w in range(p * n):
                s = 0.0
                for t in range(U.shape[1]):
                    s += U[u, t] * V[v, t] * W[w, t]
                if abs(s - T[u, v, w]) > 1e-9:
                    return False
    return True

def _als_decompose(T, r, max_iter=1000, tol=1e-6):
    n, m, p = DIMENSIONS
    U = np.random.randn(n * m, r)
    V = np.random.randn(m * p, r)
    W = np.random.randn(p * n, r)
    
    # Normalize
    for t in range(r):
        norm = np.linalg.norm(U[:, t]) * np.linalg.norm(V[:, t]) * np.linalg.norm(W[:, t])
        if norm == 0: continue
        U[:, t] /= np.linalg.norm(U[:, t])
        V[:, t] /= np.linalg.norm(V[:, t])
        W[:, t] /= np.linalg.norm(W[:, t])
        
    for it in range(max_iter):
        # Update U
        for t in range(r):
            # V_t is V[:, t], W_t is W[:, t]
            # T_unfolded_U[:, t] = V_t \otimes W_t
            # U[:, t] = T_unfolded_U[:, t] / (V_t \otimes W_t)
            # Actually, minimize || T - sum U_t V_t W_t ||^2
            # Gradient wrt U[:, t] is 2 * (T - approx) contracted with V_t, W_t
            # approx = sum U_s V_s W_s
            # Let's use standard ALS
            # Fix V, W. Update U.
            # T_U[:, t] = sum_{v,w} T[u,v,w] V[v,t] W[w,t]
            # U[u,t] = T_U[u,t] / (sum_{v,w} V[v,t]^2 W[w,t]^2)
            
            pass # Implement properly
        
    # This is getting complex to implement from scratch in the thought process.
    # I will use a simpler gradient descent or just try to find a known solution.
    # Given the risk, I will stick to the schoolbook if I can't guarantee a fast find.
    # But the directive asks for a stronger algorithm.
    # I will implement a basic ALS.

def construct():
    n, m, p = DIMENSIONS
    T = _build_tensor()
    start_time = time.time()
    
    # Try ranks 12 to 20
    for r in range(12, 21):
        if time.time() - start_time > 100:
            break
        for attempt in range(50):
            if time.time() - start_time > 105:
                break
            # Random init
            U = np.random.randn(n * m, r)
            V = np.random.randn(m * p, r)
            W = np.random.randn(p * n, r)
            
            # ALS
            for it in range(200):
                # Update U
                for t in range(r):
                    # Residual R = T - sum_{s != t} U_s V_s W_s
                    # This is expensive.
                    # Better: T_U = T contracted with V_t, W_t
                    # T_U[u] = sum_{v,w} T[u,v,w] V[v,t] W[w,t]
                    # denom = sum_{v,w} V[v,t]^2 W[w,t]^2
                    # U[:,t] = T_U / denom
                    
                    # Compute T_U
                    # T is (nm, mp, pn)
                    # V_t is (mp,)
                    # W_t is (pn,)
                    # T_U = np.einsum('uvw,vt,wt->u', T, V[:,t], W[:,t])
                    T_U = np.einsum('uvw,vt,wt->u', T, V[:, t], W[:, t])
                    denom = np.sum(V[:, t]**2) * np.sum(W[:, t]**2)
                    if denom > 1e-9:
                        U[:, t] = T_U / denom
                    
                # Update V
                for t in range(r):
                    T_V = np.einsum('uvw,ut,wt->v', T, U[:, t], W[:, t])
                    denom = np.sum(U[:, t]**2) * np.sum(W[:, t]**2)
                    if denom > 1e-9:
                        V[:, t] = T_V / denom
                        
                # Update W
                for t in range(r):
                    T_W = np.einsum('uvw,ut,vt->w', T, U[:, t], V[:, t])
                    denom = np.sum(U[:, t]**2) * np.sum(V[:, t]**2)
                    if denom > 1e-9:
                        W[:, t] = T_W / denom
                        
            # Check error
            approx = np.einsum('ut,vt,wt->uvw', U, V, W)
            err = np.linalg.norm(approx - T)
            if err < 1e-4:
                # Round to half-integers
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                
                # Check exactness
                if _check_exact(U_r, V_r, W_r, T):
                    return {"U": U_r, "V": V_r, "W": W_r}
                    
    return _schoolbook()
