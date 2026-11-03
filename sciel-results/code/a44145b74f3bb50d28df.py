import numpy as np
import time

def construct():
    n, m, p = 2, 4, 6
    t0 = time.time()
    deadline = t0 + 140
    
    # Build target tensor (sparse representation)
    # Non-zero entries: T[i*m+j, j*p+k, k*n+i] = 1
    nz_a, nz_b, nz_c = [], [], []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                nz_a.append(i * m + j)
                nz_b.append(j * p + k)
                nz_c.append(k * n + i)
    nz_a = np.array(nz_a)
    nz_b = np.array(nz_b)
    nz_c = np.array(nz_c)
    T_vals = np.ones(len(nz_a))
    
    def compute_error(U, V, W):
        # Compute ||sum_t U[:,t] V[:,t] W[:,t] - T||_F^2
        # Using sparse T
        recon = np.zeros_like(T_vals)
        for t in range(U.shape[1]):
            recon += U[nz_a, t] * V[nz_b, t] * W[nz_c, t]
        return np.sum((recon - T_vals)**2)
    
    def als_step(U, V, W, max_iter=100):
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
            # Update U
            for t in range(U.shape[1]):
                # U[a,t] = sum over (b,c) where T[a,b,c] nonzero: T[a,b,c] V[b,t] W[c,t] / sum V[b,t]^2 W[c,t]^2
                # Group by a
                vw = V[nz_b, t] * W[nz_c, t]
                # For each a, sum of T[a,b,c]*vw over (b,c)
                U_new = np.zeros(n*m)
                for idx in range(len(nz_a)):
                    a = nz_a[idx]
                    U_new[a] += T_vals[idx] * vw[idx]
                denom = np.sum((V[nz_b, t] * W[nz_c, t])**2)
                if denom > 1e-12:
                    U[:,t] = U_new / denom
            
            # Update V
            for t in range(V.shape[1]):
                uw = U[nz_a, t] * W[nz_c, t]
                V_new = np.zeros(m*p)
                for idx in range(len(nz_a)):
                    b = nz_b[idx]
                    V_new[b] += T_vals[idx] * uw[idx]
                denom = np.sum((U[nz_a, t] * W[nz_c, t])**2)
                if denom > 1e-12:
                    V[:,t] = V_new / denom
            
            # Update W
            for t in range(W.shape[1]):
                uv = U[nz_a, t] * V[nz_b, t]
                W_new = np.zeros(p*n)
                for idx in range(len(nz_a)):
                    c = nz_c[idx]
                    W_new[c] += T_vals[idx] * uv[idx]
                denom = np.sum((U[nz_a, t] * V[nz_b, t])**2)
                if denom > 1e-12:
                    W[:,t] = W_new / denom
        
        return U, V, W
    
    best_rank = 48
    best_result = None
    
    # Try ranks from 36 to 41
    for rank in range(36, 42):
        if time.time() > deadline - 30:
            break
        
        for trial in range(3):
            if time.time() > deadline - 10:
                break
            
            U = np.random.randn(n*m, rank) * 0.1
            V = np.random.randn(m*p, rank) * 0.1
            W = np.random.randn(p*n, rank) * 0.1
            
            U, V, W = als_step(U, V, W, max_iter=50)
            
            err = compute_error(U, V, W)
            if err < 1e-8:
                best_rank = rank
                best_result = (U, V, W)
                break
        
        if best_result is not None:
            break
    
    if best_result is not None:
        U, V, W = best_result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: return current rank-42 Strassen solution
    # ... (include the existing code)
