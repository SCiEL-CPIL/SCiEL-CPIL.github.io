import numpy as np
import time

DIMENSIONS = [2, 3, 9]

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    def get_schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    def verify(U, V, W, tol=1e-9):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        r = U.shape[1]
        # Reconstruct tensor and check
        T_rec = np.zeros((n*m, m*p, p*n))
        for c in range(r):
            T_rec += np.einsum('a,b,d->abd', U[:, c], V[:, c], W[:, c])
        # Check all 1-entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a, b, d = i*m+j, j*p+k, k*n+i
                    if abs(T_rec[a, b, d] - 1.0) > tol:
                        return False
        # Check zero entries (sample)
        ones_set = set()
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    ones_set.add((i*m+j, j*p+k, k*n+i))
        np.random.seed(99)
        for _ in range(200):
            a = np.random.randint(0, n*m)
            b = np.random.randint(0, m*p)
            d = np.random.randint(0, p*n)
            if (a, b, d) not in ones_set:
                if abs(T_rec[a, b, d]) > tol:
                    return False
        return True
    
    # Build the target tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Try ranks from lower bound upward
    # Lower bound: max(n*m, m*p, p*n) = max(6, 27, 18) = 27
    # Also, rank >= n*m*p / min(n,m,p) = 54/2 = 27 (slice rank)
    
    best_result = None
    
    for r in range(27, 55):
        if time.time() - start_time > 110:
            break
        
        # Try multiple random initializations
        for seed in range(5):
            if time.time() - start_time > 115:
                break
            
            np.random.seed(seed * 1000 + r)
            U = np.random.randn(n*m, r) * 0.1
            V = np.random.randn(m*p, r) * 0.1
            W = np.random.randn(p*n, r) * 0.1
            
            # ALS iterations
            for it in range(30):
                if time.time() - start_time > 118:
                    break
                
                # Update U
                KR_VW = np.einsum('bc,dc->bdc', V, W).reshape(m*p*p*n, r)
                T_flat = T.reshape(n*m, m*p*p*n)
                try:
                    U_new, _, _, _ = np.linalg.lstsq(KR_VW.T, T_flat.T, rcond=None)
                    U = U_new.T
                except:
                    pass
                
                # Update V
                KR_UW = np.einsum('ac,dc->adc', U, W).reshape(n*m*p*n, r)
                T_flat_v = T.reshape(m*p, n*m*p*n)
                try:
                    V_new, _, _, _ = np.linalg.lstsq(KR_UW.T, T_flat_v.T, rcond=None)
                    V = V_new.T
                except:
                    pass
                
                # Update W
                KR_UV = np.einsum('ab,bc->abc', U, V).reshape(n*m*m*p, r)
                T_flat_w = T.reshape(p*n, n*m*m*p)
                try:
                    W_new, _, _, _ = np.linalg.lstsq(KR_UV.T, T_flat_w.T, rcond=None)
                    W = W_new.T
                except:
                    pass
            
            if verify(U.tolist(), V.tolist(), W.tolist(), tol=1e-8):
                best_result = (r, U.tolist(), V.tolist(), W.tolist())
                break
        
        if best_result is not None:
            break
    
    if best_result is not None:
        r, U, V, W = best_result
        return {"U": U, "V": V, "W": W}
    else:
        return get_schoolbook()
