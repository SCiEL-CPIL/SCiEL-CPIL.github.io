import numpy as np
import time
import scipy.optimize as opt

def construct():
    n, m, p = 3, 3, 3
    
    # Target Tensor T
    T = np.zeros((9, 9, 9), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0
                
    def residual(X):
        C = np.einsum('ur,vr,wr->uvw', X, X, X)
        return (C - T).flatten()
        
    def verify(X):
        # Check integers
        X_int = np.round(X)
        C_int = np.einsum('ur,vr,wr->uvw', X_int, X_int, X_int)
        if np.allclose(C_int, T, atol=1e-9):
            return X_int.astype(int)
            
        # Check half-integers
        X_half = np.round(X * 2) / 2.0
        C_half = np.einsum('ur,vr,wr->uvw', X_half, X_half, X_half)
        if np.allclose(C_half, T, atol=1e-9):
            return X_half
            
        return None

    def get_schoolbook():
        rank = 27
        U = np.zeros((9, rank), dtype=int)
        V = np.zeros((9, rank), dtype=int)
        W = np.zeros((9, rank), dtype=int)
        col = 0
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    U[i * 3 + j][col] = 1
                    V[j * 3 + k][col] = 1
                    W[k * 3 + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    start_time = time.time()
    deadline = start_time + 110
    
    # Try ranks 23, 24, 25, 26
    for rank in [23, 24, 25, 26]:
        if time.time() > deadline:
            break
            
        # Try a few initializations
        for attempt in range(15):
            if time.time() > deadline:
                break
                
            # Random initialization
            X0 = np.random.randint(-2, 3, size=(9, rank)).astype(float)
            
            try:
                result = opt.least_squares(residual, X0, method='lm', max_nfev=100)
                
                # Check if residual is small enough to potentially round
                res_norm = np.linalg.norm(result.fun)
                if res_norm < 1.0:
                    sol = verify(result.x)
                    if sol is not None:
                        return {"U": sol, "V": sol, "W": sol}
                        
            except Exception:
                pass

    return get_schoolbook()
