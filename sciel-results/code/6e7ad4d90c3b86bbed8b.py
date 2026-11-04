import numpy as np
from scipy.optimize import least_squares
import time

DIMENSIONS = [2, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130
    
    def make_schoolbook():
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
    
    best = make_schoolbook()
    
    # Try to find lower rank decompositions
    for r in range(24, n*m*p):
        if time.time() > deadline:
            break
        
        np.random.seed(42 + r)
        U_init = np.random.randn(n*m, r) * 0.3
        V_init = np.random.randn(m*p, r) * 0.3
        W_init = np.random.randn(p*n, r) * 0.3
        params = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        def residuals(params, r):
            U_size = n*m*r
            V_size = m*p*r
            U = params[:U_size].reshape(n*m, r)
            V = params[U_size:U_size+V_size].reshape(m*p, r)
            W = params[U_size+V_size:].reshape(p*n, r)
            
            res = []
            for i in range(n):
                for j in range(m):
                    for jp in range(m):
                        for k in range(p):
                            for kp in range(p):
                                for ip in range(n):
                                    expected = 1.0 if (j==jp and k==kp and i==ip) else 0.0
                                    val = np.sum(U[i*m+j, :] * V[jp*p+k, :] * W[kp*n+ip, :])
                                    res.append(val - expected)
            return np.array(res)
        
        try:
            result = least_squares(residuals, params, args=(r,), method='lm', max_nfev=300)
            if result.cost < 1e-8:
                U_size = n*m*r
                V_size = m*p*r
                U = result.x[:U_size].reshape(n*m, r)
                V = result.x[U_size:U_size+V_size].reshape(m*p, r)
                W = result.x[U_size+V_size:].reshape(p*n, r)
                best = {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
                break
        except Exception:
            pass
    
    return best
