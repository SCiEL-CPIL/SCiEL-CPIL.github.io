import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [2, 2, 4]

def construct():
    n, m, p = DIMENSIONS
    rank_target = 12
    # Shapes: U (4, 12), V (8, 12), W (8, 12)
    # Total variables: 4*12 + 8*12 + 8*12 = 48 + 96 + 96 = 240? No.
    # U: 4*12=48, V: 8*12=96, W: 8*12=96. Total 240.
    # Equations: 4*8*8 = 256.
    
    def residuals(x):
        U = x[:48].reshape(4, 12)
        V = x[48:144].reshape(8, 12)
        W = x[144:240].reshape(8, 12)
        
        # Compute tensor product
        # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        # We want T to match the target tensor.
        # Target tensor has 1 at (i*m+j, j'*p+k', k'*n+i') if i=i', j=j', k=k'.
        # Since we sum over r, we just need to check the 16 positions where target is 1,
        # and the 256-16 positions where target is 0.
        
        res = np.zeros(256)
        idx = 0
        for i in range(n):
            for j in range(m):
                a = i*m + j
                for jp in range(m):
                    for k in range(p):
                        b = jp*p + k
                        for ip in range(n):
                            for kp in range(p):
                                c = kp*n + ip
                                target = 1.0 if (i==ip and j==jp and k==kp) else 0.0
                                val = np.sum(U[a, :] * V[b, :] * W[c, :])
                                res[idx] = val - target
                                idx += 1
        return res

    start_time = time.time()
    best_result = None
    best_err = float('inf')
    
    # Try a few random restarts
    for _ in range(20):
        if time.time() - start_time > 100: break
        x0 = np.random.randn(240) * 0.1
        try:
            sol = least_squares(residuals, x0, method='lm', max_nfev=1000)
            err = sol.cost
            if err < best_err:
                best_err = err
                best_result = sol.x
        except Exception:
            pass
            
    if best_result is not None and best_err < 1e-8:
        x = best_result
        U = x[:48].reshape(4, 12)
        V = x[48:144].reshape(8, 12)
        W = x[144:240].reshape(8, 12)
        
        # Check if we can round to half-integers
        # The problem asks for integer/half-integer entries.
        # If the solution is close to half-integers, we round and verify.
        U_rounded = np.round(U * 2) / 2
        V_rounded = np.round(V * 2) / 2
        W_rounded = np.round(W * 2) / 2
        
        # Verify rounded solution
        max_err = 0
        for i in range(n):
            for j in range(m):
                a = i*m + j
                for jp in range(m):
                    for k in range(p):
                        b = jp*p + k
                        for ip in range(n):
                            for kp in range(p):
                                c = kp*n + ip
                                target = 1.0 if (i==ip and j==jp and k==kp) else 0.0
                                val = np.sum(U_rounded[a, :] * V_rounded[b, :] * W_rounded[c, :])
                                max_err = max(max_err, abs(val - target))
        
        if max_err < 1e-10:
            return {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
            
    # Fallback to rank 14
    rank = 14
    U = [[0.0]*rank for _ in range(4)]
    V = [[0.0]*rank for _ in range(8)]
    W = [[0.0]*rank for _ in range(8)]
    
    def set_col(col, u_dict, v_dict, w_dict):
        for idx, val in u_dict.items():
            U[idx][col] = val
        for idx, val in v_dict.items():
            V[idx][col] = val
        for idx, val in w_dict.items():
            W[idx][col] = val
            
    # A indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # B indices: 0=(0,0), 1=(0,1), 2=(0,2), 3=(0,3), 4=(1,0), 5=(1,1), 6=(1,2), 7=(1,3)
    # W indices (k*n+i): 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1), 4=(2,0), 5=(2,1), 6=(3,0), 7=(3,1)
    
    # Left block (B[:, 0:2])
    set_col(0, {0:1, 3:1}, {0:1, 5:1}, {0:1, 3:1})
    set_col(1, {2:1, 3:1}, {0:1}, {1:1, 3:-1})
    set_col(2, {0:1}, {1:1, 5:-1}, {2:1, 3:1})
    set_col(3, {3:1}, {4:1, 0:-1}, {0:1, 1:1})
    set_col(4, {0:1, 1:1}, {5:1}, {0:-1, 2:1})
    set_col(5, {2:1, 0:-1}, {0:1, 1:1}, {3:1})
    set_col(6, {1:1, 3:-1}, {4:1, 5:1}, {0:1})
    
    # Right block (B[:, 2:4])
    set_col(7, {0:1, 3:1}, {2:1, 7:1}, {4:1, 7:1})
    set_col(8, {2:1, 3:1}, {2:1}, {5:1, 7:-1})
    set_col(9, {0:1}, {3:1, 7:-1}, {6:1, 7:1})
    set_col(10, {3:1}, {6:1, 2:-1}, {4:1, 5:1})
    set_col(11, {0:1, 1:1}, {7:1}, {4:-1, 6:1})
    set_col(12, {2:1, 0:-1}, {2:1, 3:1}, {7:1})
    set_col(13, {1:1, 3:-1}, {6:1, 7:1}, {4:1})
    
    return {"U": U, "V": V, "W": W}
