import numpy as np
import time
import sys
from scipy.optimize import least_squares

DIMENSIONS = [2, 2, 6]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110
    
    nonzero = []
    nz_set = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                nonzero.append((a, b, c))
                nz_set.add((a, b, c))
    
    all_indices = [(a, b, c) for a in range(n*m) for b in range(m*p) for c in range(p*n)]
    target_vals = np.array([1.0 if (a,b,c) in nz_set else 0.0 for a,b,c in all_indices])
    
    def get_rank_21():
        rank = 21
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        strassen_factors = [
            ({(0,0): 1, (0,1): 1}, {(0,0): 1, (0,1): 1}, {(0,0): 1, (1,0): 1}),
            ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
            ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
            ({(0,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
            ({(0,0): 1, (1,0): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
            ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
            ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
        ]
        
        r = 0
        for b in range(3):
            for t in range(7):
                u_c, v_c, w_c = strassen_factors[t]
                for (i, j), val in u_c.items():
                    U[i * m + j][r] = val
                for (j, k_local), val in v_c.items():
                    k_global = 2 * b + k_local
                    V[j * p + k_global][r] = val
                for (k_local, i), val in w_c.items():
                    k_global = 2 * b + k_local
                    W[k_global * n + i][r] = val
                r += 1
        return {"U": U, "V": V, "W": W}
    
    def verify(U, V, W, tol=1e-6):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        T_decomp = np.einsum('ar,br,cr->abc', U, V, W)
        max_err = 0.0
        for idx, (a, b, c) in enumerate(all_indices):
            max_err = max(max_err, abs(T_decomp[a, b, c] - target_vals[idx]))
        return max_err < tol, max_err
    
    def try_rank_20():
        rank = 20
        np.random.seed(999)
        best_err = float('inf')
        best_x = None
        
        def residuals(x):
            U = x[:4*rank].reshape(4, rank)
            V = x[4*rank:4*rank+12*rank].reshape(12, rank)
            W = x[4*rank+12*rank:].reshape(12, rank)
            T_decomp = np.einsum('ar,br,cr->abc', U, V, W).flatten()
            return T_decomp - target_vals
        
        base = get_rank_21()
        U21 = np.array(base["U"], dtype=float)
        V21 = np.array(base["V"], dtype=float)
        W21 = np.array(base["W"], dtype=float)
        
        for trial in range(40):
            if time.time() > deadline - 15:
                break
            
            drop = trial % 21
            keep = [i for i in range(21) if i != drop]
            U0 = U21[:, keep].copy()
            V0 = V21[:, keep].copy()
            W0 = W21[:, keep].copy()
            
            scale = 0.05 + 0.01 * trial
            U0 += scale * np.random.randn(*U0.shape)
            V0 += scale * np.random.randn(*V0.shape)
            W0 += scale * np.random.randn(*W0.shape)
            
            x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
            
            try:
                result = least_squares(residuals, x0, method='lm', max_nfev=300)
                err = np.max(np.abs(result.fun))
                if err < best_err:
                    best_err = err
                    best_x = result.x.copy()
                    if err < 1e-8:
                        break
            except Exception:
                continue
        
        if best_x is not None and best_err < 1e-6:
            U = best_x[:4*rank].reshape(4, rank)
            V = best_x[4*rank:4*rank+12*rank].reshape(12, rank)
            W = best_x[4*rank+12*rank:].reshape(12, rank)
            valid, diff = verify(U, V, W)
            if valid:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        return None
    
    result_20 = try_rank_20()
    if result_20 is not None:
        return result_20
    
    return get_rank_21()
