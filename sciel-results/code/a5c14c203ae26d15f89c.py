"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import least_squares

def construct():
    n, m, p = DIMENSIONS
    rank_school = n * m * p
    
    # Schoolbook fallback
    U_fb = [[0] * rank_school for _ in range(n * m)]
    V_fb = [[0] * rank_school for _ in range(m * p)]
    W_fb = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][col] = 1
                V_fb[j * p + k][col] = 1
                W_fb[k * n + i][col] = 1
                col += 1
    
    deadline = time.time() + 140
    best = None
    
    # Try decreasing ranks
    for r in range(16, rank_school + 1):
        if time.time() > deadline - 5:
            break
        found = False
        for attempt in range(40):
            if time.time() > deadline - 3:
                break
            seed = attempt * 1000003 + r * 7919
            np.random.seed(seed)
            U0 = np.random.randn(n*m, r) * 0.2
            V0 = np.random.randn(m*p, r) * 0.2
            W0 = np.random.randn(p*n, r) * 0.2
            
            def res_fn(x):
                U = x[:n*m*r].reshape(n*m, r)
                V = x[n*m*r:n*m*r+m*p*r].reshape(m*p, r)
                W = x[n*m*r+m*p*r:].reshape(p*n, r)
                out = []
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            out.append(np.dot(U[i*m+j], V[j*p+k] * W[k*n+i]) - 1.0)
                return out
            
            try:
                sol = least_squares(res_fn, np.concatenate([U0.ravel(), V0.ravel(), W0.ravel()]),
                                    max_nfev=3000, xtol=1e-14, ftol=1e-14)
                if np.max(np.abs(sol.fun)) < 1e-9:
                    U_c = sol.x[:n*m*r].reshape(n*m, r)
                    V_c = sol.x[n*m*r:n*m*r+m*p*r].reshape(m*p, r)
                    W_c = sol.x[n*m*r+m*p*r:].reshape(p*n, r)
                    
                    # Try rounding to half-integers
                    for U_r in [np.round(U_c*2)/2, np.round(U_c*4)/4*0.5]:
                        for V_r in [np.round(V_c*2)/2, np.round(V_c*4)/4*0.5]:
                            for W_r in [np.round(W_c*2)/2, np.round(W_c*4)/4*0.5]:
                                ok = True
                                for i in range(n):
                                    for j in range(m):
                                        for k in range(p):
                                            v = np.dot(U_r[i*m+j], V_r[j*p+k] * W_r[k*n+i])
                                            if abs(v - 1.0) > 1e-9:
                                                ok = False
                                                break
                                        if not ok: break
                                    if not ok: break
                                if ok:
                                    best = (U_r, V_r, W_r, r)
                                    found = True
                                    break
                            if found: break
                        if found: break
                    if found: break
            except Exception:
                pass
        if found:
            break
    
    if best is not None:
        return {"U": best[0].tolist(), "V": best[1].tolist(), "W": best[2].tolist()}
    return {"U": U_fb, "V": V_fb, "W": W_fb}
# EVOLVE-BLOCK-END
