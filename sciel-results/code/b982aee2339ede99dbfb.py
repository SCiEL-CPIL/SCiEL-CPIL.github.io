import numpy as np
import time

def construct():
    n, m, p = 3, 3, 4
    start = time.time()
    deadline = start + 140
    
    nm = n * m  # 9
    mp = m * p  # 12
    pn = p * n  # 12
    
    # Build target tensor M
    M = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                M[a, b, c] = 1.0
    
    def residual(U, V, W):
        R = U.shape[1]
        M_hat = np.zeros((nm, mp, pn), dtype=float)
        for r in range(R):
            M_hat += np.outer(np.outer(U[:,r], V[:,r]), W[:,r])
        return M_hat - M
    
    # Schoolbook solution
    R_school = n * m * p
    U_s = np.zeros((nm, R_school))
    V_s = np.zeros((mp, R_school))
    W_s = np.zeros((pn, R_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_s[i*m+j, col] = 1.0
                V_s[j*p+k, col] = 1.0
                W_s[k*n+i, col] = 1.0
                col += 1
    
    best_rank = R_school
    best_solution = (U_s.copy(), V_s.copy(), W_s.copy())
    
    # Try to find lower rank solutions using ALS
    for R in range(32, 27, -1):
        if time.time() > deadline - 20:
            break
        
        found = False
        n_attempts = 12
        
        for attempt in range(n_attempts):
            if time.time() > deadline - 20:
                break
            
            seed = attempt * 7919 + R * 13 + 42
            np.random.seed(seed)
            U = np.random.randn(nm, R)
            V = np.random.randn(mp, R)
            W = np.random.randn(pn, R)
            
            # ALS iterations
            for it in range(60):
                if time.time() > deadline - 20:
                    break
                
                M_flat = M.reshape(nm*mp, pn)
                
                # Solve for W
                X = np.zeros((nm*mp, R))
                for a in range(nm):
                    for b in range(mp):
                        X[a*mp+b, :] = U[a, :] * V[b, :]
                W = (np.linalg.pinv(X) @ M_flat).T
                
                # Solve for U
                Z = np.zeros((R, mp*pn))
                for r in range(R):
                    for b in range(mp):
                        for c in range(pn):
                            Z[r, b*pn+c] = V[b,r] * W[c,r]
                U = np.zeros((nm, R))
                for a in range(nm):
                    M_a = M_flat[a*mp:(a+1)*mp, :].flatten()
                    U[a, :] = np.linalg.pinv(Z) @ M_a
                
                # Solve for V
                P = np.zeros((R, nm*pn))
                for r in range(R):
                    for a in range(nm):
                        for c in range(pn):
                            P[r, a*pn+c] = U[a,r] * W[c,r]
                V = np.zeros((mp, R))
                for b in range(mp):
                    M_b = M_flat[:, b, :].flatten()
                    V[b, :] = np.linalg.pinv(P) @ M_b
            
            res = residual(U, V, W)
            res_norm = np.linalg.norm(res)
            if res_norm < 1e-4:
                U_r = np.round(2*U) / 2
                V_r = np.round(2*V) / 2
                W_r = np.round(2*W) / 2
                res_r = residual(U_r, V_r, W_r)
                if np.linalg.norm(res_r) < 1e-9:
                    best_rank = R
                    best_solution = (U_r, V_r, W_r)
                    found = True
                    break
        
        if found:
            break
    
    return {"U": best_solution[0].tolist(), "V": best_solution[1].tolist(), "W": best_solution[2].tolist()}
