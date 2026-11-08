import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # Build tensor T[a,b,c]
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a, b, c] = 1.0
    
    deadline = time.time() + 100
    best_result = None
    
    for rank in range(15, 31):
        if time.time() > deadline:
            break
        
        found = False
        for attempt in range(300):
            if time.time() > deadline:
                break
            
            # Random initialization
            U = np.random.randn(nm, rank)
            V = np.random.randn(mp, rank)
            W = np.random.randn(pn, rank)
            
            # Alternating least squares
            for it in range(300):
                # Solve for U given V, W
                VW = V * W
                for a in range(nm):
                    rhs = T[a].reshape(-1)
                    U[a, :], _, _, _ = np.linalg.lstsq(VW.T, rhs, rcond=None)
                
                # Solve for V given U, W
                UW = U * W
                for b in range(mp):
                    rhs = T[:, b].reshape(-1)
                    V[b, :], _, _, _ = np.linalg.lstsq(UW.T, rhs, rcond=None)
                
                # Solve for W given U, V
                UV = U * V
                for c in range(pn):
                    rhs = T[:, :, c].reshape(-1)
                    W[c, :], _, _, _ = np.linalg.lstsq(UV.T, rhs, rcond=None)
            
            # Check accuracy
            R = np.einsum('ar,br,cr->abc', U, V, W)
            err = np.max(np.abs(R - T))
            
            if err < 1e-6:
                # Try rounding to half-integers
                for scale_r in [2, 4, 8, 16, 32]:
                    U_r = np.round(U * scale_r) / scale_r
                    V_r = np.round(V * scale_r) / scale_r
                    W_r = np.round(W * scale_r) / scale_r
                    
                    R_r = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
                    err_r = np.max(np.abs(R_r - T))
                    
                    if err_r < 1e-9:
                        best_result = (U_r.tolist(), V_r.tolist(), W_r.tolist())
                        found = True
                        break
                
                if found:
                    break
        
        if found:
            break
    
    if best_result is None:
        # Fall back to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    else:
        U_list, V_list, W_list = best_result
        return {"U": U_list, "V": V_list, "W": W_list}
