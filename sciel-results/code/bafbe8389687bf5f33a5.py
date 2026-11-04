import numpy as np
import time

def construct():
    n, m, p = 3, 3, 4
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                g = k*n + i
                T[a, b, g] = 1.0
    
    start_time = time.time()
    timeout = 110.0
    
    # Try ranks from low to high, with more attempts at lower ranks
    # Lower bound is at least 12, but practical minimum is higher
    for r in range(20, 37):
        if time.time() - start_time > timeout:
            break
        
        # Allocate more attempts to lower ranks where success is harder
        if r < 25:
            max_attempts = 50
        elif r < 30:
            max_attempts = 100
        elif r < 34:
            max_attempts = 200
        else:
            max_attempts = 300
        
        for attempt in range(max_attempts):
            if time.time() - start_time > timeout:
                break
            
            # Use sparse random initialization with values from {-1, 0, 1}
            # Higher probability of non-zero for lower ranks
            if r < 25:
                p_nonzero = 0.3
            elif r < 30:
                p_nonzero = 0.25
            else:
                p_nonzero = 0.2
            
            U = np.random.choice([-1, 0, 1], size=(n*m, r), p=[0.5*(1-p_nonzero), p_nonzero, 0.5*(1-p_nonzero)])
            V = np.random.choice([-1, 0, 1], size=(m*p, r), p=[0.5*(1-p_nonzero), p_nonzero, 0.5*(1-p_nonzero)])
            
            # Solve for W using linear system
            W = np.zeros((p*n, r))
            valid = True
            
            for g in range(p*n):
                # Build M_g: (n*m * m*p) x r matrix
                # M_g[(a,b), c] = U[a,c] * V[b,c]
                M_g = np.zeros((n*m * m*p, r))
                b_g = np.zeros(n*m * m*p)
                
                idx = 0
                for a in range(n*m):
                    for b in range(m*p):
                        M_g[idx, :] = U[a, :] * V[b, :]
                        b_g[idx] = T[a, b, g]
                        idx += 1
                
                # Solve least squares
                try:
                    w_g, residuals, rank_g, sv = np.linalg.lstsq(M_g, b_g, rcond=None)
                except:
                    valid = False
                    break
                    
                if residuals is not None and len(residuals) > 0 and residuals[0] > 1e-10:
                    valid = False
                    break
                    
                W[g, :] = w_g
            
            if not valid:
                continue
            
            # Snap to half-integers
            U_s = np.round(U * 2) / 2
            V_s = np.round(V * 2) / 2
            W_s = np.round(W * 2) / 2
            
            # Verify exactness
            T_approx = np.einsum('ac,bc,gc->abg', U_s, V_s, W_s)
            if np.allclose(T_approx, T, atol=1e-9):
                return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
    
    # Fallback to schoolbook rank 36
    rank = 36
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
