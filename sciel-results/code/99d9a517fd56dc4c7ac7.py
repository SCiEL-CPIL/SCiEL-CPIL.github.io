import numpy as np
from scipy.optimize import least_squares
import time

def construct():
    n, m, p = 2, 3, 3
    deadline = time.time() + 145
    
    # Build target tensor T[a,b,c]
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a,b,c] = 1
    
    # Precompute slices for efficient ALS
    # T_a[b,c] = T[a,b,c], T_b[a,c] = T[a,b,c], T_c[a,b] = T[a,b,c]
    T_a = [T[a] for a in range(n*m)]  # list of (m*p, p*n)
    T_b = [T[:,b,:] for b in range(m*p)]  # list of (n*m, p*n)
    T_c = [T[:,:,c] for c in range(p*n)]  # list of (n*m, m*p)
    
    def verify(U, V, W):
        """Check if U,V,W exactly decompose T"""
        R = np.einsum('as,bs,cs->abc', U, V, W)
        return np.allclose(R, T, atol=1e-9)
    
    def to_half_int(M):
        """Round to nearest half-integer and verify"""
        M_h = np.round(M * 2) / 2.0
        return M_h
    
    def try_rank(rank, seed_base=0, n_trials=3):
        """Try to find decomposition at given rank using ALS + polish"""
        best_cost = np.inf
        best_x = None
        
        for trial in range(n_trials):
            if time.time() > deadline - 15:
                return None
            
            np.random.seed(seed_base + trial)
            
            # Initialize with structure-aware random values
            scale = 0.5
            U = np.random.randn(n*m, rank) * scale
            V = np.random.randn(m*p, rank) * scale
            W = np.random.randn(p*n, rank) * scale
            
            # ALS iterations
            for it in range(300):
                if time.time() > deadline - 20:
                    break
                
                # Update U: for each a, solve U[a,:] to minimize ||T[a] - (U[a,:] outer V^T W^T)||_F
                # T[a,b,c] = sum_s U[a,s] * V[b,s] * W[c,s]
                # Fix V,W: minimize over U[a,s]: sum_{b,c} (T[a,b,c] - sum_s U[a,s]*V[b,s]*W[c,s])^2
                # This is: for each a, U[a,:] = argmin ||T[a] - (U[a,:] * (V^T W)^T)||_F
                # V^T W has shape (rank, rank), so (V^T W)^T[b,c] = sum_s V[b,s]*W[c,s]
                # Let M_a = V^T W (rank x rank), then T[a] ≈ U[a,:] @ M_a^T
                # So U[a,:] = T[a] @ (M_a^T)^+ = T[a] @ pinv(M_a)
                
                VW = V.T @ W  # (rank, rank)
                try:
                    VW_pinv = np.linalg.pinv(VW)
                except:
                    break
                for a in range(n*m):
                    U[a,:] = T[a] @ VW_pinv.T  # (m*p, p*n) @ (p*n, rank) -> (rank,)
                
                # Update V: fix U,W
                UW = U.T @ W  # (rank, rank)
                try:
                    UW_pinv = np.linalg.pinv(UW)
                except:
                    break
                for b in range(m*p):
                    V[b,:] = T_b[b] @ UW_pinv.T
                
                # Update W: fix U,V
                UV = U.T @ V  # (rank, rank)
                try:
                    UV_pinv = np.linalg.pinv(UV)
                except:
                    break
                for c in range(p*n):
                    W[c,:] = T_c[c] @ UV_pinv.T
                
                # Compute cost
                R = np.einsum('as,bs,cs->abc', U, V, W)
                cost = np.sum((R - T)**2)
                if cost < best_cost:
                    best_cost = cost
                    best_x = (U.copy(), V.copy(), W.copy())
                if cost < 1e-12:
                    break
            
            if best_cost < 1e-10:
                break
        
        if best_x is not None and best_cost < 1e-10:
            U, V, W = best_x
            # Try to round to half-integers
            U_h = to_half_int(U)
            V_h = to_half_int(V)
            W_h = to_half_int(W)
            if verify(U_h, V_h, W_h):
                return (U_h, V_h, W_h)
            
            # Try polish with least_squares starting from current solution
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            nU = n*m*rank
            nV = m*p*rank
            nW = p*n*rank
            
            def residual(x):
                U_r = x[:nU].reshape(n*m, rank)
                V_r = x[nU:nU+nV].reshape(m*p, rank)
                W_r = x[nU+nV:].reshape(p*n, rank)
                R = np.einsum('as,bs,cs->abc', U_r, V_r, W_r)
                return (R - T).flatten()
            
            try:
                sol = least_squares(residual, x0, max_nfev=2000, xtol=1e-14, ftol=1e-14, gtol=1e-14)
                U_p = sol.x[:nU].reshape(n*m, rank)
                V_p = sol.x[nU:nU+nV].reshape(m*p, rank)
                W_p = sol.x[nU+nV:].reshape(p*n, rank)
                
                U_ph = to_half_int(U_p)
                V_ph = to_half_int(V_p)
                W_ph = to_half_int(W_p)
                if verify(U_ph, V_ph, W_ph):
                    return (U_ph, V_ph, W_ph)
            except:
                pass
        
        return None
    
    # Try ranks from low to high
    # Theoretical: for 2x3x3, rank is at most 18, likely can do better
    # Known: rank of 2x2x2 is 7 (Strassen), 2x3x3 might be around 12-15
    for rank in range(11, 19):
        if time.time() > deadline - 25:
            break
        
        result = try_rank(rank, seed_base=rank*100, n_trials=4)
        if result is not None:
            U, V, W = result
            U_out = [[float(x) for x in row] for row in U]
            V_out = [[float(x) for x in row] for row in V]
            W_out = [[float(x) for x in row] for row in W]
            return {"U": U_out, "V": V_out, "W": W_out}
    
    # Fallback: schoolbook rank 18
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1.0
                V[j*p+k][col] = 1.0
                W[k*n+i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}
