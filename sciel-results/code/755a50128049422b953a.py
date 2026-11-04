import numpy as np
import time

DIMENSIONS = [2, 3, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n  # 6, 24, 16
    
    # Build tensor T of shape (nm, mp, pn)
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    rank_lb = max(nm, mp, pn)  # 24
    
    best_result = None
    
    for r in range(rank_lb, 49):
        if time.time() > deadline - 10:
            break
        
        for attempt in range(30):
            if time.time() > deadline - 10:
                break
            
            np.random.seed(attempt)
            U = np.random.randn(nm, r)
            V = np.random.randn(mp, r)
            W = np.random.randn(pn, r)
            
            for iteration in range(100):
                if time.time() > deadline - 10:
                    break
                
                # Update W: T[a,b,c'] = sum_r U[a,r]*V[b,r]*W[c',r]
                # M[a*mp+b, r] = U[a,r]*V[b,r]
                M = np.zeros((nm*mp, r))
                for a in range(nm):
                    for b in range(mp):
                        M[a*mp+b, :] = U[a, :] * V[b, :]
                T_flat = T.reshape(nm*mp, pn)
                W = np.linalg.lstsq(M, T_flat, rcond=None)[0]
                
                # Update V: T[a,b,c'] = sum_r U[a,r]*V[b,r]*W[c',r]
                # N[a*pn+c', r] = U[a,r]*W[c',r]
                N = np.zeros((nm*pn, r))
                for a in range(nm):
                    for c in range(pn):
                        N[a*pn+c, :] = U[a, :] * W[c, :]
                T_flat2 = T.transpose(0,2,1).reshape(nm*pn, mp)
                V = np.linalg.lstsq(N, T_flat2, rcond=None)[0]
                
                # Update U: T[a,b,c'] = sum_r U[a,r]*V[b,r]*W[c',r]
                # P[b*pn+c', r] = V[b,r]*W[c',r]
                P = np.zeros((mp*pn, r))
                for b in range(mp):
                    for c in range(pn):
                        P[b*pn+c, :] = V[b, :] * W[c, :]
                T_flat3 = T.transpose(1,2,0).reshape(mp*pn, nm)
                U = np.linalg.lstsq(P, T_flat3, rcond=None)[0]
            
            # Check
            T_rec = np.zeros((nm, mp, pn))
            for rr in range(r):
                T_rec += np.einsum('a,b,c->abc', U[:,rr], V[:,rr], W[:,rr])
            
            if np.allclose(T_rec, T, atol=1e-7):
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                
                T_rec_r = np.zeros((nm, mp, pn))
                for rr in range(r):
                    T_rec_r += np.einsum('a,b,c->abc', U_r[:,rr], V_r[:,rr], W_r[:,rr])
                
                if np.allclose(T_rec_r, T, atol=1e-7):
                    best_result = {
                        "U": U_r.tolist(),
                        "V": V_r.tolist(),
                        "W": W_r.tolist()
                    }
                    break
        
        if best_result is not None:
            break
    
    if best_result is not None:
        return best_result
    
    # Fallback schoolbook
    rank = n*m*p
    U = [[0]*rank for _ in range(nm)]
    V = [[0]*rank for _ in range(mp)]
    W = [[0]*rank for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
