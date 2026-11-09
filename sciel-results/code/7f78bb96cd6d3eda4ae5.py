import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Schoolbook baseline
    rank_sb = n * m * p  # 54
    U_sb = np.zeros((nm, rank_sb))
    V_sb = np.zeros((mp, rank_sb))
    W_sb = np.zeros((pn, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m + j, col] = 1
                V_sb[j*p + k, col] = 1
                W_sb[k*n + i, col] = 1
                col += 1
                
    # Build the multiplication tensor T
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    start_time = time.time()
    time_limit = 140
    
    best_rank = rank_sb
    best_U = U_sb.copy()
    best_V = V_sb.copy()
    best_W = W_sb.copy()
    
    lower_bound = 30
    
    for rank in range(rank_sb - 1, lower_bound - 1, -1):
        if time.time() - start_time > time_limit - 20:
            break
        
        found = False
        max_restarts = 5
        
        for restart in range(max_restarts):
            if time.time() - start_time > time_limit - 20:
                break
            
            np.random.seed(42 + restart * 137 + rank * 7)
            
            U = np.random.randn(nm, rank)
            V = np.random.randn(mp, rank)
            W = np.random.randn(pn, rank)
            
            U /= (np.linalg.norm(U) + 1e-10)
            V /= (np.linalg.norm(V) + 1e-10)
            W /= (np.linalg.norm(W) + 1e-10)
            
            max_iter = 100
            prev_error = float('inf')
            for iteration in range(max_iter):
                if time.time() - start_time > time_limit - 20:
                    break
                
                # Solve for U
                P_flat = np.einsum('br,cr->bcr', V, W).reshape(mp * pn, rank)
                T_flat_a = T.reshape(nm, mp * pn)
                try:
                    U_new, _, _, _ = np.linalg.lstsq(P_flat.T, T_flat_a.T, rcond=None)
                    U = U_new.T
                except:
                    break
                
                # Solve for V
                P_flat_v = np.einsum('ar,cr->acr', U, W).reshape(nm * pn, rank)
                T_flat_b = T.transpose(1, 0, 2).reshape(mp, nm * pn)
                try:
                    V_new, _, _, _ = np.linalg.lstsq(P_flat_v.T, T_flat_b.T, rcond=None)
                    V = V_new.T
                except:
                    break
                
                # Solve for W
                P_flat_w = np.einsum('ar,br->abr', U, V).reshape(nm * mp, rank)
                T_flat_c = T.transpose(2, 0, 1).reshape(pn, nm * mp)
                try:
                    W_new, _, _, _ = np.linalg.lstsq(P_flat_w.T, T_flat_c.T, rcond=None)
                    W = W_new.T
                except:
                    break
                
                if iteration % 10 == 0 or iteration == max_iter - 1:
                    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
                    error = np.linalg.norm(T_recon - T)
                    if error < 1e-8:
                        found = True
                        break
                    if error > prev_error * 0.999 and iteration > 20:
                        params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
                        
                        def obj_grad(p):
                            u = p[:nm*rank].reshape(nm, rank)
                            v = p[nm*rank:nm*rank+mp*rank].reshape(mp, rank)
                            w = p[nm*rank+mp*rank:].reshape(pn, rank)
                            t_rec = np.einsum('ar,br,cr->abc', u, v, w)
                            err = t_rec - T
                            gu = np.einsum('abc,br,cr->ar', err, v, w)
                            gv = np.einsum('abc,ar,cr->br', err, u, w)
                            gw = np.einsum('abc,ar,br->cr', err, u, v)
                            grad = np.concatenate([gu.flatten(), gv.flatten(), gw.flatten()])
                            return np.sum(err**2), grad
                        
                        res = minimize(obj_grad, params, method='L-BFGS-B', jac=True, options={'maxiter': 50, 'ftol': 1e-12})
                        
                        u = res.x[:nm*rank].reshape(nm, rank)
                        v = res.x[nm*rank:nm*rank+mp*rank].reshape(mp, rank)
                        w = res.x[nm*rank+mp*rank:].reshape(pn, rank)
                        
                        T_recon = np.einsum('ar,br,cr->abc', u, v, w)
                        error = np.linalg.norm(T_recon - T)
                        if error < 1e-8:
                            U, V, W = u, v, w
                            found = True
                            break
                        
                        U, V, W = u, v, w
                    prev_error = error
            
            if found:
                best_rank = rank
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
                break
        
        if found:
            continue
    
    if best_rank < rank_sb:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
