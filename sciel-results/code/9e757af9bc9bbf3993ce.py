import numpy as np
from scipy.optimize import minimize
import time

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    start_time = time.time()
    deadline = 600 - 60 # 540s limit
    
    best_w = None
    best_s = None
    best_val = -np.inf
    
    # Try n=32 and n=64
    for n in [32, 64]:
        if time.time() > deadline: break
        
        support = np.arange(n, dtype=np.float64)
        
        # Precompute T
        # T[k, l, i, j] = 1 if support[i] + support[j] < 2*support[l] - support[k]
        # Vectorized creation
        si = support[:, None, None, None]
        sj = support[None, :, None, None]
        sk = support[None, None, :, None]
        sl = support[None, None, None, :]
        
        # Condition: si + sj < 2*sl - sk
        # Shapes:
        # si: (n, 1, 1, 1)
        # sj: (1, n, 1, 1)
        # sk: (1, 1, n, 1)
        # sl: (1, 1, 1, n)
        # 2*sl - sk: (1, 1, n, n)
        # si + sj: (n, n, 1, 1)
        
        lhs = si + sj # (n, n, 1, 1)
        rhs = 2*sl - sk # (1, 1, n, n)
        
        # We need T[k, l, i, j]
        # Current lhs is indexed [i, j, k, l]? No.
        # si is index i. sj is index j.
        # sk is index k. sl is index l.
        # lhs[i, j, 0, 0] = si[i] + sj[j]
        # rhs[0, 0, k, l] = 2*sl[l] - sk[k]
        
        # We want T[k, l, i, j] = 1 if lhs[i, j] < rhs[k, l]
        # lhs_expanded = lhs[:, :, None, None] # (n, n, 1, 1)
        # rhs_expanded = rhs[None, None, :, :] # (1, 1, n, n)
        
        T = (lhs[:, :, None, None] < rhs[None, None, :, :]).astype(np.float64)
        # T shape: (n, n, n, n) indexed [i, j, k, l]
        # We need T indexed [k, l, i, j] for easier contraction?
        # Or just keep it and adjust einsum.
        
        def compute_P(w):
            # w shape (n,)
            # W_ij = w_i w_j
            W = np.outer(w, w)
            # V_kl = sum_{i,j} T[i, j, k, l] W[i, j]
            # T is (n, n, n, n) [i, j, k, l]
            # V = np.tensordot(T, W, axes=([0,1],[0,1]))
            # V shape (n, n) [k, l]
            V = np.tensordot(T, W, axes=([0,1],[0,1]))
            # P = sum_{k,l} w_k w_l V_kl
            P = np.sum(np.outer(w, w) * V)
            return P
            
        def compute_grad(w):
            # dP/dw_m
            # P = sum_{k,l,i,j} T[i,j,k,l] w_k w_l w_i w_j
            # dP/dw_m = sum_{l,i,j} T[i,j,m,l] w_l w_i w_j  (m=k)
            #           + sum_{k,i,j} T[i,j,k,m] w_k w_i w_j  (m=l)
            #           + sum_{k,l,j} T[i,j,k,l] delta_{i,m} w_k w_l w_j (m=i)
            #           + sum_{k,l,i} T[i,j,k,l] delta_{j,m} w_k w_l w_i (m=j)
            
            # Term 1 (m=k):
            # Sum over l, i, j of T[i, j, m, l] w_l w_i w_j
            # T[:,:,m,:] has shape (n, n, 1, n) -> squeeze to (n, n, n) [i, j, l]
            # We need sum_{i,j,l} T[i,j,m,l] w_i w_j w_l
            # This is T[:,:,m,:] contracted with w,w,w
            
            # Let's use einsum
            # T is (i, j, k, l)
            
            # G1[m] = sum_{i,j,l} T[i,j,m,l] w_i w_j w_l
            # G2[m] = sum_{i,j,k} T[i,j,k,m] w_i w_j w_k
            # G3[m] = sum_{k,l,j} T[m,j,k,l] w_k w_l w_j
            # G4[m] = sum_{k,l,i} T[i,m,k,l] w_k w_l w_i
            
            G1 = np.einsum('ijml, i, j, l -> m', T, w, w, w)
            G2 = np.einsum('ijkml, i, j, k -> m', T, w, w, w) # Wait, T is i,j,k,l. So T[i,j,k,m]
            # T shape (i,j,k,l).
            # G2 needs T[i,j,k,m].
            # np.einsum('ijk l, i, j, k -> l', T, w, w, w)
            G2 = np.einsum('ijkl, i, j, k -> l', T, w, w, w)
            
            # G3 needs T[m,j,k,l].
            # np.einsum('m j k l, k, l, j -> m', T, w, w, w)
            G3 = np.einsum('mjkl, k, l, j -> m', T, w, w, w)
            
            # G4 needs T[i,m,k,l].
            # np.einsum('i m k l, k, l, i -> m', T, w, w, w)
            G4 = np.einsum('imkl, k, l, i -> m', T, w, w, w)
            
            return G1 + G2 + G3 + G4

        # Optimization
        # Use SLSQP
        # Constraints: sum(w) = 1, w >= 0
        
        # Restart
        n_restarts = 20
        for r in range(n_restarts):
            if time.time() > deadline: break
            
            if r == 0:
                w0 = np.ones(n) / n
            else:
                w0 = np.random.dirichlet(np.ones(n))
                
            # Add a bit of noise to break symmetry?
            
            res = minimize(
                lambda w: -compute_P(w), 
                w0, 
                jac=lambda w: -compute_grad(w),
                method='SLSQP',
                bounds=[(0, 1)] * n,
                constraints={'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
                options={'maxiter': 1000, 'ftol': 1e-12}
            )
            
            w_opt = res.x
            val = compute_P(w_opt)
            
            if val > best_val:
                best_val = val
                best_w = w_opt
                best_s = support

    # Return best
    # Normalize weights
    w_norm = best_w / np.sum(best_w)
    return best_s, w_norm
