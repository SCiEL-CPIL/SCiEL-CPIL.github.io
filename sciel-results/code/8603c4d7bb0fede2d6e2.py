# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 145.0  # Leave 5s margin
    
    # Schoolbook fallback
    rank_schoolbook = n * m * p
    U_fb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_fb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_fb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][col] = 1
                V_fb[j * p + k][col] = 1
                W_fb[k * n + i][col] = 1
                col += 1
    
    best = {"U": U_fb, "V": V_fb, "W": W_fb}
    best_rank = rank_schoolbook
    
    # Precompute indices
    nz_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                nz_indices.append((a, b, c))
                
    nz_a = np.array([x[0] for x in nz_indices])
    nz_b = np.array([x[1] for x in nz_indices])
    nz_c = np.array([x[2] for x in nz_indices])
    
    # Precompute mask for zeros
    mask_nz = np.zeros((n*m, m*p, p*n), dtype=bool)
    for a, b, c in nz_indices:
        mask_nz[a, b, c] = True
    mask_zero = ~mask_nz

    def tensor_residual(U, V, W):
        """Compute residual of tensor decomposition."""
        r = U.shape[1]
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        T_rec = np.einsum('at,bt,ct->abc', U, V, W)
        
        val_nz = T_rec[nz_a, nz_b, nz_c]
        res_nz = np.sum((val_nz - 1.0) ** 2)
        
        # Sum of squares of zero entries
        val_zero = T_rec[mask_zero]
        res_zero = np.sum(val_zero ** 2)
        
        return res_nz + res_zero

    def verify(U, V, W, tol=1e-7):
        """Verify exact decomposition."""
        r = U.shape[1]
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        T_rec = np.einsum('at,bt,ct->abc', U, V, W)
        
        val_nz = T_rec[nz_a, nz_b, nz_c]
        if np.max(np.abs(val_nz - 1.0)) > tol:
            return False
        
        val_zero = T_rec[mask_zero]
        if np.max(np.abs(val_zero)) > tol:
            return False
            
        return True

    # Try ranks from lower to higher
    # Start from 40, as 2*5*5=50 is schoolbook, and rank reduction is expected
    for r in range(40, 50):
        if time.time() > deadline - 10:
            break
        
        best_residual = float('inf')
        found = False
        
        for trial in range(8):
            if time.time() > deadline - 5:
                break
            
            # Initialization
            U = np.random.randn(n*m, r) * 0.5
            V = np.random.randn(m*p, r) * 0.5
            W = np.random.randn(p*n, r) * 0.5
            
            for t in range(r):
                nu = np.linalg.norm(U[:, t])
                nv = np.linalg.norm(V[:, t])
                nw = np.linalg.norm(W[:, t])
                if nu > 1e-10 and nv > 1e-10 and nw > 1e-10:
                    scale = (nu * nv * nw) ** (1/3)
                    U[:, t] /= scale
                    V[:, t] /= scale
                    W[:, t] /= scale
            
            params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            def obj(x):
                Ux = x[:n*m*r].reshape(n*m, r)
                Vx = x[n*m*r:n*m*r+m*p*r].reshape(m*p, r)
                Wx = x[n*m*r+m*p*r:].reshape(p*n, r)
                return tensor_residual(Ux, Vx, Wx)
            
            try:
                res = minimize(obj, params, method='L-BFGS-B', 
                              options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8})
                
                if res.fun < best_residual:
                    best_residual = res.fun
                    U_opt = res.x[:n*m*r].reshape(n*m, r)
                    V_opt = res.x[n*m*r:n*m*r+m*p*r].reshape(m*p, r)
                    W_opt = res.x[n*m*r+m*p*r:].reshape(p*n, r)
                    
                    # Check unrounded
                    if verify(U_opt, V_opt, W_opt, tol=1e-7):
                        best = {
                            "U": U_opt.tolist(),
                            "V": V_opt.tolist(),
                            "W": W_opt.tolist()
                        }
                        best_rank = r
                        found = True
                        break
                    
                    # Try rounding to nearest 0.5
                    U_round = np.round(U_opt * 2) / 2
                    V_round = np.round(V_opt * 2) / 2
                    W_round = np.round(W_opt * 2) / 2
                    
                    if verify(U_round, V_round, W_round, tol=1e-6):
                        best = {
                            "U": U_round.tolist(),
                            "V": V_round.tolist(),
                            "W": W_round.tolist()
                        }
                        best_rank = r
                        found = True
                        break
                        
            except Exception:
                pass
        
        if found:
            break
            
    return best
# EVOLVE-BLOCK-END
