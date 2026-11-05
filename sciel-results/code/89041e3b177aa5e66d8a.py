import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115
    
    nm = n * m  # 9
    mp = m * p  # 21
    pn = p * n  # 21
    
    # Build the target tensor
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    def compute_error_and_grad(x, r):
        U = x[:nm*r].reshape(nm, r)
        V = x[nm*r:nm*r+mp*r].reshape(mp, r)
        W = x[nm*r+mp*r:].reshape(pn, r)
        
        # S[a,b,c] = sum_t U[a,t] V[b,t] W[c,t]
        S = np.einsum('at,bt,ct->abc', U, V, W)
        diff = S - T
        err = np.sum(diff**2)
        
        # Gradient
        # d(err)/d(U[a,t]) = 2 * sum_{b,c} diff[a,b,c] * V[b,t] * W[c,t]
        # = 2 * diff[a,:,:] @ (V[:,t] * W[:,t])... no
        # = 2 * sum_{b,c} diff[a,b,c] V[b,t] W[c,t]
        # Let M = V * W elementwise? No.
        # For fixed a,t: sum_{b,c} diff[a,b,c] V[b,t] W[c,t]
        # = (diff[a,:,:] * V[:,t][:,None] * W[:,t][None,:]).sum()
        # Or: let D_a = diff[a,:,:], then sum_{b,c} D_a[b,c] V[b,t] W[c,t]
        # = D_a @ (V[:,t] * W[:,t])? No, that's sum_b D_a[b,:] @ (V[b,t]*W[:,t])... 
        # Actually: sum_{b,c} D_a[b,c] V[b,t] W[c,t] = (D_a * (V[:,t][:,None] * W[:,t][None,:])).sum()
        
        # More efficiently:
        # grad_U[a,t] = 2 * sum_{b,c} diff[a,b,c] V[b,t] W[c,t]
        # = 2 * (diff @ (V * W)[:,t])? Let's see:
        # (diff @ X)[a,t] = sum_b diff[a,b,:] @ X[b,t]... no, diff is 3D.
        
        # Let me use einsum:
        # grad_U = 2 * np.einsum('abc,bt,ct->at', diff, V, W)
        # grad_V = 2 * np.einsum('abc,at,ct->bt', diff, U, W)
        # grad_W = 2 * np.einsum('abc,at,bt->ct', diff, U, V)
        
        grad_U = 2.0 * np.einsum('abc,bt,ct->at', diff, V, W)
        grad_V = 2.0 * np.einsum('abc,at,ct->bt', diff, U, W)
        grad_W = 2.0 * np.einsum('abc,at,bt->ct', diff, U, V)
        
        grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
        return err, grad
    
    # Schoolbook solution
    best_rank = nm * mp // p  # 9*21/7 = 27? No, schoolbook is n*m*p = 63
    best_rank = n * m * p  # 63
    U_school = np.zeros((nm, best_rank))
    V_school = np.zeros((mp, best_rank))
    W_school = np.zeros((pn, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1.0
                V_school[j*p+k, col] = 1.0
                W_school[k*n+i, col] = 1.0
                col += 1
    
    best_result = {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
    
    # Try to find lower ranks
    for r in range(best_rank - 1, 20, -1):
        if time.time() > deadline:
            break
        
        found = False
        max_restarts = 3
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            
            np.random.seed(restart * 1000 + r)
            U_init = np.random.randn(nm, r) * 0.5
            V_init = np.random.randn(mp, r) * 0.5
            W_init = np.random.randn(pn, r) * 0.5
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                res = minimize(
                    compute_error_and_grad,
                    x0,
                    args=(r,),
                    method='L-BFGS-B',
                    jac=True,
                    options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
                )
                
                if res.fun < 1e-10:
                    U_sol = res.x[:nm*r].reshape(nm, r)
                    V_sol = res.x[nm*r:nm*r+mp*r].reshape(mp, r)
                    W_sol = res.x[nm*r+mp*r:].reshape(pn, r)
                    
                    # Verify
                    S = np.einsum('at,bt,ct->abc', U_sol, V_sol, W_sol)
                    max_err = np.max(np.abs(S - T))
                    
                    if max_err < 1e-6:
                        best_rank = r
                        best_result = {
                            "U": U_sol.tolist(),
                            "V": V_sol.tolist(),
                            "W": W_sol.tolist()
                        }
                        found = True
                        break
                        
            except Exception:
                pass
        
        if not found:
            # If we can't find rank r, try r-1, but likely to fail
            continue
    
    return best_result
