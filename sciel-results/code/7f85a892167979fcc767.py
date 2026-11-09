import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [4, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS  # 4, 4, 6
    nm = n * m  # 16
    mp = m * p  # 24
    pn = p * n  # 24
    
    # Precompute index arrays
    i_idx, j_idx, k_idx = [], [], []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                i_idx.append(i)
                j_idx.append(j)
                k_idx.append(k)
    i_idx = np.array(i_idx)
    j_idx = np.array(j_idx)
    k_idx = np.array(k_idx)
    
    U_rows = (i_idx * m + j_idx).astype(int)
    V_rows = (j_idx * p + k_idx).astype(int)
    W_rows = (k_idx * n + i_idx).astype(int)
    num_terms = len(i_idx)  # 96
    
    # Schoolbook solution
    schoolbook_rank = n * m * p
    U_sb = np.zeros((nm, schoolbook_rank))
    V_sb = np.zeros((mp, schoolbook_rank))
    W_sb = np.zeros((pn, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1.0
                V_sb[j*p+k, col] = 1.0
                W_sb[k*n+i, col] = 1.0
                col += 1
    
    best_rank = schoolbook_rank
    best_U = U_sb.copy()
    best_V = V_sb.copy()
    best_W = W_sb.copy()
    
    np.random.seed(999)
    
    def make_objective(target_rank):
        def obj_and_grad(params):
            U_p = params[:nm*target_rank].reshape(nm, target_rank)
            V_p = params[nm*target_rank:nm*target_rank+mp*target_rank].reshape(mp, target_rank)
            W_p = params[nm*target_rank+mp*target_rank:].reshape(pn, target_rank)
            
            # Vectorized computation
            # current[t] = sum_c U_p[U_rows[t], c] * V_p[V_rows[t], c] * W_p[W_rows[t], c]
            # = sum_c U_p[U_rows, c] * V_p[V_rows, c] * W_p[W_rows, c] for each t
            
            # Build matrices: U_sel[t,c] = U_p[U_rows[t], c], etc.
            U_sel = U_p[U_rows, :]  # shape (num_terms, target_rank)
            V_sel = V_p[V_rows, :]  # shape (num_terms, target_rank)
            W_sel = W_p[W_rows, :]  # shape (num_terms, target_rank)
            
            # current[t] = sum_c U_sel[t,c] * V_sel[t,c] * W_sel[t,c]
            current = np.sum(U_sel * V_sel * W_sel, axis=1)
            
            error = 1.0 - current
            loss = np.sum(error**2)
            
            # Gradient: d_loss/d_U_p[i,j,c] = -2 * sum_t error[t] * V_sel[t,c] * W_sel[t,c] * (U_rows[t]==i*m+j)
            # More efficiently:
            dU_sel = -2 * error[:, None] * V_sel * W_sel  # (num_terms, target_rank)
            dV_sel = -2 * error[:, None] * U_sel * W_sel
            dW_sel = -2 * error[:, None] * U_sel * V_sel
            
            # Scatter back
            dU = np.zeros_like(U_p)
            dV = np.zeros_like(V_p)
            dW = np.zeros_like(W_p)
            
            np.add.at(dU, (U_rows, np.arange(target_rank)[None, :]), dU_sel)
            np.add.at(dV, (V_rows, np.arange(target_rank)[None, :]), dV_sel)
            np.add.at(dW, (W_rows, np.arange(target_rank)[None, :]), dW_sel)
            
            grad = np.concatenate([dU.flatten(), dV.flatten(), dW.flatten()])
            return loss, grad
        
        return obj_and_grad
    
    # Try target ranks from low to high
    for target_rank in range(40, best_rank + 1):
        if time.time() > deadline:
            break
        
        max_restarts = 40 if target_rank < 65 else 20
        obj_func = make_objective(target_rank)
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            
            U_r = np.random.randn(nm, target_rank) * 0.3
            V_r = np.random.randn(mp, target_rank) * 0.3
            W_r = np.random.randn(pn, target_rank) * 0.3
            
            x0 = np.concatenate([U_r.flatten(), V_r.flatten(), W_r.flatten()])
            
            try:
                result = minimize(obj_func, x0, method='L-BFGS-B', jac=True,
                                 options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-6, 'maxls': 50})
                
                final_loss = result.fun
                if final_loss < 1e-10:
                    U_sol = result.x[:nm*target_rank].reshape(nm, target_rank)
                    V_sol = result.x[nm*target_rank:nm*target_rank+mp*target_rank].reshape(mp, target_rank)
                    W_sol = result.x[nm*target_rank+mp*target_rank:].reshape(pn, target_rank)
                    
                    # Double-check
                    U_sel = U_sol[U_rows, :]
                    V_sel = V_sol[V_rows, :]
                    W_sel = W_sol[W_rows, :]
                    current = np.sum(U_sel * V_sel * W_sel, axis=1)
                    check_loss = np.sum((1.0 - current)**2)
                    
                    if check_loss < 1e-8 and target_rank < best_rank:
                        best_rank = target_rank
                        best_U = U_sol.copy()
                        best_V = V_sol.copy()
                        best_W = W_sol.copy()
                        
            except Exception:
                pass
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
