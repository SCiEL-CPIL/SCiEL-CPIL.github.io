import time
import random
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 4, 8]
DEADLINE_SECONDS = 115

def _create_target_tensor(n, m, p):
    """Create the sparse target tensor as a dense array for easy comparison."""
    # Dimensions: (n*m, m*p, p*n)
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                T[idx_u, idx_v, idx_w] = 1
    return T

def _reconstruct(U, V, W):
    """Reconstruct the tensor from factors."""
    return np.einsum('ar,br,cr->abc', U, V, W)

def _verify_exact(U, V, W, n, m, p, tol=1e-9):
    """Check if the decomposition is exact."""
    T_rec = _reconstruct(U, V, W)
    T_target = _create_target_tensor(n, m, p)
    return np.allclose(T_rec, T_target, atol=tol)

def _try_rank_sub(n, m, p, r, time_budget, rng):
    """Try to find a rank-r decomposition for M_{n,m,p}."""
    start = time.time()
    nm, mp, pn = n * m, m * p, p * n
    T = _create_target_tensor(n, m, p)
    
    best_error = float('inf')
    best_U, best_V, best_W = None, None, None
    
    # Initialize
    U = rng.normal(size=(nm, r))
    V = rng.normal(size=(mp, r))
    W = rng.normal(size=(pn, r))
    
    # Normalize
    U /= np.linalg.norm(U)
    V /= np.linalg.norm(V)
    W /= np.linalg.norm(W)
    
    x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    
    def objective(x):
        U_flat = x[:nm*r].reshape(nm, r)
        V_flat = x[nm*r:nm*r+mp*r].reshape(mp, r)
        W_flat = x[nm*r+mp*r:].reshape(pn, r)
        T_rec = _reconstruct(U_flat, V_flat, W_flat)
        return np.sum((T_rec - T) ** 2)
        
    try:
        res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 100, 'maxfun': 200})
        x_opt = res.x
        U_opt = x_opt[:nm*r].reshape(nm, r)
        V_opt = x_opt[nm*r:nm*r+mp*r].reshape(mp, r)
        W_opt = x_opt[nm*r+mp*r:].reshape(pn, r)
        
        err = res.fun
        if err < best_error:
            best_error = err
            best_U, best_V, best_W = U_opt, V_opt, W_opt
            
        # Check if exact after rounding
        U_round = np.round(best_U)
        V_round = np.round(best_V)
        W_round = np.round(best_W)
        if _verify_exact(U_round, V_round, W_round, n, m, p):
            return U_round, V_round, W_round
            
        U_half = np.round(best_U * 2) / 2
        V_half = np.round(best_V * 2) / 2
        W_half = np.round(best_W * 2) / 2
        if _verify_exact(U_half, V_half, W_half, n, m, p):
            return U_half, V_half, W_half
            
    except Exception:
        pass
        
    return None

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    rng = np.random.default_rng(42)
    
    # Schoolbook fallback
    U_sb = np.zeros((n*m, n*m*p))
    V_sb = np.zeros((m*p, n*m*p))
    W_sb = np.zeros((p*n, n*m*p))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
                
    # Strategy: Decompose M_{3,4,8} into two M_{3,4,4}
    # M_{3,4,8} = M_{3,4,4} (k=0..3) + M_{3,4,4} (k=4..7)
    # Search for rank 47 for M_{3,4,4}
    
    sub_n, sub_m, sub_p = 3, 4, 4
    target_rank_sub = 47
    
    U_sub = None
    V_sub = None
    W_sub = None
    
    # Try to find rank 47 for M_{3,4,4}
    # Time budget for sub-problem: 50s
    time_budget_sub = 50
    for restart in range(5):
        if time.time() - start_time > DEADLINE_SECONDS - 10:
            break
        res = _try_rank_sub(sub_n, sub_m, sub_p, target_rank_sub, time_budget_sub, rng)
        if res is not None:
            U_sub, V_sub, W_sub = res
            break
            
    if U_sub is not None:
        r = target_rank_sub
        # Construct M_{3,4,8} from two M_{3,4,4}
        U_new = np.zeros((n*m, 2*r))
        V_new = np.zeros((m*p, 2*r))
        W_new = np.zeros((p*n, 2*r))
        
        # First half (k=0..3)
        U_new[:, 0:r] = U_sub
        V_new[0:sub_m*sub_p, 0:r] = V_sub # j=0..3, k=0..3
        W_new[0:sub_p*sub_n, 0:r] = W_sub # k=0..3, i=0..2
        
        # Second half (k=4..7)
        U_new[:, r:2*r] = U_sub
        V_new[sub_m*sub_p:m*p, r:2*r] = V_sub # j=0..3, k=4..7 (shifted)
        W_new[sub_p*sub_n:p*n, r:2*r] = W_sub # k=4..7, i=0..2 (shifted)
        
        # Verify
        if _verify_exact(U_new, V_new, W_new, n, m, p):
            return {"U": U_new.tolist(), "V": V_new.tolist(), "W": W_new.tolist()}
            
    # Fallback to schoolbook
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
