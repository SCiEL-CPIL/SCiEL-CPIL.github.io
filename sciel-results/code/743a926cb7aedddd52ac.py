import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    # Construct the target tensor T
    T = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    start_time = time.time()
    deadline = 140.0 # Leave 10s margin
    
    # Schoolbook rank 18 as fallback
    rank_fb = n * m * p
    U_fb = np.zeros((dim_u, rank_fb))
    V_fb = np.zeros((dim_v, rank_fb))
    W_fb = np.zeros((dim_w, rank_fb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j, col] = 1
                V_fb[j * p + k, col] = 1
                W_fb[k * n + i, col] = 1
                col += 1
    
    best_rank = rank_fb
    best_U = U_fb
    best_V = V_fb
    best_W = W_fb

    def check_exact(U, V, W):
        U_int = np.round(U * 2).astype(int)
        V_int = np.round(V * 2).astype(int)
        W_int = np.round(W * 2).astype(int)
        
        T_recon = np.einsum('ar,br,cr->abc', U_int, V_int, W_int)
        return np.all(T_recon == 8 * T)

    def try_rank(r):
        if time.time() > deadline - 5:
            return None
        
        # We use a structured approach. 
        # For matrix multiplication, we can often find solutions with entries in {-1, 0, 1} or half-integers.
        # Random search over a discrete grid might be too slow.
        # Let's use a continuous optimization but with a better initialization strategy.
        # Or, we can try to solve the linear system for a fixed pattern of signs.
        
        # Given the small size, let's try a local search around the schoolbook solution?
        # No, the rank is different.
        
        # Let's try random restarts with L-BFGS-B, but with a more aggressive timeout per restart.
        # And we will try to snap to half-integers.
        
        # To improve success rate, we can try to fix one matrix (e.g., W) to have simple entries?
        # Or just use random restarts with a tighter tolerance and more restarts if time permits.
        
        max_restarts = 50
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break
            
            # Random initialization with a specific scale
            # Sometimes starting with small values helps L-BFGS-B
            U = np.random.randn(dim_u, r) * 0.5
            V = np.random.randn(dim_v, r) * 0.5
            W = np.random.randn(dim_w, r) * 0.5
            
            params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            def objective(params):
                U_p = params[:dim_u*r].reshape(dim_u, r)
                V_p = params[dim_u*r:dim_u*r+dim_v*r].reshape(dim_v, r)
                W_p = params[dim_u*r+dim_v*r:].reshape(dim_w, r)
                T_calc = np.einsum('ar,br,cr->abc', U_p, V_p, W_p)
                err = np.sum((T_calc - T)**2)
                return err
            
            try:
                # Use a tight budget per restart
                # maxfun=1000 is usually enough for small problems to either converge or fail
                res = minimize(objective, params, method='L-BFGS-B', 
                               options={'maxiter': 200, 'maxfun': 1000, 'gtol': 1e-6})
                
                if res.fun < 1e-6:
                    U_p = res.x[:dim_u*r].reshape(dim_u, r)
                    V_p = res.x[dim_u*r:dim_u*r+dim_v*r].reshape(dim_v, r)
                    W_p = res.x[dim_u*r+dim_v*r:].reshape(dim_w, r)
                    
                    # Snap to half-integers
                    U_s = U_p * 2
                    V_s = V_p * 2
                    W_s = W_p * 2
                    
                    U_r = np.round(U_s)
                    V_r = np.round(V_s)
                    W_r = np.round(W_s)
                    
                    if check_exact(U_r/2.0, V_r/2.0, W_r/2.0):
                        return U_r/2.0, V_r/2.0, W_r/2.0
                        
            except Exception:
                pass
                
        return None

    # Try rank 15
    res = try_rank(15)
    if res:
        U, V, W = res
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    # Try rank 16
    if time.time() < deadline - 10:
        res = try_rank(16)
        if res:
            U, V, W = res
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
    # Try rank 17
    if time.time() < deadline - 5:
        res = try_rank(17)
        if res:
            U, V, W = res
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback to rank 18
    return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
# EVOLVE-BLOCK-END
