import time
import numpy as np
import itertools
from scipy.optimize import minimize

_start_time = time.time()
TIME_LIMIT = 148.0

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    U_t = U.reshape(n, m, rank)
    V_t = V.reshape(m, p, rank)
    W_t = W.reshape(p, n, rank)
    
    decomp_tensor = np.zeros((n, m, p), dtype=float)
    for r in range(rank):
        u_r = U[:, r].reshape(n, m)
        v_r = V[:, r].reshape(m, p)
        w_r = W[:, r].reshape(p, n)
        decomp_tensor += u_r * v_r * w_r.transpose(1, 0) # w_r[k, i] -> w_r.T[i, k]
        
    target = np.ones((n, m, p), dtype=float)
    if not np.allclose(decomp_tensor, target, atol=tol):
        return False
    return True

def _try_find_decomposition(n, m, p, target_rank, max_iters=500, time_limit=10.0):
    # Flattened variables: U (n*m*rank), V (m*p*rank), W (p*n*rank)
    size_U = n * m * target_rank
    size_V = m * p * target_rank
    size_W = p * n * target_rank
    total_size = size_U + size_V + size_W
    
    def objective(x):
        U_flat = x[:size_U]
        V_flat = x[size_U:size_U+size_V]
        W_flat = x[size_U+size_V:]
        
        U = U_flat.reshape(n, m, target_rank)
        V = V_flat.reshape(m, p, target_rank)
        W = W_flat.reshape(p, n, target_rank)
        
        # Compute sum_r U[i,j,r] V[j,k,r] W[k,i,r]
        # We want this to be 1 for all i,j,k
        # Use einsum for speed
        # U: (n, m, r), V: (m, p, r), W: (p, n, r)
        # Term: U[i,j,r] * V[j,k,r] * W[k,i,r]
        # Sum over r.
        # Let's do it step by step to avoid huge intermediates if possible, 
        # but n,m,p are small (3,5,5).
        # U * W.T? No.
        
        # Contract U and W first?
        # U[i,j,r] * W[k,i,r] -> (n, m, p, r)
        # Then multiply by V[j,k,r] and sum r.
        
        # Actually, just use einsum
        # 'ijr,jkr,kr->ijk'
        val = np.einsum('ijr,jkr,kr->ijk', U, V, W)
        
        error = val - 1.0
        return np.sum(error**2)
    
    # Initialize with schoolbook-like structure but with random perturbations
    # Or just random
    np.random.seed(42)
    x0 = np.random.randn(total_size) * 0.1
    
    # We want to find a solution. 
    # Since the system is underdetermined, there are many solutions.
    # We can add a regularization to keep values small, which might help with half-integer snapping.
    
    start_time = time.time()
    # Use L-BFGS-B or COBYLA? L-BFGS-B is good.
    # But we need to handle the time limit.
    
    # We'll run a few iterations and check time.
    # Since we can't easily interrupt scipy.optimize, we'll use a callback to check time.
    
    stop_flag = [False]
    def callback(xk):
        if time.time() - start_time > time_limit:
            stop_flag[0] = True
            return True
        return False
    
    try:
        res = minimize(objective, x0, method='L-BFGS-B', callback=callback, 
                       options={'maxiter': max_iters, 'gtol': 1e-8})
        
        if stop_flag[0]:
            return None
            
        if res.fun < 1e-9:
            x_opt = res.x
            U_flat = x_opt[:size_U]
            V_flat = x_opt[size_U:size_U+size_V]
            W_flat = x_opt[size_U+size_V:]
            
            U = U_flat.reshape(n*m, target_rank)
            V = V_flat.reshape(m*p, target_rank)
            W = W_flat.reshape(p*n, target_rank)
            
            # Check for half-integers
            # Multiply by 2 and check if close to integer
            U2 = U * 2
            V2 = V * 2
            W2 = W * 2
            
            U_rounded = np.round(U2)
            V_rounded = np.round(V2)
            W_rounded = np.round(W2)
            
            if np.allclose(U2, U_rounded, atol=1e-6) and \
               np.allclose(V2, V_rounded, atol=1e-6) and \
               np.allclose(W2, W_rounded, atol=1e-6):
                
                U_half = U_rounded / 2.0
                V_half = V_rounded / 2.0
                W_half = W_rounded / 2.0
                
                if _verify_decomposition(U_half, V_half, W_half, n, m, p, tol=1e-9):
                    return {
                        "U": U_half.tolist(),
                        "V": V_half.tolist(),
                        "W": W_half.tolist()
                    }
            
            # If not half-integer, we might still return it if the evaluator accepts floats?
            # "integer/half-integer entries" suggests we should stick to that.
            # If we can't find a half-integer solution, we return None.
            
    except Exception:
        pass
        
    return None

def construct():
    if time.time() - _start_time > TIME_LIMIT:
        raise TimeoutError("Time limit exceeded")
    
    n, m, p = 3, 5, 5
    
    # Try rank 74
    if time.time() - _start_time < TIME_LIMIT - 20:
        res_74 = _try_find_decomposition(n, m, p, 74, time_limit=15.0)
        if res_74:
            return res_74
            
    # Fall back to schoolbook rank 75
    rank_schoolbook = n * m * p
    U = [[0] * rank_schoolbook for _ in range(n * m)]
    V = [[0] * rank_schoolbook for _ in range(m * p)]
    W = [[0] * rank_schoolbook for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
                
    return {"U": U, "V": V, "W": W}
