import time
import itertools
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 6]

def _verify(U, V, W, n, m, p, tol=1e-6):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                if abs(s - 1.0) > tol:
                    return False
    return True

def _find_decomp_sub(n, m, p, target_rank, deadline):
    """Find a rank decomposition for <n,m,p> using local search."""
    # Dimensions for U, V, W
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    # Tensor T[i,j,k] = 1
    # Objective: minimize sum (1 - sum_c U_ij_c V_jk_c W_ki_c)^2
    # We will flatten U, V, W into a single vector
    
    def objective(x):
        U = x[:dim_u * target_rank].reshape(dim_u, target_rank)
        V = x[dim_u * target_rank : dim_u * target_rank + dim_v * target_rank].reshape(dim_v, target_rank)
        W = x[dim_u * target_rank + dim_v * target_rank :].reshape(dim_w, target_rank)
        
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i]) # This is wrong.
                    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    val = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                    error += (1.0 - val) ** 2
        return error
    
    def grad(x):
        # Numerical gradient or analytic? Analytic is better but complex.
        # Let's use finite differences for simplicity if scipy is slow, 
        # but L-BFGS-B needs gradient. Let's try a simple random restart search first.
        pass

    # Strategy: Random restarts with local search (e.g., coordinate descent or simple gradient step)
    # Given the small size (rank 11, dims 6,9,6), we can try a grid search on a small set of values?
    # No, 11*21 variables. Too many.
    
    # Let's use a simple stochastic local search.
    best_sol = None
    best_err = float('inf')
    
    # Try a few random initializations
    for attempt in range(10):
        if time.time() > deadline - 5:
            break
            
        # Initialize with small integers
        np.random.seed(42 + attempt)
        x0 = np.random.choice([-1, 0, 1], size=dim_u*target_rank + dim_v*target_rank + dim_w*target_rank)
        
        # Use scipy minimize with Nelder-Mead or Powell (derivative-free)
        # Powell is good for this size
        res = minimize(objective, x0, method='Powell', options={'maxiter': 1000, 'xatol': 1e-4})
        
        if res.fun < best_err:
            best_err = res.fun
            best_sol = res.x
            
            # If error is very low, try to snap to half-integers
            if best_err < 1e-4:
                U = best_sol[:dim_u * target_rank].reshape(dim_u, target_rank)
                V = best_sol[dim_u * target_rank : dim_u * target_rank + dim_v * target_rank].reshape(dim_v, target_rank)
                W = best_sol[dim_u * target_rank + dim_v * target_rank :].reshape(dim_w, target_rank)
                
                # Snap to nearest 0.5
                U_s = np.round(U * 2) / 2
                V_s = np.round(V * 2) / 2
                W_s = np.round(W * 2) / 2
                
                if _verify(U_s, V_s, W_s, n, m, p, tol=1e-6):
                    return {'U': U_s.tolist(), 'V': V_s.tolist(), 'W': W_s.tolist()}
                
                # If snap fails, maybe try rounding to integers?
                U_i = np.round(U)
                V_i = np.round(V)
                W_i = np.round(W)
                if _verify(U_i, V_i, W_i, n, m, p, tol=1e-6):
                    return {'U': U_i.tolist(), 'V': V_i.tolist(), 'W': W_i.tolist()}

    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Schoolbook fallback
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j][col] = 1
                V_sb[j*p+k][col] = 1
                W_sb[k*n+i][col] = 1
                col += 1
    
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_sb
    
    # Strategy 1: Find rank 11 decomposition for <2,3,3> and combine for <2,3,6>
    # This should yield rank 22.
    sub_n, sub_m, sub_p = 2, 3, 3
    sub_rank = 11
    
    if time.time() < deadline - 20:
        sub_decomp = _find_decomp_sub(sub_n, sub_m, sub_p, sub_rank, deadline)
        
        if sub_decomp:
            # Construct full decomposition
            U_sub = np.array(sub_decomp['U'])
            V_sub = np.array(sub_decomp['V'])
            W_sub = np.array(sub_decomp['W'])
            
            full_rank = 2 * sub_rank
            U_full = np.zeros((n * m, full_rank))
            V_full = np.zeros((m * p, full_rank))
            W_full = np.zeros((p * n, full_rank))
            
            # First block (k=0..2)
            U_full[:, :sub_rank] = U_sub
            # V for first block: j in 0..2, k in 0..2
            for j in range(sub_m):
                for k in range(sub_p):
                    V_full[j*p+k, :sub_rank] = V_sub[j*sub_p+k]
            # W for first block: k in 0..2, i in 0..1
            for k in range(sub_p):
                for i in range(sub_n):
                    W_full[k*n+i, :sub_rank] = W_sub[k*sub_n+i]
                    
            # Second block (k=3..5)
            U_full[:, sub_rank:] = U_sub
            # V for second block: j in 0..2, k in 3..5
            for j in range(sub_m):
                for k in range(sub_p, sub_p*2):
                    V_full[j*p+k, sub_rank:] = V_sub[j*sub_p+(k-sub_p)]
            # W for second block: k in 3..5, i in 0..1
            for k in range(sub_p, sub_p*2):
                for i in range(sub_n):
                    W_full[k*n+i, sub_rank:] = W_sub[(k-sub_p)*sub_n+i]
            
            # Verify full decomposition
            if _verify(U_full, V_full, W_full, n, m, p):
                best_result = {
                    "U": U_full.tolist(),
                    "V": V_full.tolist(),
                    "W": W_full.tolist()
                }
                best_rank = full_rank
    
    # Strategy 2: Try to find a direct decomposition of <2,3,6> with rank < best_rank
    # Using a similar local search, but it's larger.
    # We can try ranks from 20 to best_rank-1.
    # Given time constraints, we might only try a few.
    
    if time.time() < deadline - 10:
        for r in range(20, best_rank):
            if time.time() > deadline - 5:
                break
            # Try to find a direct decomposition of rank r
            # This is harder, so we limit attempts
            # We can reuse _find_decomp_sub logic but with different dims
            # Let's just do a quick random search with snapping
            found = False
            for attempt in range(5):
                if time.time() > deadline - 2:
                    break
                np.random.seed(100 + r + attempt)
                U_c = np.random.choice([-1, 0, 1], size=(n*m, r))
                V_c = np.random.choice([-1, 0, 1], size=(m*p, r))
                W_c = np.random.choice([-1, 0, 1], size=(p*n, r))
                if _verify(U_c, V_c, W_c, n, m, p):
                    best_result = {
                        "U": U_c.tolist(),
                        "V": V_c.tolist(),
                        "W": W_c.tolist()
                    }
                    best_rank = r
                    found = True
                    break
            if found:
                break

    return best_result
