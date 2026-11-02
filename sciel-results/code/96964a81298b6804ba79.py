import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 8]

def solve_small(n, m, p, target_rank, max_iter=1000, timeout=10):
    """
    Finds a decomposition for n x m x p with target_rank.
    Returns U, V, W if found, else None.
    """
    # Target Tensor
    # T[u, v, w] = 1 if u=im+j, v=jp+k, w=pn+i is consistent?
    # W index is k*n + i.
    # U index is i*m + j.
    # V index is j*p + k.
    
    U_dim = n * m
    V_dim = m * p
    W_dim = p * n
    
    T = np.zeros((U_dim, V_dim, W_dim))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0
                
    # Initialize parameters
    # U (U_dim, r), V (V_dim, r), W (W_dim, r)
    # Flatten for optimization
    np.random.seed(42)
    
    def flatten_params(params, r):
        U = params[:U_dim * r].reshape(U_dim, r)
        V = params[U_dim * r : U_dim * r + V_dim * r].reshape(V_dim, r)
        W = params[U_dim * r + V_dim * r : U_dim * r + V_dim * r + W_dim * r].reshape(W_dim, r)
        return U, V, W

    def objective(params):
        U, V, W = flatten_params(params, target_rank)
        # Compute T_algo
        # T_algo[u, v, w] = sum_r U[u, r] V[v, r] W[w, r]
        # Efficiently: sum_r U[:, r] V[:, r]^T W[:, r]^T
        # This is an einsum
        T_algo = np.einsum('ur,vr,wr->uvw', U, V, W)
        error = T_algo - T
        return np.sum(error**2)

    def gradient(params):
        U, V, W = flatten_params(params, target_rank)
        T_algo = np.einsum('ur,vr,wr->uvw', U, V, W)
        error = T_algo - T
        # d(error^2)/d(U) = 2 * sum_{v,w} error[u,v,w] V[v,r] W[w,r]
        # dU[u, r] = 2 * sum_{v,w} error[u,v,w] V[v,r] W[w,r]
        dU = 2.0 * np.einsum('uvw,vr,wr->ur', error, V, W)
        dV = 2.0 * np.einsum('uvw,ur,wr->vr', error, U, W)
        dW = 2.0 * np.einsum('uvw,ur,vr->wr', error, U, V)
        return np.concatenate([dU.flatten(), dV.flatten(), dW.flatten()])

    # Try a few restarts
    best_val = np.inf
    best_params = None
    
    start_time = time.time()
    for _ in range(3):
        # Random init
        params = np.random.randn(U_dim * target_rank + V_dim * target_rank + W_dim * target_rank) * 0.1
        res = minimize(objective, params, jac=gradient, method='L-BFGS-B', 
                       options={'maxiter': max_iter, 'maxfun': 10000})
        
        if time.time() - start_time > timeout:
            break
            
        if res.fun < best_val:
            best_val = res.fun
            best_params = res.x
            
        # If we found a very good solution, break
        if best_val < 1e-8:
            break

    if best_val < 1e-8 and best_params is not None:
        U, V, W = flatten_params(best_params, target_rank)
        # Round to nearest integer or half-integer if possible?
        # The problem allows integer/half-integer.
        # For now, return the floats. The evaluator checks if they "exactly decompose".
        # If they are close to 0, we might need to snap them.
        # However, Laderman solutions are often integers or halves.
        # Let's snap to nearest 0.5 if close.
        U = np.where(np.abs(U - np.round(U*2)/2) < 1e-5, np.round(U*2)/2, U)
        V = np.where(np.abs(V - np.round(V*2)/2) < 1e-5, np.round(V*2)/2, V)
        W = np.where(np.abs(W - np.round(W*2)/2) < 1e-5, np.round(W*2)/2, W)
        
        # Verify snapped version
        T_algo_snapped = np.einsum('ur,vr,wr->uvw', U, V, W)
        if np.max(np.abs(T_algo_snapped - T)) < 1e-5:
            return U, V, W
            
    return None

def construct():
    n, m, p = DIMENSIONS
    
    # Try to find low rank for 3x3x2
    # If found, use it to build 3x3x8
    # 3x3x2 rank 11 is expected.
    
    U_sub, V_sub, W_sub = solve_small(3, 3, 2, 11, max_iter=200, timeout=20)
    
    if U_sub is not None:
        # Construct 3x3x8
        # U is 9 x (4*11)
        # V is 24 x (4*11)
        # W is 24 x (4*11)
        
        r_sub = U_sub.shape[1]
        r_total = 4 * r_sub
        
        U = np.zeros((9, r_total))
        V = np.zeros((24, r_total))
        W = np.zeros((24, r_total))
        
        # The 4 blocks correspond to k in [0,1], [2,3], [4,5], [6,7]
        # For each block b (0..3):
        # U part is U_sub (same for all)
        # V part is V_sub, but rows correspond to j*p+k.
        # In sub-problem, p=2. Rows are j*2+k_sub.
        # In full problem, p=8. Rows are j*8+k_full.
        # k_full = 2*b + k_sub.
        # So V_full[j, 2*b+k_sub, col] = V_sub[j, k_sub, col] ?
        # V index in sub: v_sub = j*2 + k_sub.
        # V index in full: v_full = j*8 + (2*b + k_sub).
        # So for each block b, we copy V_sub rows into V_full rows corresponding to that block.
        
        for b in range(4):
            u_offset = b * r_sub
            # U is the same for all blocks, so just copy
            U[:, u_offset:u_offset+r_sub] = U_sub
            
            for j in range(3):
                for k_sub in range(2):
                    v_sub_idx = j * 2 + k_sub
                    k_full = 2 * b + k_sub
                    v_full_idx = j * 8 + k_full
                    V[v_full_idx, u_offset:u_offset+r_sub] = V_sub[v_sub_idx]
                    
            # W index in sub: w_sub = k_sub * 3 + i.
            # W index in full: w_full = k_full * 3 + i.
            for i in range(3):
                for k_sub in range(2):
                    w_sub_idx = k_sub * 3 + i
                    k_full = 2 * b + k_sub
                    w_full_idx = k_full * 3 + i
                    W[w_full_idx, u_offset:u_offset+r_sub] = W_sub[w_sub_idx]
                    
        return {"U": U, "V": V, "W": W}
        
    # Fallback to schoolbook
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
