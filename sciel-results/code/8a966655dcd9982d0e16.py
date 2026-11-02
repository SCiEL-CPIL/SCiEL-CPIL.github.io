import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = 2, 3, 6
    
    # Schoolbook decomposition
    schoolbook_rank = n * m * p
    U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
                
    best_rank = schoolbook_rank
    best_U = U_sb
    best_V = V_sb
    best_W = W_sb
    
    # Target: Find decomposition for 2x3x3
    n_sub, m_sub, p_sub = 2, 3, 3
    
    def get_tensor(n, m, p):
        """
        Construct the multiplication tensor T of size (n*m, m*p, p*n)
        T[u, v, w] = 1 if u=(i,j), v=(j,k), w=(k,i) match
        """
        T = np.zeros((n * m, m * p, p * n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = i * m + j
                    v = j * p + k
                    w = k * n + i
                    T[u, v, w] = 1
        return T

    T_sub = get_tensor(n_sub, m_sub, p_sub)
    
    def find_decomp(n, m, p, rank, max_trials=20):
        T = get_tensor(n, m, p)
        size_u = n * m
        size_v = m * p
        size_w = p * n
        
        for trial in range(max_trials):
            if time.time() > deadline - 10:
                return None
            
            # Random init
            U = np.random.randn(size_u, rank) * 0.5
            V = np.random.randn(size_v, rank) * 0.5
            W = np.random.randn(size_w, rank) * 0.5
            
            def params_to_tensors(params):
                U = params[:size_u*rank].reshape(size_u, rank)
                V = params[size_u*rank:size_u*rank + size_v*rank].reshape(size_v, rank)
                W = params[size_u*rank + size_v*rank:].reshape(size_w, rank)
                return U, V, W
                
            def objective(params):
                U, V, W = params_to_tensors(params)
                # Compute sum of outer products
                # T_approx[u, v, w] = sum_r U[u, r] V[v, r] W[w, r]
                # We can compute this efficiently using einsum or loops
                # T_approx = np.einsum('ur,vr,wr->uvw', U, V, W)
                # This might be slow for large rank, but rank is small.
                T_approx = np.einsum('ur,vr,wr->uvw', U, V, W)
                diff = T - T_approx
                return np.sum(diff**2)
                
            initial_params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            try:
                result = minimize(objective, initial_params, method='L-BFGS-B', maxiter=200, 
                                  options={'maxls': 50, 'gtol': 1e-6})
                if result.fun < 1e-8:
                    U, V, W = params_to_tensors(result.x)
                    # Round to half-integers
                    U_r = np.round(U * 2) / 2
                    V_r = np.round(V * 2) / 2
                    W_r = np.round(W * 2) / 2
                    
                    # Verify
                    T_approx = np.einsum('ur,vr,wr->uvw', U_r, V_r, W_r)
                    if np.allclose(T_approx, T, atol=1e-6):
                        return U_r.tolist(), V_r.tolist(), W_r.tolist()
            except Exception:
                pass
        return None

    # Try ranks 7, 8, 9 for 2x3x3
    for rank in [7, 8, 9]:
        if time.time() > deadline - 20:
            break
            
        decomp = find_decomp(n_sub, m_sub, p_sub, rank, max_trials=20)
        if decomp:
            U_sub, V_sub, W_sub = decomp
            # Construct 2x3x6 decomposition
            R = rank
            total_rank = 2 * R
            U_full = [[0] * total_rank for _ in range(n * m)] # 6 x 2R
            V_full = [[0] * total_rank for _ in range(m * p)] # 18 x 2R
            W_full = [[0] * total_rank for _ in range(p * n)] # 12 x 2R
            
            # First block: k in 0..2
            for r in range(R):
                for i in range(n_sub):
                    for j in range(m_sub):
                        U_full[i * m + j][r] = U_sub[i * m_sub + j][r]
                        
                for j in range(m_sub):
                    for k in range(p_sub):
                        V_full[j * p + k][r] = V_sub[j * p_sub + k][r]
                        
                for k in range(p_sub):
                    for i in range(n_sub):
                        W_full[k * n + i][r] = W_sub[k * n_sub + i][r]
                        
                # Second block: k in 3..5
                for i in range(n_sub):
                    for j in range(m_sub):
                        U_full[i * m + j][R + r] = U_sub[i * m_sub + j][r]
                        
                for j in range(m_sub):
                    for k in range(p_sub):
                        V_full[j * p + (k + 3)][R + r] = V_sub[j * p_sub + k][r]
                        
                for k in range(p_sub):
                    for i in range(n_sub):
                        W_full[(k + 3) * n + i][R + r] = W_sub[k * n_sub + i][r]
            
            best_rank = total_rank
            best_U = U_full
            best_V = V_full
            best_W = W_full
            break # Found a solution, stop searching for lower ranks
            
    return {"U": best_U, "V": best_V, "W": best_W}
