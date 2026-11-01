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
    T_sub = np.ones((n_sub, m_sub, p_sub))
    
    # Try to find rank R decomposition for 2x3x3
    # We will try R from 9 to 11
    
    def find_decomp(n, m, p, rank, max_trials=50):
        T = np.ones((n, m, p))
        for trial in range(max_trials):
            if time.time() > deadline - 10:
                return None
            
            # Random init
            U = np.random.randn(n, rank) * 0.5
            V = np.random.randn(m, rank) * 0.5
            W = np.random.randn(p, rank) * 0.5
            
            # ALS or similar
            # Let's use a simple gradient descent or minimize
            # Flatten params
            def params_to_tensors(params):
                U = params[:n*rank].reshape(n, rank)
                V = params[n*rank:n*rank + m*rank].reshape(m, rank)
                W = params[n*rank + m*rank:].reshape(p, rank)
                return U, V, W
                
            def objective(params):
                U, V, W = params_to_tensors(params)
                # Compute sum of outer products
                T_approx = np.zeros((n, m, p))
                for r in range(rank):
                    T_approx += np.einsum('i,j,k->ijk', U[:,r], V[:,r], W[:,r])
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
                    T_approx = np.zeros((n, m, p))
                    for r in range(rank):
                        T_approx += np.einsum('i,j,k->ijk', U_r[:,r], V_r[:,r], W_r[:,r])
                    if np.allclose(T_approx, T, atol=1e-6):
                        return U_r.tolist(), V_r.tolist(), W_r.tolist()
            except Exception:
                pass
        return None

    # Try ranks 9, 10, 11
    for rank in [9, 10, 11]:
        if time.time() > deadline - 20:
            break
            
        decomp = find_decomp(n_sub, m_sub, p_sub, rank, max_trials=20)
        if decomp:
            U_sub, V_sub, W_sub = decomp
            # Construct 2x3x6 decomposition
            # U is 6xR, V is 18xR, W is 12xR
            # U_sub is 2xR, V_sub is 3xR, W_sub is 3xR
            # We duplicate the decomposition for the two 3-column blocks of the 6-column matrix
            # The 6 columns of B are split into two 3-column blocks.
            # The indices for V are j*p + k.
            # For the first block (k=0,1,2), V indices are j*6 + k.
            # For the second block (k=3,4,5), V indices are j*6 + (k+3).
            # We need to map the sub-decomposition to the full indices.
            
            R = rank
            total_rank = 2 * R
            U_full = [[0] * total_rank for _ in range(n * m)] # 6 x 2R
            V_full = [[0] * total_rank for _ in range(m * p)] # 18 x 2R
            W_full = [[0] * total_rank for _ in range(p * n)] # 12 x 2R
            
            # First block: k in 0..2
            for r in range(R):
                # U part: same for both blocks?
                # The term is U[i,j] * V[j,k] * W[k,i]
                # In the sub-problem, W is indexed by k (0..2) and i (0..1).
                # In the full problem, W is indexed by k (0..5) and i (0..1).
                # For the first block, k_full = k_sub.
                # For the second block, k_full = k_sub + 3.
                
                # U_full[i*m + j] gets U_sub[i,j] for both blocks?
                # No, the rank-1 term is a single vector in U, V, W.
                # We create 2R terms.
                # Term r (0..R-1) corresponds to the first block.
                # Term R+r (R..2R-1) corresponds to the second block.
                
                # For term r:
                for i in range(n_sub):
                    for j in range(m_sub):
                        U_full[i * m + j][r] = U_sub[i][j]
                        
                for j in range(m_sub):
                    for k in range(p_sub):
                        V_full[j * p + k][r] = V_sub[j][k]
                        
                for k in range(p_sub):
                    for i in range(n_sub):
                        W_full[k * n + i][r] = W_sub[k][i]
                        
                # For term R+r:
                for i in range(n_sub):
                    for j in range(m_sub):
                        U_full[i * m + j][R + r] = U_sub[i][j]
                        
                for j in range(m_sub):
                    for k in range(p_sub):
                        V_full[j * p + (k + 3)][R + r] = V_sub[j][k]
                        
                for k in range(p_sub):
                    for i in range(n_sub):
                        W_full[(k + 3) * n + i][R + r] = W_sub[k][i]
            
            best_rank = total_rank
            best_U = U_full
            best_V = V_full
            best_W = W_full
            break # Found a solution, stop searching for lower ranks
            
    return {"U": best_U, "V": best_V, "W": best_W}
