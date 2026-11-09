import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 9]

def construct():
    start_time = time.time()
    deadline = 140.0  # Leave 10s margin for 150s limit
    
    n, m, p = DIMENSIONS
    
    # Fallback: Schoolbook
    def schoolbook(n, m, p):
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    # Target: Find rank-17 decomposition for 2x3x3
    # Then replicate for 3 blocks of 9 columns
    # Total rank = 3 * 17 = 51
    
    n_sub, m_sub, p_sub = 2, 3, 3
    target_rank = 17
    
    # Build target tensor T for 2x3x3
    # T[I, J, K] where I=i*m+j, J=j*p+k, K=k*n+i
    # n=2, m=3, p=3
    # I in 0..5, J in 0..8, K in 0..5
    T = np.zeros((n_sub * m_sub, m_sub * p_sub, p_sub * n_sub))
    for i in range(n_sub):
        for j in range(m_sub):
            for k in range(p_sub):
                I = i * m_sub + j
                J = j * p_sub + k
                K = k * n_sub + i
                T[I, J, K] = 1.0

    def flatten_factors(U, V, W):
        return np.concatenate([U.flatten(), V.flatten(), W.flatten()])

    def unflatten_factors(x):
        size_U = n_sub * m_sub * target_rank
        size_V = m_sub * p_sub * target_rank
        size_W = p_sub * n_sub * target_rank
        U = x[:size_U].reshape(n_sub * m_sub, target_rank)
        V = x[size_U:size_U+size_V].reshape(m_sub * p_sub, target_rank)
        W = x[size_U+size_V:].reshape(p_sub * n_sub, target_rank)
        return U, V, W

    def objective(x):
        U, V, W = unflatten_factors(x)
        # Compute sum U_c V_c W_c
        # Tensor product sum
        # Reshape to (rank, nm, mp, pn)
        U_t = U.T  # (rank, nm)
        V_t = V.T  # (rank, mp)
        W_t = W.T  # (rank, pn)
        
        # Efficiently compute the tensor sum
        # This is a sum of outer products
        # Can use einsum
        # T_approx[I, J, K] = sum_c U[I, c] V[J, c] W[K, c]
        T_approx = np.einsum('ic,jc,kc->ijk', U, V, W)
        return np.sum((T_approx - T)**2)

    # Initial guess: random small integers
    np.random.seed(42)
    # Start with schoolbook-like structure but with fewer terms?
    # Just random small values
    x0 = np.random.uniform(-1, 1, size=(n_sub * m_sub + m_sub * p_sub + p_sub * n_sub) * target_rank)
    
    best_result = None
    best_score = float('inf')
    
    # Try a few restarts
    for restart in range(5):
        if time.time() > deadline:
            break
            
        x0 = np.random.uniform(-1, 1, size=(n_sub * m_sub + m_sub * p_sub + p_sub * n_sub) * target_rank)
        
        try:
            res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 200, 'ftol': 1e-10})
            if res.fun < best_score:
                best_score = res.fun
                best_result = res.x
                
                # Check if close to integers/half-integers
                if best_score < 1e-6:
                    U, V, W = unflatten_factors(best_result)
                    # Round to nearest 0.5
                    U_r = np.round(U * 2) / 2
                    V_r = np.round(V * 2) / 2
                    W_r = np.round(W * 2) / 2
                    
                    # Verify exactness
                    T_approx = np.einsum('ic,jc,kc->ijk', U_r, V_r, W_r)
                    if np.allclose(T_approx, T, atol=1e-9):
                        # Success! Construct full 2x3x9 decomposition
                        # Repeat this pattern 3 times
                        total_rank = target_rank * 3
                        U_full = [[0] * total_rank for _ in range(n * m)]
                        V_full = [[0] * total_rank for _ in range(m * p)]
                        W_full = [[0] * total_rank for _ in range(p * n)]
                        
                        # Map indices
                        # Sub-block t (0,1,2) corresponds to columns 3*t to 3*t+2 of B, and 3*t to 3*t+2 of C
                        # In our 2x3x3 decomposition:
                        # U indices: i*m_sub + j (0..5)
                        # V indices: j*p_sub + k (0..8) where k is local 0..2
                        # W indices: k*n_sub + i (0..5) where k is local 0..2
                        
                        # In full 2x3x9:
                        # U indices: i*3 + j (0..5) - same
                        # V indices: j*9 + k_global (0..26)
                        #   k_global = 3*t + k_local
                        # W indices: k_global*2 + i (0..17)
                        #   k_global = 3*t + k_local
                        
                        for t in range(3):
                            for c_local in range(target_rank):
                                c_global = t * target_rank + c_local
                                
                                # Fill U
                                for i_u in range(n_sub * m_sub):
                                    U_full[i_u][c_global] = U_r[i_u, c_local]
                                    
                                # Fill V
                                for j_v in range(m_sub):
                                    for k_local in range(p_sub):
                                        i_v = j_v * p_sub + k_local # Local V index
                                        k_global = 3 * t + k_local
                                        i_v_global = j_v * p + k_global
                                        V_full[i_v_global][c_global] = V_r[i_v, c_local]
                                        
                                # Fill W
                                for k_local in range(p_sub):
                                    for i_w in range(n_sub):
                                        i_w_local = k_local * n_sub + i_w
                                        i_w_global = (3 * t + k_local) * n + i_w
                                        W_full[i_w_global][c_global] = W_r[i_w_local, c_local]
                                        
                        return {"U": U_full, "V": V_full, "W": W_full}
                        
        except Exception:
            pass

    # Fallback to schoolbook
    return schoolbook(n, m, p)
