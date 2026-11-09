import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    dim_u = n * m  # 6
    dim_v = m * p  # 18
    dim_w = p * n  # 12
    
    nz_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                nz_indices.append((u_idx, v_idx, w_idx))
    
    start_time = time.time()
    deadline = 140.0
    
    # Strategy: Decompose 3x6 into two 3x3 blocks.
    # Sub-problem: 2x3 x 3x3.
    # If we find rank r_s for sub-problem, total rank is 2*r_s.
    # Lower bound for sub-problem is max(6, 9, 6) = 9.
    # So we try r_s = 9, 10, ...
    
    def find_sub_decomposition(r_s):
        """Find rank r_s decomposition for 2x3 x 3x3."""
        n_s, m_s, p_s = 2, 3, 3
        dim_u_s = n_s * m_s  # 6
        dim_v_s = m_s * p_s  # 9
        dim_w_s = p_s * n_s  # 6
        
        T_s = np.zeros((dim_u_s, dim_v_s, dim_w_s))
        for i in range(n_s):
            for j in range(m_s):
                for k in range(p_s):
                    u_idx = i * m_s + j
                    v_idx = j * p_s + k
                    w_idx = k * n_s + i
                    T_s[u_idx, v_idx, w_idx] = 1.0
        
        # Flatten tensor for optimization
        T_flat = T_s.flatten()
        
        best_result = None
        
        # Try multiple random initializations
        num_attempts = 10
        for attempt in range(num_attempts):
            if time.time() > deadline - 10:
                break
                
            np.random.seed(attempt + 100)
            
            # Initialize factors
            U = np.random.randn(dim_u_s, r_s) * 0.5
            V = np.random.randn(dim_v_s, r_s) * 0.5
            W = np.random.randn(dim_w_s, r_s) * 0.5
            
            # Define objective: sum of squared residuals
            def objective(params):
                U_p = params[:dim_u_s * r_s].reshape(dim_u_s, r_s)
                V_p = params[dim_u_s * r_s : dim_u_s * r_s + dim_v_s * r_s].reshape(dim_v_s, r_s)
                W_p = params[dim_u_s * r_s + dim_v_s * r_s :].reshape(dim_w_s, r_s)
                
                T_approx = np.zeros((dim_u_s, dim_v_s, dim_w_s))
                for c in range(r_s):
                    T_approx += np.einsum('i,j,k->ijk', U_p[:, c], V_p[:, c], W_p[:, c])
                
                residual = T_approx - T_s
                return np.sum(residual ** 2)
            
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            # Use L-BFGS-B or similar
            try:
                res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 1000, 'gtol': 1e-8})
                if res.fun < 1e-6:
                    # Check if we can snap to half-integers
                    U_opt = res.x[:dim_u_s * r_s].reshape(dim_u_s, r_s)
                    V_opt = res.x[dim_u_s * r_s : dim_u_s * r_s + dim_v_s * r_s].reshape(dim_v_s, r_s)
                    W_opt = res.x[dim_u_s * r_s + dim_v_s * r_s :].reshape(dim_w_s, r_s)
                    
                    # Snap to half-integers
                    U_snap = np.round(2 * U_opt) / 2
                    V_snap = np.round(2 * V_opt) / 2
                    W_snap = np.round(2 * W_opt) / 2
                    
                    # Verify exactness
                    T_check = np.zeros((dim_u_s, dim_v_s, dim_w_s))
                    for c in range(r_s):
                        T_check += np.einsum('i,j,k->ijk', U_snap[:, c], V_snap[:, c], W_snap[:, c])
                    
                    if np.allclose(T_check, T_s, atol=1e-8):
                        return (U_snap, V_snap, W_snap)
                        
            except Exception as e:
                pass
                
        return None
    
    # Try to find rank 9 sub-decomposition
    sub_res = find_sub_decomposition(9)
    
    if sub_res is not None:
        U_s, V_s, W_s = sub_res
        r_s = U_s.shape[1]
        
        # Construct full decomposition
        U_full = np.zeros((dim_u, 2 * r_s))
        V_full = np.zeros((dim_v, 2 * r_s))
        W_full = np.zeros((dim_w, 2 * r_s))
        
        # Block 1: k in {0,1,2}
        for c in range(r_s):
            U_full[:, c] = U_s[:, c]
            for j in range(3):
                for k in range(3):
                    v_idx_small = j * 3 + k
                    v_idx_full = j * 6 + k
                    V_full[v_idx_full, c] = V_s[v_idx_small, c]
            for k in range(3):
                for i in range(2):
                    w_idx_small = k * 2 + i
                    w_idx_full = k * 2 + i
                    W_full[w_idx_full, c] = W_s[w_idx_small, c]
        
        # Block 2: k in {3,4,5}
        for c in range(r_s):
            c2 = r_s + c
            U_full[:, c2] = U_s[:, c]
            for j in range(3):
                for k in range(3):
                    v_idx_small = j * 3 + k
                    v_idx_full = j * 6 + (k + 3)
                    V_full[v_idx_full, c2] = V_s[v_idx_small, c]
            for k in range(3):
                for i in range(2):
                    w_idx_small = k * 2 + i
                    w_idx_full = (k + 3) * 2 + i
                    W_full[w_idx_full, c2] = W_s[w_idx_small, c]
        
        # Verify full decomposition
        T_full = np.zeros((dim_u, dim_v, dim_w))
        for c in range(2 * r_s):
            T_full += np.einsum('i,j,k->ijk', U_full[:, c], V_full[:, c], W_full[:, c])
        
        valid = True
        for u_idx, v_idx, w_idx in nz_indices:
            if not np.isclose(T_full[u_idx, v_idx, w_idx], 1.0, atol=1e-8):
                valid = False
                break
        
        if valid:
            return {
                "U": U_full.tolist(),
                "V": V_full.tolist(),
                "W": W_full.tolist()
            }
            
    # Fallback: Schoolbook
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
# EVOLVE-BLOCK-END
