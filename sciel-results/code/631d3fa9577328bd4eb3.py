import numpy as np
import time

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    num_u = n * m  # 9
    num_v = m * p  # 21
    num_w = p * n  # 21
    
    # Build target tensor T
    T = np.zeros((num_u, num_v, num_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0
    
    T_flat = T.flatten()
    
    def compute_tensor(U, V, W):
        """Compute sum_r U[u,r] * V[v,r] * W[w,r] for all u,v,w"""
        # Use einsum for efficiency
        return np.einsum('ur,vr,wr->uvw', U, V, W)
    
    def check_solution(U, V, W, tol=1e-10):
        T_calc = compute_tensor(U, V, W)
        diff = T_calc - T
        return np.max(np.abs(diff)) < tol
    
    def objective(params, R):
        u_size = num_u * R
        v_size = num_v * R
        w_size = num_w * R
        
        U_flat = params[:u_size]
        V_flat = params[u_size:u_size+v_size]
        W_flat = params[u_size+v_size:u_size+v_size+w_size]
        
        U = U_flat.reshape(num_u, R)
        V = V_flat.reshape(num_v, R)
        W = W_flat.reshape(num_w, R)
        
        T_calc = compute_tensor(U, V, W)
        diff = T_calc.flatten() - T_flat
        return np.dot(diff, diff)
    
    # Schoolbook solution as fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(num_u)]
        V = [[0] * rank for _ in range(num_v)]
        W = [[0] * rank for _ in range(num_w)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    
    # Try ranks from low to high
    # Lower bound: at least max(num_u, num_v, num_w) = 21
    # But tensor rank can be lower than that in some cases
    # For matrix multiplication, rank is at least n*m (or similar bounds)
    # Let's try from 21 upward
    
    best_solution = None
    best_rank = None
    
    # Try a range of ranks
    for R in range(21, 64):
        if time.time() > deadline:
            break
        
        # Multiple restarts for each rank
        max_restarts = 5
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            
            np.random.seed(42 + R * 1000 + restart)
            
            U_init = np.random.randn(num_u, R)
            V_init = np.random.randn(num_v, R)
            W_init = np.random.randn(num_w, R)
            
            # Normalize to help optimization
            U_init /= np.linalg.norm(U_init) + 1e-10
            V_init /= np.linalg.norm(V_init) + 1e-10
            W_init /= np.linalg.norm(W_init) + 1e-10
            
            params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            # Use scipy minimize if available, otherwise simple gradient descent
            try:
                from scipy.optimize import minimize
                res = minimize(
                    objective,
                    params_init,
                    args=(R,),
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
                )
                
                if res.fun < 1e-8:
                    u_size = num_u * R
                    v_size = num_v * R
                    U_opt = res.x[:u_size].reshape(num_u, R)
                    V_opt = res.x[u_size:u_size+v_size].reshape(num_v, R)
                    W_opt = res.x[u_size+v_size:u_size+v_size+w_size].reshape(num_w, R)
                    
                    if check_solution(U_opt, V_opt, W_opt):
                        # Try to round to half-integers
                        U_round = np.round(U_opt * 2) / 2
                        V_round = np.round(V_opt * 2) / 2
                        W_round = np.round(W_opt * 2) / 2
                        
                        if check_solution(U_round, V_round, W_round):
                            return {
                                "U": U_round.tolist(),
                                "V": V_round.tolist(),
                                "W": W_round.tolist()
                            }
                        else:
                            # Return the unrounded solution if it's valid
                            # The task says integer/half-integer, but let's try
                            # Actually, we should only return valid half-integer solutions
                            # If rounding fails, don't accept this solution
                            pass
            except Exception:
                pass
    
    # If no lower rank found, return schoolbook
    return schoolbook()
# EVOLVE-BLOCK-END
