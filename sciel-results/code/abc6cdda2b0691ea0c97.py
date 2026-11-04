# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin for evaluator overhead
    
    n, m, p = 3, 5, 6
    rank_schoolbook = n * m * p  # 90
    
    # Schoolbook fallback
    U_sb = np.zeros((n*m, rank_schoolbook))
    V_sb = np.zeros((m*p, rank_schoolbook))
    W_sb = np.zeros((p*n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    def to_list(mat):
        return mat.tolist()
    
    schoolbook = {"U": to_list(U_sb), "V": to_list(V_sb), "W": to_list(W_sb)}
    
    # Try to find lower rank using optimization
    best_decomp = schoolbook
    
    for target_rank in [85, 80, 75, 70, 65, 60, 55, 50]:
        if time.time() > deadline - 25:
            break
        
        found = False
        for attempt in range(3):
            if time.time() > deadline - 15:
                break
            
            # Initialize with small random values
            np.random.seed(attempt * 1000 + target_rank)  # Reproducibility
            U_init = np.random.randn(n*m, target_rank) * 0.3
            V_init = np.random.randn(m*p, target_rank) * 0.3
            W_init = np.random.randn(p*n, target_rank) * 0.3
            
            # Flatten for optimization
            u_size = n*m * target_rank
            v_size = m*p * target_rank
            w_size = p*n * target_rank
            total_size = u_size + v_size + w_size
            
            def flatten(U, V, W):
                return np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            def unflatten(x):
                U = x[:u_size].reshape(n*m, target_rank)
                V = x[u_size:u_size+v_size].reshape(m*p, target_rank)
                W = x[u_size+v_size:total_size].reshape(p*n, target_rank)
                return U, V, W
            
            def objective(x):
                U, V, W = unflatten(x)
                error = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            # Compute sum_a U[i*m+j,a] * V[j*p+k,a] * W[k*n+i,a]
                            s = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                            error += (s - 1.0) ** 2
                return error
            
            x0 = flatten(U_init, V_init, W_init)
            
            try:
                # Use L-BFGS-B with limited iterations
                result = minimize(objective, x0, method='L-BFGS-B', 
                                 options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-8})
                
                if result.fun < 1e-4:  # Close enough to try rounding
                    U_opt, V_opt, W_opt = unflatten(result.x)
                    
                    # Round to nearest half-integers
                    U_round = np.round(U_opt * 2) / 2.0
                    V_round = np.round(V_opt * 2) / 2.0
                    W_round = np.round(W_opt * 2) / 2.0
                    
                    # Check if rounded version is exact
                    valid = True
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                s = np.dot(U_round[i*m+j], V_round[j*p+k] * W_round[k*n+i])
                                if abs(s - 1.0) > 1e-9:
                                    valid = False
                                    break
                        if not valid:
                            break
                    if not valid:
                        break
                    
                    if valid:
                        best_decomp = {"U": to_list(U_round), "V": to_list(V_round), "W": to_list(W_round)}
                        found = True
                        break
                        
            except Exception:
                continue
        
        if found:
            break
    
    return best_decomp
# EVOLVE-BLOCK-END
