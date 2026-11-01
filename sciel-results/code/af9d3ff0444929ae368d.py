# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    return _construct_impl()

def _construct_impl():
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = 3, 3, 6
    dim_A = n * m  # 9
    dim_B = m * p  # 18
    dim_C = p * n  # 18

    # Precompute the indices of the non-zero entries of the tensor
    T_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_indices.append((a, b, c))

    # Precompute target tensor
    target_T = np.zeros((dim_A, dim_B, dim_C))
    for a, b, c in T_indices:
        target_T[a, b, c] = 1.0

    def generate_schoolbook():
        rank = n * m * p # 54
        U = np.zeros((dim_A, rank))
        V = np.zeros((dim_B, rank))
        W = np.zeros((dim_C, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a, col] = 1
                    V[b, col] = 1
                    W[c, col] = 1
                    col += 1
        return U, V, W

    best_rank = 54
    best_U, best_V, best_W = generate_schoolbook()

    # Try ranks from 39 to 53
    for target_rank in range(39, 54):
        if time.time() > deadline - 20:
            break
        
        rank = target_rank
        # More restarts for lower ranks as they are harder but more valuable
        max_restarts = 10 if rank < 45 else 5
        
        for restart in range(max_restarts):
            if time.time() > deadline - 10:
                break
            
            # Initialize with random integers/half-integers
            U_init = np.random.choice([-1, 0, 1], size=(dim_A, rank))
            V_init = np.random.choice([-1, 0, 1], size=(dim_B, rank))
            W_init = np.random.choice([-1, 0, 1], size=(dim_C, rank))
            
            # Add small noise to avoid exact zeros in gradients if needed
            U_init += np.random.uniform(-0.01, 0.01, size=U_init.shape)
            V_init += np.random.uniform(-0.01, 0.01, size=V_init.shape)
            W_init += np.random.uniform(-0.01, 0.01, size=W_init.shape)
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            def obj_func(x, rank=rank):
                size_U = dim_A * rank
                size_V = dim_B * rank
                U = x[:size_U].reshape(dim_A, rank)
                V = x[size_U:size_U + size_V].reshape(dim_B, rank)
                W = x[size_U + size_V:].reshape(dim_C, rank)
                T = np.einsum('ar,br,cr->abc', U, V, W)
                return np.sum((T - target_T)**2)
            
            try:
                res = minimize(obj_func, x0, method='L-BFGS-B', options={'maxiter': 50, 'maxfun': 500})
            except Exception:
                continue
                
            if res.fun < 1e-4:
                x_opt = res.x
                size_U = dim_A * rank
                size_V = dim_B * rank
                U_opt = x_opt[:size_U].reshape(dim_A, rank)
                V_opt = x_opt[size_U:size_U + size_V].reshape(dim_B, rank)
                W_opt = x_opt[size_U + size_V:].reshape(dim_C, rank)
                
                U_rounded = np.round(U_opt * 2) / 2
                V_rounded = np.round(V_opt * 2) / 2
                W_rounded = np.round(W_opt * 2) / 2
                
                T_check = np.einsum('ar,br,cr->abc', U_rounded, V_rounded, W_rounded)
                if np.allclose(T_check, target_T, atol=1e-9):
                    best_rank = rank
                    best_U, best_V, best_W = U_rounded, V_rounded, W_rounded
                    break
        
        if best_rank < 54:
            break

    U_list = [[float(x) for x in row] for row in best_U.tolist()]
    V_list = [[float(x) for x in row] for row in best_V.tolist()]
    W_list = [[float(x) for x in row] for row in best_W.tolist()]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
