"""Matrix multiplication 3x3 x 3x4 tensor decomposition."""
import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s kill

    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 36

    # Precompute valid indices
    valid_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                alpha = i * m + j
                beta = j * p + k
                gamma = k * n + i
                valid_indices.append((alpha, beta, gamma))
    
    valid_set = set(valid_indices)
    
    # All possible indices
    all_indices = []
    for a in range(n * m):
        for b in range(m * p):
            for g in range(p * n):
                all_indices.append((a, b, g))
    invalid_indices = [t for t in all_indices if t not in valid_set]

    def compute_tensor(U, V, W):
        """Compute T[alpha,beta,gamma] = sum_c U[alpha,c]*V[beta,c]*W[gamma,c]"""
        return np.einsum('ac,bc,dc->abd', U, V, W)

    def loss_and_grad(params, U_shape, V_shape, W_shape, r):
        num_u = U_shape[0] * r
        num_v = V_shape[0] * r
        num_w = W_shape[0] * r
        
        U = params[:num_u].reshape(U_shape)
        V = params[num_u:num_u+num_v].reshape(V_shape)
        W = params[num_u+num_v:num_u+num_v+num_w].reshape(W_shape)
        
        T = compute_tensor(U, V, W)
        
        loss = 0.0
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        # Valid indices: target 1
        for (a, b, g) in valid_indices:
            err = T[a, b, g] - 1.0
            loss += err * err
            grad_U[a, :] += 2.0 * err * V[b, :] * W[g, :]
            grad_V[b, :] += 2.0 * err * U[a, :] * W[g, :]
            grad_W[g, :] += 2.0 * err * U[a, :] * V[b, :]
        
        # Invalid indices: target 0
        for (a, b, g) in invalid_indices:
            err = T[a, b, g]
            loss += err * err
            grad_U[a, :] += 2.0 * err * V[b, :] * W[g, :]
            grad_V[b, :] += 2.0 * err * U[a, :] * W[g, :]
            grad_W[g, :] += 2.0 * err * U[a, :] * V[b, :]
        
        grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
        return loss, grad

    def snap_and_check(U, V, W):
        U_s = np.round(U * 2) / 2
        V_s = np.round(V * 2) / 2
        W_s = np.round(W * 2) / 2
        
        T = compute_tensor(U_s, V_s, W_s)
        
        for (a, b, g) in valid_indices:
            if abs(T[a, b, g] - 1.0) > 1e-9:
                return False
        for (a, b, g) in invalid_indices:
            if abs(T[a, b, g]) > 1e-9:
                return False
        return True

    def schoolbook():
        rank = rank_schoolbook
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

    def to_lists(U, V, W):
        U = np.round(np.asarray(U, dtype=float) * 2) / 2
        V = np.round(np.asarray(V, dtype=float) * 2) / 2
        W = np.round(np.asarray(W, dtype=float) * 2) / 2
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Try ranks from lower bound to schoolbook-1
    # Use gradient-based optimization with multiple restarts
    best_result = schoolbook()
    
    # Try ranks from 24 to 35 (lower bound estimate)
    for r in range(24, rank_schoolbook):
        if time.time() > deadline:
            break

        U_shape = (n * m, r)
        V_shape = (m * p, r)
        W_shape = (p * n, r)
        
        found = False
        # Number of restarts: adapt based on time
        max_restarts = 15
        for attempt in range(max_restarts):
            if time.time() > deadline:
                break
            
            # Initialize with small random values from a structured set
            # Use {-1, 0, 1} as initial values for better convergence
            U_init = np.random.choice([-1, 0, 1], size=U_shape)
            V_init = np.random.choice([-1, 0, 1], size=V_shape)
            W_init = np.random.choice([-1, 0, 1], size=W_shape)
            
            params = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                res = minimize(loss_and_grad, params, 
                               args=(U_shape, V_shape, W_shape, r),
                               jac=True, method='L-BFGS-B',
                               options={'maxiter': 300, 'maxfun': 3000, 'ftol': 1e-12})
            except Exception:
                continue
            
            if res.fun < 1e-8:
                num_u = U_shape[0] * r
                num_v = V_shape[0] * r
                U_final = res.x[:num_u].reshape(U_shape)
                V_final = res.x[num_u:num_u+num_v].reshape(V_shape)
                W_final = res.x[num_u+num_v:num_u+num_v+num_w].reshape(W_shape)
                
                if snap_and_check(U_final, V_final, W_final):
                    best_result = to_lists(U_final, V_final, W_final)
                    found = True
                    break
        
        if found:
            break

    return best_result
# EVOLVE-BLOCK-END
