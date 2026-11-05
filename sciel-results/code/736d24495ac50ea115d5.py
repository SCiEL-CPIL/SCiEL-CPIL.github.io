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

    # Precompute valid indices and target tensor
    # T_target[a,b,g] is 1 if (a,b,g) is a valid multiplication index, else 0
    # a = i*m + j
    # b = j*p + k
    # g = k*n + i
    T_target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                g = k * n + i
                T_target[a, b, g] = 1.0

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
        E = T - T_target
        
        loss = np.sum(E * E)
        
        # Gradients
        # dL/dU[a,c] = 2 * sum_{b,g} E[a,b,g] V[b,c] W[g,c]
        grad_U = 2.0 * np.einsum('abg,bc,gc->ac', E, V, W)
        # dL/dV[b,c] = 2 * sum_{a,g} E[a,b,g] U[a,c] W[g,c]
        grad_V = 2.0 * np.einsum('abg,ac,gc->bc', E, U, W)
        # dL/dW[g,c] = 2 * sum_{a,b} E[a,b,g] U[a,c] V[b,c]
        grad_W = 2.0 * np.einsum('abg,ac,bc->gc', E, U, V)
        
        grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
        return loss, grad

    def snap_and_check(U, V, W):
        U_s = np.round(U * 2) / 2
        V_s = np.round(V * 2) / 2
        W_s = np.round(W * 2) / 2
        
        T = compute_tensor(U_s, V_s, W_s)
        
        # Check max error
        if np.max(np.abs(T - T_target)) > 1e-9:
            return False
        return True

    def schoolbook():
        rank = rank_schoolbook
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
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    def to_lists(U, V, W):
        U = np.round(np.asarray(U, dtype=float) * 2) / 2
        V = np.round(np.asarray(V, dtype=float) * 2) / 2
        W = np.round(np.asarray(W, dtype=float) * 2) / 2
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Try ranks from lower bound to schoolbook-1
    # Use gradient-based optimization with multiple restarts
    best_result = schoolbook()
    
    # Try ranks from 22 to 35
    # Rank 22 is the rank of 4x4x4, and 3x3x4 is a sub-tensor, so 22 is a valid upper bound.
    # Lower bounds suggest < 22 is unlikely.
    for r in range(22, rank_schoolbook):
        if time.time() > deadline:
            break

        U_shape = (n * m, r)
        V_shape = (m * p, r)
        W_shape = (p * n, r)
        
        found = False
        
        # Adaptive restarts: more for lower ranks, less for higher
        # We have ~110s total.
        # Let's allocate time per rank.
        # If we find a solution, we break.
        
        # Number of restarts: 
        # Vectorized loss is fast, so we can do more.
        # Allocate more time/restarts for the most promising ranks (22-25)
        max_restarts = 100 if r < 25 else 50
        
        for attempt in range(max_restarts):
            if time.time() > deadline:
                break
            
            # Initialization strategies:
            # Use discrete values from {-1, -0.5, 0, 0.5, 1} to bias towards integer/half-integer solutions
            # This aligns with the requirement for half-integer entries.
            
            U_init = np.random.choice([-1, -0.5, 0, 0.5, 1], size=U_shape)
            V_init = np.random.choice([-1, -0.5, 0, 0.5, 1], size=V_shape)
            W_init = np.random.choice([-1, -0.5, 0, 0.5, 1], size=W_shape)
            
            params = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                res = minimize(loss_and_grad, params, 
                               args=(U_shape, V_shape, W_shape, r),
                               jac=True, method='L-BFGS-B',
                               options={'maxiter': 1000, 'maxfun': 10000, 'ftol': 1e-12, 'gtol': 1e-8})
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
