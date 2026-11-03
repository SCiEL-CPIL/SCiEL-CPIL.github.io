import numpy as np
import time

def construct():
    n, m, p = 2, 3, 6
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # Trivial solution
    rank_trivial = n * m * p  # 36
    U_trivial = np.zeros((n*m, rank_trivial))
    V_trivial = np.zeros((m*p, rank_trivial))
    W_trivial = np.zeros((p*n, rank_trivial))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_trivial[i*m+j, col] = 1
                V_trivial[j*p+k, col] = 1
                W_trivial[k*n+i, col] = 1
                col += 1
    
    best_result = {"U": U_trivial.tolist(), "V": V_trivial.tolist(), "W": W_trivial.tolist()}
    best_rank = rank_trivial
    
    # Precompute indices
    indices = []
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    indices.append((i, j, jp, k, 1.0 if j == jp else 0.0))
    
    def check_solution(U, V, W):
        """Check if the solution is exact."""
        for (i, j, jp, k, target) in indices:
            val = np.dot(U[i*m+j, :], V[jp*p+k, :] * W[k*n+i, :])
            if abs(val - target) > 1e-9:
                return False
        return True
    
    def compute_error(U, V, W):
        """Compute the sum of squared errors."""
        err = 0.0
        for (i, j, jp, k, target) in indices:
            val = np.dot(U[i*m+j, :], V[jp*p+k, :] * W[k*n+i, :])
            err += (val - target) ** 2
        return err
    
    # Try ranks from low to high
    for target_rank in range(18, best_rank):
        if time.time() > deadline - 20:
            break
        
        found = False
        for attempt in range(3):
            if time.time() > deadline - 20:
                break
            
            # Random initialization with small values
            np.random.seed(42 + target_rank * 100 + attempt)
            U = np.random.randn(n*m, target_rank) * 0.1
            V = np.random.randn(m*p, target_rank) * 0.1
            W = np.random.randn(p*n, target_rank) * 0.1
            
            # Use iterative optimization
            params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            def loss_and_grad(params):
                U_p = params[:n*m*target_rank].reshape(n*m, target_rank)
                V_p = params[n*m*target_rank:(n*m+m*p)*target_rank].reshape(m*p, target_rank)
                W_p = params[(n*m+m*p)*target_rank:].reshape(p*n, target_rank)
                
                err = 0.0
                grad_U = np.zeros_like(U_p)
                grad_V = np.zeros_like(V_p)
                grad_W = np.zeros_like(W_p)
                
                for (i, j, jp, k, target) in indices:
                    u_idx = i*m + j
                    v_idx = jp*p + k
                    w_idx = k*n + i
                    
                    # val = sum_r U[u_idx, r] * V[v_idx, r] * W[w_idx, r]
                    prod = V_p[v_idx, :] * W_p[w_idx, :]
                    val = np.dot(U_p[u_idx, :], prod)
                    diff = val - target
                    
                    err += diff * diff
                    
                    # Gradients
                    grad_U[u_idx, :] += 2 * diff * prod
                    grad_V[v_idx, :] += 2 * diff * U_p[u_idx, :] * W_p[w_idx, :]
                    grad_W[w_idx, :] += 2 * diff * U_p[u_idx, :] * V_p[v_idx, :]
                
                grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
                return err, grad
            
            # Simple gradient descent with line search
            lr = 0.01
            for step in range(200):
                if time.time() > deadline - 20:
                    break
                
                loss, grad = loss_and_grad(params)
                if loss < 1e-10:
                    break
                
                # Gradient descent step
                params_new = params - lr * grad
                
                # Check if loss decreased
                loss_new, _ = loss_and_grad(params_new)
                if loss_new < loss:
                    params = params_new
                else:
                    lr *= 0.5
                
                if lr < 1e-8:
                    break
            
            # Check solution
            U_sol = params[:n*m*target_rank].reshape(n*m, target_rank)
            V_sol = params[n*m*target_rank:(n*m+m*p)*target_rank].reshape(m*p, target_rank)
            W_sol = params[(n*m+m*p)*target_rank:].reshape(p*n, target_rank)
            
            # Round to nearest half-integer
            U_r = np.round(U_sol * 2) / 2
            V_r = np.round(V_sol * 2) / 2
            W_r = np.round(W_sol * 2) / 2
            
            if check_solution(U_r, V_r, W_r):
                best_rank = target_rank
                best_result = {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                found = True
                break
        
        if found:
            break
    
    return best_result
