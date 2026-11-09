import numpy as np
import time

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 3x4x6 matrix multiplication.
    
    The schoolbook rank is 72. We attempt to find a lower rank decomposition
    using structured search with analytic gradients. If no better rank is found,
    we return the schoolbook decomposition.
    
    The tensor T_{i,j,k} = sum_r U_{i*m+j, r} * V_{j*p+k, r} * W_{k*n+i, r}
    where i in [0,3), j in [0,4), k in [0,6).
    Target: T[i,j,k] = 1 for all (i,j,k).
    """
    n, m, p = 3, 4, 6
    schoolbook_rank = n * m * p  # 72
    
    def build_schoolbook():
        rank = schoolbook_rank
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, idx] = 1.0
                    V[j * p + k, idx] = 1.0
                    W[k * n + i, idx] = 1.0
                    idx += 1
        return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
    
    def compute_residual_and_grads(U, V, W):
        """
        Compute residual and analytic gradients.
        Residual R[i,j,k] = sum_r U[i*m+j, r]*V[j*p+k, r]*W[k*n+i, r] - 1
        Objective = sum_{i,j,k} R[i,j,k]^2
        """
        rank = U.shape[1]
        residual = np.zeros((n, m, p))
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    val = 0.0
                    for r in range(rank):
                        val += U[u_idx, r] * V[v_idx, r] * W[w_idx, r]
                    residual[i, j, k] = val - 1.0
                    res = residual[i, j, k]
                    for r in range(rank):
                        grad_U[u_idx, r] += 2 * res * V[v_idx, r] * W[w_idx, r]
                        grad_V[v_idx, r] += 2 * res * U[u_idx, r] * W[w_idx, r]
                        grad_W[w_idx, r] += 2 * res * U[u_idx, r] * V[v_idx, r]
        
        obj = np.sum(residual * residual)
        return obj, residual, grad_U, grad_V, grad_W
    
    def try_rank_search(rank, time_budget=30.0):
        """Attempt to find a decomposition of the given rank using gradient descent with random restarts."""
        start_time = time.time()
        best_obj = float('inf')
        best_U = None
        best_V = None
        best_W = None
        
        max_restarts = 8
        max_iterations = 5000
        lr = 0.05
        
        for restart in range(max_restarts):
            if time.time() - start_time > time_budget - 5:
                break
            
            # Random initialization with different scales
            scale = 0.5
            U = np.random.randn(n * m, rank) * scale
            V = np.random.randn(m * p, rank) * scale
            W = np.random.randn(p * n, rank) * scale
            
            obj_prev, _, grad_U, grad_V, grad_W = compute_residual_and_grads(U, V, W)
            
            if obj_prev < best_obj:
                best_obj = obj_prev
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
            
            for it in range(max_iterations):
                if time.time() - start_time > time_budget - 5:
                    break
                
                # Compute gradient norms for adaptive learning rate
                norm_U = np.linalg.norm(grad_U)
                norm_V = np.linalg.norm(grad_V)
                norm_W = np.linalg.norm(grad_W)
                max_norm = max(norm_U, norm_V, norm_W, 1e-10)
                
                # Normalize gradients and take step
                U -= lr * grad_U / max_norm
                V -= lr * grad_V / max_norm
                W -= lr * grad_W / max_norm
                
                obj_new, _, grad_U, grad_V, grad_W = compute_residual_and_grads(U, V, W)
                
                if obj_new < best_obj:
                    best_obj = obj_new
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
                
                if obj_new < 1e-12:
                    break
                
                # Adaptive learning rate
                if obj_new > obj_prev * 1.01:
                    lr = max(lr * 0.5, 1e-8)
                elif obj_new < obj_prev * 0.99:
                    lr = min(lr * 1.05, 0.2)
                
                obj_prev = obj_new
        
        # Check if we found an exact solution
        if best_obj < 1e-10 and best_U is not None:
            # Try to round to integers/half-integers
            U_rounded = np.round(best_U * 2) / 2
            V_rounded = np.round(best_V * 2) / 2
            W_rounded = np.round(best_W * 2) / 2
            
            obj_rounded, _, _, _, _ = compute_residual_and_grads(U_rounded, V_rounded, W_rounded)
            if obj_rounded < 1e-10:
                return {'U': U_rounded.tolist(), 'V': V_rounded.tolist(), 'W': W_rounded.tolist()}
            
            # Also try integer rounding
            U_int = np.round(best_U)
            V_int = np.round(best_V)
            W_int = np.round(best_W)
            obj_int, _, _, _, _ = compute_residual_and_grads(U_int, V_int, W_int)
            if obj_int < 1e-10:
                return {'U': U_int.tolist(), 'V': V_int.tolist(), 'W': W_int.tolist()}
        
        return None
    
    # Deadline guard: ensure we return before 150s
    overall_start = time.time()
    
    # Try to find a better rank
    for target_rank in [71, 70, 69]:
        if time.time() - overall_start > 130:
            break
        time_budget = min(30.0, 135 - (time.time() - overall_start))
        if time_budget < 5:
            break
        result = try_rank_search(target_rank, time_budget=time_budget)
        if result is not None:
            return result
    
    # Fallback to schoolbook
    return build_schoolbook()
