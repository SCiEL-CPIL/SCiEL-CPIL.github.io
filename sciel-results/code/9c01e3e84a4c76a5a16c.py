import numpy as np
import time
import sys

# DIMENSIONS
n, m, p = 3, 3, 5

def construct():
    start_time = time.time()
    deadline = 120.0  # seconds
    
    # Schoolbook rank
    rank_schoolbook = n * m * p
    
    # Precompute the target tensor T
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                I = i * m + j
                J = j * p + k
                K = k * n + i
                T[I, J, K] = 1.0

    def compute_residual(U, V, W):
        R = np.einsum('ic,jc,kc->ijk', U, V, W)
        R -= T
        return R

    def smooth_penalty(X):
        """Smooth penalty that encourages X to be near half-integers."""
        return np.sum(np.sin(np.pi * 2 * X)**2)

    def objective_and_grad(params, r, penalty_weight=0.3):
        size_U = n * m * r
        size_V = m * p * r
        size_W = p * n * r
        
        U = params[:size_U].reshape(n*m, r)
        V = params[size_U:size_U+size_V].reshape(m*p, r)
        W = params[size_U+size_V:].reshape(p*n, r)
        
        R = compute_residual(U, V, W)
        loss = np.sum(R**2)
        
        # Add smooth penalty for non-half-integer values
        penalty = penalty_weight * (smooth_penalty(U) + 
                                    smooth_penalty(V) + 
                                    smooth_penalty(W))
        total_loss = loss + penalty
        
        # Gradient of loss
        grad_U = 2.0 * np.einsum('ijk,jc,kc->ic', R, V, W)
        grad_V = 2.0 * np.einsum('ijk,ic,kc->jc', R, U, W)
        grad_W = 2.0 * np.einsum('ijk,ic,jc->kc', R, U, V)
        
        # Gradient of penalty: d/dx [sin^2(pi*2x)] = 2*pi*sin(4*pi*x)
        pen_grad_U = penalty_weight * 2.0 * np.pi * np.sin(4.0 * np.pi * U)
        pen_grad_V = penalty_weight * 2.0 * np.pi * np.sin(4.0 * np.pi * V)
        pen_grad_W = penalty_weight * 2.0 * np.pi * np.sin(4.0 * np.pi * W)
        
        grad_U += pen_grad_U
        grad_V += pen_grad_V
        grad_W += pen_grad_W
        
        grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
        
        return total_loss, grad

    def check_exactness(U, V, W, tol=1e-9):
        """Check if U, V, W are close to half-integers and give exact decomposition"""
        U_rounded = np.round(U * 2) / 2
        V_rounded = np.round(V * 2) / 2
        W_rounded = np.round(W * 2) / 2
        
        if np.max(np.abs(U - U_rounded)) > 1e-5:
            return False, None, None, None
        if np.max(np.abs(V - V_rounded)) > 1e-5:
            return False, None, None, None
        if np.max(np.abs(W - W_rounded)) > 1e-5:
            return False, None, None, None
            
        R_rounded = compute_residual(U_rounded, V_rounded, W_rounded)
        if np.max(np.abs(R_rounded)) < tol:
            return True, U_rounded, V_rounded, W_rounded
        return False, None, None, None

    def try_rank(r, num_restarts=8, max_iter=300):
        """Try to find a rank-r decomposition"""
        from scipy.optimize import minimize
        
        for restart in range(num_restarts):
            if time.time() > deadline - 15:
                return None
            
            np.random.seed(99999 + restart * 7777 + r * 13)
            params = np.random.randn(r * (n*m + m*p + p*n)) * 0.3
            
            try:
                res = minimize(
                    lambda x: objective_and_grad(x, r, penalty_weight=0.3)[0],
                    params,
                    jac=lambda x: objective_and_grad(x, r, penalty_weight=0.3)[1],
                    method='L-BFGS-B',
                    options={'gtol': 1e-8, 'maxiter': max_iter, 'maxls': 40}
                )
                
                if res.fun < 1e-5:
                    U, V, W = res.x[:n*m*r].reshape(n*m, r), \
                              res.x[n*m*r:n*m*r+m*p*r].reshape(m*p, r), \
                              res.x[n*m*r+m*p*r:].reshape(p*n, r)
                    
                    is_exact, U_r, V_r, W_r = check_exactness(U, V, W)
                    if is_exact:
                        return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                        
            except Exception:
                pass
        
        return None

    # Search for minimal rank
    for r in range(25, rank_schoolbook + 1):
        if time.time() > deadline - 20:
            break
            
        result = try_rank(r, num_restarts=6, max_iter=300)
        if result is not None:
            return result

    # If no low rank found, return schoolbook
    rank = rank_schoolbook
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, column] = 1
                V[j*p + k, column] = 1
                W[k*n + i, column] = 1
                column += 1
                
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
