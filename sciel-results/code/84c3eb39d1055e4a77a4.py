import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 3]

def _verify_decomp(U, V, W, n, m, p):
    """Verify that the decomposition exactly reproduces the matrix multiplication tensor."""
    # U, V, W are (n*m, R), (m*p, R), (p*n, R)
    # Reshape to (n, m, R), (m, p, R), (p, n, R)
    U3 = U.reshape(n, m, -1)
    V3 = V.reshape(m, p, -1)
    W3 = W.reshape(p, n, -1)
    
    # T[i, j, k] = sum_r U[i, j, r] V[j, k, r] W[k, i, r]
    # Note: W index is k*n + i, so W3[k, i, r]
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    
    return np.allclose(T, np.ones((n, m, p)), atol=1e-8)

def _make_schoolbook(n, m, p):
    """Create schoolbook decomposition (rank = n*m*p)."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _optimize_rank(R, n, m, p, time_budget, max_restarts=50):
    """
    Optimize for a rank-R decomposition using L-BFGS-B with vectorized operations.
    """
    start_time = time.time()
    T_true = np.ones((n, m, p))
    
    def obj_and_grad(params):
        U = params[:n*m*R].reshape(n, m, R)
        V = params[n*m*R:2*n*m*R].reshape(m, p, R)
        W = params[2*n*m*R:3*n*m*R].reshape(p, n, R)
        
        # Compute T
        T = np.einsum('ijr,jkr,kir->ijk', U, V, W)
        diff = T - T_true
        loss = np.dot(diff.flatten(), diff.flatten())
        
        # Compute Gradients
        # dL/dT[i,j,k] = 2 * diff[i,j,k]
        D = 2 * diff
        
        # grad_U[i,j,r] = sum_k D[i,j,k] * V[j,k,r] * W[k,i,r]
        grad_U = np.einsum('ijk,jkr,kir->ijr', D, V, W)
        # grad_V[j,k,r] = sum_i D[i,j,k] * U[i,j,r] * W[k,i,r]
        grad_V = np.einsum('ijk,ijr,kir->jkr', D, U, W)
        # grad_W[k,i,r] = sum_j D[i,j,k] * U[i,j,r] * V[j,k,r]
        grad_W = np.einsum('ijk,ijr,jkr->kir', D, U, V)
        
        grad_flat = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
        
        return loss, grad_flat

    per_restart_budget = time_budget / max_restarts
    
    for restart in range(max_restarts):
        if time.time() - start_time > time_budget:
            break
        
        # Random initialization
        U_init = np.random.randn(n, m, R) * 0.5
        V_init = np.random.randn(m, p, R) * 0.5
        W_init = np.random.randn(p, n, R) * 0.5
        
        params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        try:
            # Use L-BFGS-B
            res = minimize(obj_and_grad, params_init, method='L-BFGS-B', jac=True, 
                           options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12, 'maxls': 50})
            
            if res.fun < 1e-6:
                U_opt = res.x[:n*m*R].reshape(n, m, R)
                V_opt = res.x[n*m*R:2*n*m*R].reshape(m, p, R)
                W_opt = res.x[2*n*m*R:3*n*m*R].reshape(p, n, R)
                
                # Round to nearest half-integer
                U_round = np.round(U_opt * 2) / 2
                V_round = np.round(V_opt * 2) / 2
                W_round = np.round(W_opt * 2) / 2
                
                # Flatten for verification
                U_flat = U_round.flatten()
                V_flat = V_round.flatten()
                W_flat = W_round.flatten()
                
                if _verify_decomp(U_flat, V_flat, W_flat, n, m, p):
                    return U_flat, V_flat, W_flat
                    
        except Exception:
            pass
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    # Try optimization for decreasing ranks
    target_ranks = [19, 20, 21, 22, 23, 24, 25, 26]
    
    # Allocate time: more time for lower ranks
    # 19: 40%, 20: 20%, 21: 15%, rest: 5% each
    time_allocations = {
        19: 0.40,
        20: 0.20,
        21: 0.15,
        22: 0.08,
        23: 0.08,
        24: 0.04,
        25: 0.02,
        26: 0.02
    }
    
    for R in target_ranks:
        if time.time() > deadline:
            break
        
        remaining_time = deadline - time.time()
        # Scale allocation by remaining time to ensure we don't overshoot
        # If we have 140s left, and we want 40% of total budget (140), that's 56s.
        # But if we are late, we adjust.
        # Simple heuristic: use a fixed fraction of the *initial* budget, capped by remaining time.
        initial_budget = 140
        target_time = initial_budget * time_allocations.get(R, 0.01)
        time_for_rank = min(target_time, remaining_time)
        
        if time_for_rank < 1.0:
            continue
            
        result = _optimize_rank(R, n, m, p, time_for_rank, max_restarts=100)
        if result is not None:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook
    return _make_schoolbook(n, m, p)
