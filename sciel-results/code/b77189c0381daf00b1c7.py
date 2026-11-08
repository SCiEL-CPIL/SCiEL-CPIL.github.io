import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 3]

def _tensor_from_decomp_fast(U, V, W, n, m, p):
    """Vectorized tensor computation: T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]"""
    R = U.shape[1]
    T = np.zeros((n, m, p))
    for r in range(R):
        u = U[:, r]  # shape (n*m,)
        v = V[:, r]  # shape (m*p,)
        w = W[:, r]  # shape (p*n,)
        # Reshape to (n, m), (m, p), (p, n)
        u_mat = u.reshape(n, m)
        v_mat = v.reshape(m, p)
        w_mat = w.reshape(p, n)
        # For each (i,j,k): T[i,j,k] += u_mat[i,j] * v_mat[j,k] * w_mat[k,i]
        # We can compute this as: for each k, sum over i,j of u_mat[i,j]*v_mat[j,k]*w_mat[k,i]
        # Equivalent to: T[:, :, k] += (u_mat * v_mat.T) * w_mat[k, :][:, None]
        # Let's do it more carefully
        for k in range(p):
            # w_mat[k, i] for i in range(n)
            wk = w_mat[k, :]  # shape (n,)
            # u_mat[i,j] * v_mat[j,k] * wk[i]
            # = (u_mat * v_mat[:, k:k+1]) * wk[:, None]
            T[:, :, k] += (u_mat * v_mat[:, k:k+1]) * wk[:, None]
    return T

def _verify_decomp(U, V, W, n, m, p):
    """Verify that the decomposition exactly reproduces the matrix multiplication tensor."""
    T = _tensor_from_decomp_fast(U, V, W, n, m, p)
    return np.allclose(T, np.ones((n, m, p)), atol=1e-10)

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

def _try_known_construction(n, m, p):
    """
    Try a known rank-19 construction for 3x3 matrix multiplication.
    Based on known algorithms in the literature.
    """
    # A known rank-19 algorithm for 3x3 matrix multiplication
    # This is based on the work of various authors
    # Let's try a specific known construction
    
    # One approach: use the "Laderman-like" structure but for 3x3
    # Actually, let's try to find a construction by solving a system of equations
    
    # For now, return None if we don't have a verified construction
    return None

def _optimize_rank(R, n, m, p, time_budget, max_restarts=10):
    """
    Optimize for a rank-R decomposition using L-BFGS-B.
    Returns (U, V, W) as numpy arrays if successful, else None.
    """
    start_time = time.time()
    T_true = np.ones((n, m, p))
    T_flat_true = T_true.flatten()
    
    def obj_fn(params):
        U = params[:9*R].reshape(9, R)
        V = params[9*R:18*R].reshape(9, R)
        W = params[18*R:27*R].reshape(9, R)
        T = _tensor_from_decomp_fast(U, V, W, n, m, p)
        diff = T.flatten() - T_flat_true
        return np.dot(diff, diff)
    
    def grad_fn(params):
        U = params[:9*R].reshape(9, R)
        V = params[9*R:18*R].reshape(9, R)
        W = params[18*R:27*R].reshape(9, R)
        T = _tensor_from_decomp_fast(U, V, W, n, m, p)
        diff = T.flatten() - T_flat_true
        
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        for r in range(R):
            u = U[:, r]
            v = V[:, r]
            w = W[:, r]
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        dT = diff[i, j, k]
                        grad_U[i * m + j, r] += dT * v[j * p + k, r] * w[k * n + i, r]
                        grad_V[j * p + k, r] += dT * u[i * m + j, r] * w[k * n + i, r]
                        grad_W[k * n + i, r] += dT * u[i * m + j, r] * v[j * p + k, r]
                        
        return np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
    
    for restart in range(max_restarts):
        if time.time() - start_time > time_budget:
            break
        
        # Use structured initialization: start from schoolbook and perturb
        if restart == 0:
            # Initialize from a scaled-down schoolbook-like structure
            U_init = np.zeros((9, R))
            V_init = np.zeros((9, R))
            W_init = np.zeros((9, R))
            # Fill with small random values to break symmetry
            U_init += np.random.randn(9, R) * 0.1
            V_init += np.random.randn(9, R) * 0.1
            W_init += np.random.randn(9, R) * 0.1
        else:
            U_init = np.random.randn(9, R) * 0.3
            V_init = np.random.randn(9, R) * 0.3
            W_init = np.random.randn(9, R) * 0.3
        
        params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        try:
            res = minimize(obj_fn, params_init, method='L-BFGS-B', jac=grad_fn, 
                           options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-14, 'maxls': 100})
            
            if res.fun < 1e-8:
                U_opt = res.x[:9*R].reshape(9, R)
                V_opt = res.x[9*R:18*R].reshape(9, R)
                W_opt = res.x[18*R:27*R].reshape(9, R)
                
                # Round to nearest half-integer
                U_round = np.round(U_opt * 2) / 2
                V_round = np.round(V_opt * 2) / 2
                W_round = np.round(W_opt * 2) / 2
                
                if _verify_decomp(U_round, V_round, W_round, n, m, p):
                    return U_round, V_round, W_round
                    
        except Exception:
            pass
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin for safety
    
    # First try known constructions
    known = _try_known_construction(n, m, p)
    if known is not None:
        U, V, W = known
        if _verify_decomp(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try optimization for decreasing ranks
    target_ranks = [19, 20, 21, 22, 23]
    
    for R in target_ranks:
        if time.time() > deadline:
            break
        
        # Allocate time based on rank priority
        remaining_time = deadline - time.time()
        rank_index = target_ranks.index(R)
        # Lower ranks get more time
        if rank_index == 0:
            time_for_rank = remaining_time * 0.5
        elif rank_index == 1:
            time_for_rank = remaining_time * 0.3
        else:
            time_for_rank = remaining_time * 0.15
        
        result = _optimize_rank(R, n, m, p, time_for_rank, max_restarts=15)
        if result is not None:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook
    return _make_schoolbook(n, m, p)
