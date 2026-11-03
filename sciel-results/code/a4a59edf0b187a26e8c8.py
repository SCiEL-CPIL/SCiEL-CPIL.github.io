"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    # Tensor dimensions
    dim_u = n * m  # 12
    dim_v = m * p  # 20
    dim_w = p * n  # 15
    
    # Precompute index arrays for vectorized evaluation
    # For each (i,j,k) triple, we need U[i*m+j, :], V[j*p+k, :], W[k*n+i, :]
    i_idx = np.repeat(np.repeat(np.arange(n), m), p)  # shape (n*m*p,)
    j_idx = np.repeat(np.arange(m), p)
    j_idx = np.tile(j_idx, n)
    k_idx = np.tile(np.arange(p), n * m)
    
    u_idx = i_idx * m + j_idx  # shape (n*m*p,)
    v_idx = j_idx * p + k_idx  # shape (n*m*p,)
    w_idx = k_idx * n + i_idx  # shape (n*m*p,)
    
    num_entries = n * m * p  # 60
    
    # Schoolbook as fallback
    rank_sb = n * m * p
    U_sb = np.zeros((dim_u, rank_sb))
    V_sb = np.zeros((dim_v, rank_sb))
    W_sb = np.zeros((dim_w, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    best_result = (U_sb, V_sb, W_sb)
    best_rank = rank_sb
    
    def compute_error(U, V, W):
        """Vectorized error computation: sum of (sum_r U_u[r]*V_v[r]*W_w[r] - 1)^2"""
        # U[u_idx, :] has shape (60, rank)
        # V[v_idx, :] has shape (60, rank)  
        # W[w_idx, :] has shape (60, rank)
        # Element-wise product: (60, rank), then sum over rank axis
        products = U[u_idx, :] * V[v_idx, :] * W[w_idx, :]
        sums = products.sum(axis=1)  # shape (60,)
        errors = sums - 1.0
        return np.sum(errors ** 2)
    
    def compute_gradient(U, V, W):
        """Compute gradient of error w.r.t. U, V, W"""
        products = U[u_idx, :] * V[v_idx, :] * W[w_idx, :]
        sums = products.sum(axis=1)
        errors = sums - 1.0  # shape (60,)
        
        # d(error)/d(U[u, r]) = 2 * error[u] * V[v, r] * W[w, r]
        # Need to accumulate over all entries that use each (u, r) pair
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        # For each entry e, the contribution to grad_U[u_idx[e], r] is 2*errors[e]*V[v_idx[e],r]*W[w_idx[e],r]
        # Vectorized: for each r, grad_U[:, r] = sum over e where u_idx[e] == u of 2*errors[e]*V[v_idx[e],r]*W[w_idx[e],r]
        
        # Build contribution matrices
        # contrib_U[e, r] = 2 * errors[e] * V[v_idx[e], r] * W[w_idx[e], r]
        contrib_U = 2.0 * errors[:, np.newaxis] * V[v_idx, :] * W[w_idx, :]  # (60, rank)
        contrib_V = 2.0 * errors[:, np.newaxis] * U[u_idx, :] * W[w_idx, :]  # (60, rank)
        contrib_W = 2.0 * errors[:, np.newaxis] * U[u_idx, :] * V[v_idx, :]  # (60, rank)
        
        # Scatter-add: for each unique u value, sum contributions
        for r in range(U.shape[1]):
            np.add.at(grad_U, u_idx, contrib_U[:, r])
            np.add.at(grad_V, v_idx, contrib_V[:, r])
            np.add.at(grad_W, w_idx, contrib_W[:, r])
        
        return grad_U, grad_V, grad_W
    
    def objective_and_grad(x):
        """Objective and gradient for L-BFGS"""
        U = x[:dim_u * rank].reshape(dim_u, rank)
        V = x[dim_u * rank:dim_u * rank + dim_v * rank].reshape(dim_v, rank)
        W = x[dim_u * rank + dim_v * rank:].reshape(dim_w, rank)
        
        err = compute_error(U, V, W)
        gU, gV, gW = compute_gradient(U, V, W)
        grad = np.concatenate([gU.flatten(), gV.flatten(), gW.flatten()])
        
        return err, grad
    
    # Try to find lower rank decompositions
    for rank in range(20, rank_sb + 1):
        if time.time() > deadline:
            break
        
        found = False
        
        # Try multiple random initializations
        for attempt in range(8):
            if time.time() > deadline:
                break
            
            if attempt == 0:
                np.random.seed(rank * 100 + attempt)
                U_init = np.random.randn(dim_u, rank)
                V_init = np.random.randn(dim_v, rank)
                W_init = np.random.randn(dim_w, rank)
            elif attempt == 1:
                np.random.seed(rank * 100 + attempt + 10)
                U_init = np.random.randn(dim_u, rank) * 0.5
                V_init = np.random.randn(dim_v, rank) * 0.5
                W_init = np.random.randn(dim_w, rank) * 0.5
            elif attempt == 2:
                np.random.seed(rank * 100 + attempt + 20)
                U_init = np.zeros((dim_u, rank))
                V_init = np.zeros((dim_v, rank))
                W_init = np.zeros((dim_w, rank))
                for r in range(rank):
                    u_support = np.random.choice(dim_u, size=max(2, dim_u // 3), replace=False)
                    v_support = np.random.choice(dim_v, size=max(2, dim_v // 3), replace=False)
                    w_support = np.random.choice(dim_w, size=max(2, dim_w // 3), replace=False)
                    U_init[u_support, r] = np.random.choice([-1, 1], size=len(u_support))
                    V_init[v_support, r] = np.random.choice([-1, 1], size=len(v_support))
                    W_init[w_support, r] = np.random.choice([-1, 1], size=len(w_support))
            elif attempt == 3:
                np.random.seed(rank * 100 + attempt + 30)
                # Structured: try to build from schoolbook-like patterns
                U_init = np.zeros((dim_u, rank))
                V_init = np.zeros((dim_v, rank))
                W_init = np.zeros((dim_w, rank))
                for r in range(rank):
                    i = r % n
                    j = (r // n) % m
                    k = (r // (n * m)) % p
                    U_init[i * m + j, r] = 1
                    V_init[j * p + k, r] = 1
                    W_init[k * n + i, r] = 1
                    # Add some random perturbation
                    if r < rank - 1:
                        U_init[i * m + (j + 1) % m, r] = 0.5
                        V_init[(j + 1) % m * p + k, r] = 0.5
                        W_init[k * n + (i + 1) % n, r] = 0.5
            else:
                np.random.seed(rank * 100 + attempt + 40)
                U_init = np.random.randn(dim_u, rank)
                V_init = np.random.randn(dim_v, rank)
                W_init = np.random.randn(dim_w, rank)
                # Normalize columns
                for r in range(rank):
                    U_init[:, r] /= np.linalg.norm(U_init[:, r]) + 1e-10
                    V_init[:, r] /= np.linalg.norm(V_init[:, r]) + 1e-10
                    W_init[:, r] /= np.linalg.norm(W_init[:, r]) + 1e-10
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            # Check initial error
            init_err = compute_error(U_init, V_init, W_init)
            
            # Use L-BFGS-B for gradient-based optimization
            try:
                result = minimize(objective_and_grad, x0, method='L-BFGS-B',
                                  jac=True,  # We provide gradient
                                  options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10,
                                           'maxls': 50, 'maxcor': 10})
                
                if result.fun < 1e-10:
                    U_opt = result.x[:dim_u * rank].reshape(dim_u, rank)
                    V_opt = result.x[dim_u * rank:dim_u * rank + dim_v * rank].reshape(dim_v, rank)
                    W_opt = result.x[dim_u * rank + dim_v * rank:].reshape(dim_w, rank)
                    
                    # Try rounding to half-integers
                    U_rounded = np.round(U_opt * 2) / 2
                    V_rounded = np.round(V_opt * 2) / 2
                    W_rounded = np.round(W_opt * 2) / 2
                    
                    # Verify rounded
                    max_err = compute_error(U_rounded, V_rounded, W_rounded)
                    if max_err < 1e-9:
                        best_result = (U_rounded, V_rounded, W_rounded)
                        best_rank = rank
                        found = True
                        break
                    
                    # Try rounding to integers
                    U_int = np.round(U_opt)
                    V_int = np.round(V_opt)
                    W_int = np.round(W_opt)
                    max_err_int = compute_error(U_int, V_int, W_int)
                    if max_err_int < 1e-9:
                        best_result = (U_int, V_int, W_int)
                        best_rank = rank
                        found = True
                        break
                        
            except Exception:
                pass
        
        if found:
            break
    
    # Convert to lists
    U_list = best_result[0].tolist()
    V_list = best_result[1].tolist()
    W_list = best_result[2].tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
