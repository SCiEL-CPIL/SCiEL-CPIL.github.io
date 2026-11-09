import numpy as np
import time
import sys
from scipy.optimize import minimize

DIMENSIONS = [3, 4, 5]

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x4 * 4x5 matrix multiplication.
    Uses scipy.optimize.minimize with analytical gradients to find low-rank decompositions.
    """
    start_time = time.time()
    deadline = 110  # Keep safety margin before 120s limit
    
    n, m, p = 3, 4, 5
    nm, mp, pn = n*m, m*p, p*n  # 12, 20, 15
    
    # Precompute indices for fast access
    # We need to map (i, j, k) to indices in U, V, W
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    
    # Flatten the tensor indices
    # There are n*m*p = 60 equations
    # Variables: r * (nm + mp + pn)
    
    def objective_and_gradient(theta, r):
        """
        theta: flattened vector of [U_flat, V_flat, W_flat]
        r: rank
        Returns (objective_value, gradient_vector)
        """
        # Unpack
        U = theta[0:nm*r].reshape(nm, r)
        V = theta[nm*r:nm*r+mp*r].reshape(mp, r)
        W = theta[nm*r+mp*r:nm*r+mp*r+pn*r].reshape(pn, r)
        
        # Compute S[i, j, k] = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
        # We can vectorize this calculation.
        # S is a tensor of shape (n, m, p)
        
        # Efficient calculation:
        # For each c, contribution is outer product of U[:,c], V[:,c], W[:,c]
        # But we need to sum them up with specific index mappings.
        
        # Let's build the tensor S directly.
        # S[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        
        S = np.zeros((n, m, p))
        
        # Loop over rank c for clarity, might be slow if r is large, but r ~ 50 is small.
        # 50 iterations is fast.
        for c in range(r):
            u_c = U[:, c] # shape (nm,)
            v_c = V[:, c] # shape (mp,)
            w_c = W[:, c] # shape (pn,)
            
            # Reshape u_c to (n, m), v_c to (m, p), w_c to (p, n)
            u_reshaped = u_c.reshape(n, m)
            v_reshaped = v_c.reshape(m, p)
            w_reshaped = w_c.reshape(p, n)
            
            # S[i,j,k] += u_reshaped[i,j] * v_reshaped[j,k] * w_reshaped[k,i]
            # This is an einsum: 'ij,jk,ki->ijk'
            S += np.einsum('ij,jk,ki->ijk', u_reshaped, v_reshaped, w_reshaped)
        
        # Objective: sum (S - 1)^2
        diff = S - 1.0
        obj = np.sum(diff**2)
        
        # Gradient
        # d_obj / d_U[i*m+j, c] = 2 * (S[i,j,k] - 1) * V[j*p+k, c] * W[k*n+i, c]
        # Note: For a fixed (i,j), S[i,j,k] depends on k.
        # The gradient for U[i*m+j, c] is sum_k 2 * (S[i,j,k] - 1) * V[j*p+k, c] * W[k*n+i, c]
        
        grad_U = np.zeros((nm, r))
        grad_V = np.zeros((mp, r))
        grad_W = np.zeros((pn, r))
        
        for c in range(r):
            u_c = U[:, c]
            v_c = V[:, c]
            w_c = W[:, c]
            
            # Gradient for U
            # grad_U[i*m+j, c] = 2 * sum_k (S[i,j,k] - 1) * v_c[j*p+k] * w_c[k*n+i]
            # Let's compute this via einsum or loops.
            # v_c reshaped (m, p), w_c reshaped (p, n)
            v_reshaped = v_c.reshape(m, p)
            w_reshaped = w_c.reshape(p, n)
            
            # diff reshaped (n, m, p)
            # grad_U[i, j] = 2 * sum_k diff[i, j, k] * v_reshaped[j, k] * w_reshaped[k, i]
            # This is 'ijk,jk,ki->ij'
            grad_U_c = 2.0 * np.einsum('ijk,jk,ki->ij', diff, v_reshaped, w_reshaped)
            grad_U[:, c] = grad_U_c.flatten()
            
            # Gradient for V
            # grad_V[j*p+k, c] = 2 * sum_i (S[i,j,k] - 1) * U[i*m+j, c] * W[k*n+i, c]
            # u_reshaped (n, m), w_reshaped (p, n)
            u_reshaped = u_c.reshape(n, m)
            # 'ijk,ij,ki->jk'
            grad_V_c = 2.0 * np.einsum('ijk,ij,ki->jk', diff, u_reshaped, w_reshaped)
            grad_V[:, c] = grad_V_c.flatten()
            
            # Gradient for W
            # grad_W[k*n+i, c] = 2 * sum_j (S[i,j,k] - 1) * U[i*m+j, c] * V[j*p+k, c]
            # u_reshaped (n, m), v_reshaped (m, p)
            # 'ijk,ij,jk->ki'
            grad_W_c = 2.0 * np.einsum('ijk,ij,jk->ki', diff, u_reshaped, v_reshaped)
            grad_W[:, c] = grad_W_c.flatten()
            
        grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
        
        return obj, grad

    def check_solution(U, V, W, rank):
        """Compute max absolute error."""
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i*m + j
                    v_idx = j*p + k
                    w_idx = k*n + i
                    val = np.dot(U[u_idx, :rank], V[v_idx, :rank] * W[w_idx, :rank])
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err

    def solve_for_rank(r, seed):
        """
        Attempt to find a decomposition of rank r.
        Returns (error, U, V, W) or (inf, None, None, None) if failed.
        """
        rng = np.random.RandomState(seed)
        
        # Initialization
        # Scale matters. Let's start with small random values.
        U_init = rng.randn(nm, r) * 0.5
        V_init = rng.randn(mp, r) * 0.5
        W_init = rng.randn(pn, r) * 0.5
        
        theta0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        options = {
            'maxiter': 1000,
            'ftol': 1e-12,
            'gtol': 1e-8,
            'maxls': 50
        }
        
        try:
            res = minimize(
                objective_and_gradient,
                theta0,
                args=(r,),
                jac=True,
                method='L-BFGS-B',
                options=options
            )
            
            if res.success or res.fun < 1e-10:
                theta_opt = res.x
                U_opt = theta_opt[0:nm*r].reshape(nm, r)
                V_opt = theta_opt[nm*r:nm*r+mp*r].reshape(mp, r)
                W_opt = theta_opt[nm*r+mp*r:nm*r+mp*r+pn*r].reshape(pn, r)
                
                # Verify
                err = check_solution(U_opt, V_opt, W_opt, r)
                if err < 1e-8:
                    return err, U_opt, V_opt, W_opt
                
        except Exception as e:
            pass
            
        return float('inf'), None, None, None

    # Schoolbook baseline
    rank_schoolbook = n * m * p
    U_sb = np.zeros((nm, rank_schoolbook))
    V_sb = np.zeros((mp, rank_schoolbook))
    W_sb = np.zeros((pn, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1.0
                V_sb[j*p+k, col] = 1.0
                W_sb[k*n+i, col] = 1.0
                col += 1
    
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # Try ranks from low to high
    # Known best for 3x4x5 is 52. We will try 50, 51, 52.
    # If 52 works, we stop. If not, we try higher? No, if 52 fails, 53 might be easier?
    # Usually, if a rank is possible, higher ranks are also possible (by adding zero columns).
    # So if we can't find 52, we should try 53, 54... up to 59.
    # However, the landscape might be harder for lower ranks.
    # Let's try 52 first as it's a known bound. Then 50, 51. Then 53-59.
    
    ranks_to_try = [52, 50, 51, 53, 54, 55, 56, 57, 58, 59]
    
    for rank in ranks_to_try:
        if time.time() - start_time > deadline - 5:
            break
            
        found = False
        # Try multiple seeds
        for seed in [42, 123, 999, 2024, 555, 777]:
            if time.time() - start_time > deadline - 2:
                break
                
            err, U_c, V_c, W_c = solve_for_rank(rank, seed)
            
            if err < 1e-8:
                best_rank = rank
                best_U, best_V, best_W = U_c, V_c, W_c
                found = True
                break
        
        if found:
            # If we found a solution, we might want to try lower ranks?
            # We are iterating [52, 50, 51...]. If 52 works, we break.
            # But maybe 50 works?
            # The list is ordered to prioritize likely candidates.
            # If 52 works, it's a good improvement over 60.
            # We can break to save time, or continue to see if 50 works.
            # Given the time budget, breaking on first success for a rank < 60 is safe.
            # However, if 52 is found, 50 is better.
            # Let's reorder: try 50, 51, 52 first?
            # 50 is lower bound? No.
            # Let's just break. 52 is a significant improvement.
            break
            
    # Final verification
    if best_rank < rank_schoolbook:
        final_err = check_solution(best_U, best_V, best_W, best_rank)
        if final_err > 1e-8:
            # Fallback
            best_rank = rank_schoolbook
            best_U, best_V, best_W = U_sb, V_sb, W_sb

    def to_list(mat):
        return [[float(x) for x in row] for row in mat]
    
    U_out = to_list(best_U[:, :best_rank])
    V_out = to_list(best_V[:, :best_rank])
    W_out = to_list(best_W[:, :best_rank])
    
    return {"U": U_out, "V": V_out, "W": W_out}
