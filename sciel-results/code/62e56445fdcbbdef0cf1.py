import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for matrix multiplication 3x3 * 3x7.
    Uses numerical optimization to search for ranks lower than the schoolbook rank of 63.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # Safety margin

    # Pre-compute indices for the non-zero entries of the tensor
    # T[a, b, c] = 1 if a=(i,j), b=(j,k), c=(i,k)
    # a = i*m + j
    # b = j*p + k
    # c = k*n + i
    indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                indices.append((a, b, c))
    
    num_entries = len(indices) # 63

    def objective_and_gradient(x, r, indices):
        """
        x is a flat array of U, V, W concatenated.
        U: (n*m, r)
        V: (m*p, r)
        W: (p*n, r)
        """
        nm = n * m
        mp = m * p
        pn = p * n
        
        # Unpack
        U = x[:nm*r].reshape(nm, r)
        V = x[nm*r:nm*r+mp*r].reshape(mp, r)
        W = x[nm*r+mp*r:].reshape(pn, r)
        
        # Compute S[a, b, c] = sum_t U[a,t] V[b,t] W[c,t]
        # We only need to compute for the specific indices in `indices`
        # But for gradient, we need derivatives for all a, b, c?
        # No, only for the ones in indices.
        
        S = np.zeros(num_entries)
        # We can compute this efficiently by iterating over t?
        # Or just loop over entries. 63 entries is small.
        for idx, (a, b, c) in enumerate(indices):
            S[idx] = np.dot(U[a, :], np.multiply(V[b, :], W[c, :]))
            
        # Error
        diff = 1.0 - S
        err = np.sum(diff**2)
        
        # Regularization
        reg = 1e-6 * (np.sum(U**2) + np.sum(V**2) + np.sum(W**2))
        total_err = err + reg
        
        # Gradient
        # d(err)/d(U[a,t]) = -2 * sum_{(i,j,k) in indices s.t. a=i*m+j} diff[idx] * V[b,t] * W[c,t]
        # Note: a determines i, j. So for a fixed a, we sum over k.
        
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        for idx, (a, b, c) in enumerate(indices):
            d = diff[idx]
            # Contribution to U[a, :]
            # grad_U[a, t] += -2 * d * V[b, t] * W[c, t]
            grad_U[a, :] -= 2.0 * d * (V[b, :] * W[c, :])
            # Contribution to V[b, :]
            # grad_V[b, t] += -2 * d * U[a, t] * W[c, t]
            grad_V[b, :] -= 2.0 * d * (U[a, :] * W[c, :])
            # Contribution to W[c, :]
            # grad_W[c, t] += -2 * d * U[a, t] * V[b, :]
            grad_W[c, :] -= 2.0 * d * (U[a, :] * V[b, :])
            
        # Add regularization gradient
        grad_U += 2e-6 * U
        grad_V += 2e-6 * V
        grad_W += 2e-6 * W
        
        grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
        
        return total_err, grad

    best_result = None
    best_rank = n * m * p # 63
    
    # Schoolbook solution as fallback
    U_school = np.zeros((n*m, best_rank))
    V_school = np.zeros((m*p, best_rank))
    W_school = np.zeros((p*n, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1.0
                V_school[j*p+k, col] = 1.0
                W_school[k*n+i, col] = 1.0
                col += 1
                
    # Try to find lower ranks
    # We iterate downwards from 62
    for r in range(best_rank - 1, 30, -1):
        if time.time() > deadline:
            break
            
        # Try a few random restarts for this rank
        found = False
        max_restarts = 5
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
                
            # Initialize
            # Scale: U, V, W should be small initially?
            # If we start with 0, we are stuck.
            # Start with small random noise.
            scale = 1.0
            U_init = np.random.randn(n*m, r) * scale
            V_init = np.random.randn(m*p, r) * scale
            W_init = np.random.randn(p*n, r) * scale
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                res = minimize(
                    objective_and_gradient, 
                    x0, 
                    args=(r, indices), 
                    method='L-BFGS-B', 
                    jac=True, 
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 20}
                )
                
                if res.success and res.fun < 1e-8:
                    # Verify
                    U_sol = res.x[:n*m*r].reshape(n*m, r)
                    V_sol = res.x[n*m*r:n*m*r+m*p*r].reshape(m*p, r)
                    W_sol = res.x[n*m*r+m*p*r:].reshape(p*n, r)
                    
                    # Check exactness
                    max_err = 0.0
                    for a, b, c in indices:
                        val = np.dot(U_sol[a, :], np.multiply(V_sol[b, :], W_sol[c, :]))
                        err = abs(1.0 - val)
                        if err > max_err:
                            max_err = err
                            
                    if max_err < 1e-6:
                        best_rank = r
                        best_result = {
                            "U": U_sol.tolist(),
                            "V": V_sol.tolist(),
                            "W": W_sol.tolist()
                        }
                        found = True
                        break # Move to next lower rank
                        
            except Exception:
                pass

        if not found:
            # If we failed to find a solution for r, it might be impossible or hard.
            # We can continue to r-1, but it's likely if r is impossible, r-1 is too.
            # However, sometimes a lower rank is easier to find? Unlikely.
            # Let's just continue.
            pass

    if best_result is not None:
        return best_result
    else:
        return {
            "U": U_school.tolist(),
            "V": V_school.tolist(),
            "W": W_school.tolist()
        }
# EVOLVE-BLOCK-END
