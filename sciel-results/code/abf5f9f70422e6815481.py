import numpy as np
import time
import itertools
from scipy.optimize import minimize

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # Target rank is minimal. Start from lower bound.
    # Lower bound: (nm + mp + pn - 2) / 2 = 18
    start_rank = 18
    max_rank = 25 # Try up to 25
    
    deadline = time.time() + 110 # 10s margin
    
    # Precompute indices for non-zero entries
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    nonzero_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                nonzero_indices.append((u_idx, v_idx, w_idx))
    
    # Total number of equations is (n*m) * (m*p) * (p*n)
    # But we only need to check the non-zero ones for the '1' constraint.
    # The '0' constraints are the rest.
    # However, solving 1600 equations is heavy.
    # We can define the objective as sum of squares of (sum - target) for ALL entries.
    
    def get_error(params, R):
        # params is flattened U, V, W
        u_size = n * m * R
        v_size = m * p * R
        w_size = p * n * R
        U = params[:u_size].reshape(n*m, R)
        V = params[u_size:u_size+v_size].reshape(m*p, R)
        W = params[u_size+v_size:].reshape(p*n, R)
        
        error = 0.0
        # Check non-zero entries (should be 1)
        for u_idx, v_idx, w_idx in nonzero_indices:
            val = np.dot(U[u_idx], V[v_idx] * W[w_idx]) # No, sum over r
            # val = sum_r U[u_idx, r] * V[v_idx, r] * W[w_idx, r]
            # Efficient: np.einsum or loop
            # Since R is small, loop is fine
            s = 0
            for r in range(R):
                s += U[u_idx, r] * V[v_idx, r] * W[w_idx, r]
            error += (s - 1.0) ** 2
        
        # Check zero entries (should be 0)
        # This is expensive. We can sample or do full check.
        # Full check: 1600 entries.
        # 1600 * R operations.
        # For R=20, 32000 ops. Very fast.
        for u_idx in range(n*m):
            for v_idx in range(m*p):
                for w_idx in range(p*n):
                    if (u_idx, v_idx, w_idx) in nonzero_indices:
                        continue
                    s = 0
                    for r in range(R):
                        s += U[u_idx, r] * V[v_idx, r] * W[w_idx, r]
                    error += s * s
        return error

    # Better: use vectorized operations for the zero check?
    # U (nm, R), V (mp, R), W (pn, R).
    # Tensor T (nm, mp, pn) = sum_r U[:, r] outer V[:, r] outer W[:, r].
    # We can build the tensor and check.
    
    def build_tensor(U, V, W, R):
        T = np.zeros((n*m, m*p, p*n))
        for r in range(R):
            T += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
        return T

    for R in range(start_rank, max_rank + 1):
        if time.time() > deadline:
            break
            
        best_solution = None
        best_error = float('inf')
        
        # Try a few random initializations
        num_restarts = 5 if R < 20 else 2
        
        for _ in range(num_restarts):
            if time.time() > deadline:
                break
                
            # Random initialization
            # Scale matters. 
            # For Strassen-like, entries are small integers.
            # Let's try random integers in [-2, 2] or similar?
            # Or random floats.
            params0 = np.random.randn(n*m*R + m*p*R + p*n*R) * 1.0
            
            # Objective function for optimizer
            def obj(params):
                u_size = n * m * R
                v_size = m * p * R
                w_size = p * n * R
                U = params[:u_size].reshape(n*m, R)
                V = params[u_size:u_size+v_size].reshape(m*p, R)
                W = params[u_size+v_size:].reshape(p*n, R)
                
                T = build_tensor(U, V, W, R)
                
                # Target tensor
                target = np.zeros((n*m, m*p, p*n))
                for u_idx, v_idx, w_idx in nonzero_indices:
                    target[u_idx, v_idx, w_idx] = 1.0
                
                error = np.sum((T - target) ** 2)
                return error
            
            res = minimize(obj, params0, method='Nelder-Mead', options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8})
            
            if res.fun < best_error:
                best_error = res.fun
                best_solution = res.x
                
            if best_error < 1e-10:
                break
        
        if best_solution is not None and best_error < 1e-8:
            # Try to round to half-integers
            u_size = n * m * R
            v_size = m * p * R
            w_size = p * n * R
            
            U = best_solution[:u_size].reshape(n*m, R)
            V = best_solution[u_size:u_size+v_size].reshape(m*p, R)
            W = best_solution[u_size+v_size:].reshape(p*n, R)
            
            # Round to nearest 0.5
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2
            
            # Verify exactness
            T_r = build_tensor(U_r, V_r, W_r, R)
            target = np.zeros((n*m, m*p, p*n))
            for u_idx, v_idx, w_idx in nonzero_indices:
                target[u_idx, v_idx, w_idx] = 1.0
            
            if np.allclose(T_r, target, atol=1e-9):
                # Found a valid half-integer decomposition
                # Convert to lists of floats (or ints if integer)
                U_list = U_r.tolist()
                V_list = V_r.tolist()
                W_list = W_r.tolist()
                
                # Clean up values like 1.0 to 1, 0.5 to 0.5
                def clean_val(v):
                    if abs(v - round(v)) < 1e-9:
                        return int(round(v))
                    elif abs(v - round(v*2)/2) < 1e-9:
                        return v
                    return v
                
                U_clean = [[clean_val(v) for v in row] for row in U_list]
                V_clean = [[clean_val(v) for v in row] for row in V_list]
                W_clean = [[clean_val(v) for v in row] for row in W_list]
                
                return {"U": U_clean, "V": V_clean, "W": W_clean}
        
        # If no solution found for this R, try next R

    # Fallback: Schoolbook
    rank = n * m * p
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
