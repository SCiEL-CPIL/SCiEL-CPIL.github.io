import numpy as np
import time
import sys
from scipy.optimize import minimize
from fractions import Fraction

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    # Schoolbook decomposition as fallback
    U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    start_time = time.time()
    deadline = start_time + 135.0  # 135s search, 15s safety margin
    
    # Precompute target tensor
    TARGET_TENSOR = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                TARGET_TENSOR[i*m+j, j*p+k, k*n+i] = 1.0
    
    def try_rank(r):
        """Try to find a decomposition of rank r using L-BFGS-B optimization."""
        if time.time() > deadline:
            return None
        
        # Objective: minimize ||T - sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r]||^2
        # Using einsum for efficiency
        def objective_and_grad(params):
            if time.time() > deadline:
                return np.inf, np.zeros_like(params)
            
            U = params[:n*m*r].reshape(n*m, r)
            V = params[n*m*r:n*m*r + m*p*r].reshape(m*p, r)
            W = params[n*m*r + m*p*r:].reshape(p*n, r)
            
            # T_approx[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
            T_approx = np.einsum('ar,br,cr->abc', U, V, W)
            diff = T_approx - TARGET_TENSOR
            obj = np.sum(diff**2)
            
            # Gradient computation
            grad_W = 2 * np.einsum('abc,ar,br->cr', diff, U, V)
            grad_U = 2 * np.einsum('abc,br,cr->ar', diff, V, W)
            grad_V = 2 * np.einsum('abc,ar,cr->br', diff, U, W)
            
            grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
            
            return obj, grad
        
        # Try several random initializations with different strategies
        for attempt in range(20):
            if time.time() > deadline - 5:
                break
            
            np.random.seed(1000 + r * 100 + attempt)
            
            if attempt < 5:
                # Standard random initialization
                scale = 0.5
                U0 = np.random.randn(n*m, r) * scale
                V0 = np.random.randn(m*p, r) * scale
                W0 = np.random.randn(p*n, r) * scale
            elif attempt < 10:
                # Sparse initialization - fewer non-zeros
                scale = 1.0
                U0 = np.random.randn(n*m, r) * scale
                V0 = np.random.randn(m*p, r) * scale
                W0 = np.random.randn(p*n, r) * scale
                # Zero out ~50% of entries
                mask_U = np.random.rand(n*m, r) < 0.5
                mask_V = np.random.rand(m*p, r) < 0.5
                mask_W = np.random.rand(p*n, r) < 0.5
                U0[mask_U] = 0
                V0[mask_V] = 0
                W0[mask_W] = 0
            else:
                # Structured initialization based on schoolbook pattern
                U0 = np.zeros((n*m, r))
                V0 = np.zeros((m*p, r))
                W0 = np.zeros((p*n, r))
                # Distribute schoolbook terms across r ranks
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            rank_idx = (i * m + j) * p + k
                            rank_idx = rank_idx % r
                            U0[i*m+j, rank_idx] += 1
                            V0[j*p+k, rank_idx] += 1
                            W0[k*n+i, rank_idx] += 1
                # Add small random perturbation
                U0 += np.random.randn(n*m, r) * 0.1
                V0 += np.random.randn(m*p, r) * 0.1
                W0 += np.random.randn(p*n, r) * 0.1
            
            x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
            
            try:
                res = minimize(
                    objective_and_grad,
                    x0,
                    method='L-BFGS-B',
                    jac=True,
                    options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
                )
                
                if res.fun < 1e-8:
                    # Extract solution
                    U = res.x[:n*m*r].reshape(n*m, r)
                    V = res.x[n*m*r:n*m*r + m*p*r].reshape(m*p, r)
                    W = res.x[n*m*r + m*p*r:].reshape(p*n, r)
                    
                    # Try to recover rational values
                    rational_solution = recover_rational(U, V, W, r, n, m, p, TARGET_TENSOR)
                    if rational_solution is not None:
                        return rational_solution
                        
            except Exception as e:
                continue
        
        return None
    
    def recover_rational(U, V, W, r, n, m, p, T_target):
        """Try to recover integer/half-integer values from float solution."""
        if time.time() > deadline:
            return None
        
        # Scale each matrix to try to get rational values
        all_vals = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        nonzero = all_vals[np.abs(all_vals) > 1e-10]
        
        if len(nonzero) == 0:
            return None
        
        # Try different scaling factors and rounding strategies
        for scale_factor in [1, 2, 4, 8, 16, 32, 64]:
            U_s = U * scale_factor
            V_s = V * scale_factor
            W_s = W * scale_factor
            
            # Round to nearest half-integer
            U_r = np.rint(U_s * 2) / 2
            V_r = np.rint(V_s * 2) / 2
            W_r = np.rint(W_s * 2) / 2
            
            # Verify
            T_check = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
            if np.allclose(T_check, T_target, atol=1e-8):
                # Verify all entries are integer or half-integer
                valid = True
                for mat in [U_r, V_r, W_r]:
                    doubled = mat * 2
                    if not np.allclose(doubled, np.rint(doubled), atol=1e-10):
                        valid = False
                        break
                if valid:
                    return {
                        "U": U_r.tolist(),
                        "V": V_r.tolist(),
                        "W": W_r.tolist()
                    }
        
        # Also try without scaling
        U_r = np.rint(U * 2) / 2
        V_r = np.rint(V * 2) / 2
        W_r = np.rint(W * 2) / 2
        
        T_check = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
        if np.allclose(T_check, T_target, atol=1e-8):
            valid = True
            for mat in [U_r, V_r, W_r]:
                doubled = mat * 2
                if not np.allclose(doubled, np.rint(doubled), atol=1e-10):
                    valid = False
                    break
            if valid:
                return {
                    "U": U_r.tolist(),
                    "V": V_r.tolist(),
                    "W": W_r.tolist()
                }
        
        # Try integer-only rounding
        for scale_factor in [1, 2, 4, 8]:
            U_s = U * scale_factor
            V_s = V * scale_factor
            W_s = W * scale_factor
            
            U_r = np.rint(U_s)
            V_r = np.rint(V_s)
            W_r = np.rint(W_s)
            
            T_check = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
            if np.allclose(T_check, T_target, atol=1e-8):
                valid = True
                for mat in [U_r, V_r, W_r]:
                    if not np.allclose(mat, np.rint(mat), atol=1e-10):
                        valid = False
                        break
                if valid:
                    return {
                        "U": U_r.tolist(),
                        "V": V_r.tolist(),
                        "W": W_r.tolist()
                    }
        
        return None
    
    # Try ranks from 34 down to 28
    for r in range(34, 27, -1):
        if time.time() > deadline:
            break
        
        result = try_rank(r)
        if result is not None:
            return result
    
    # Fallback
    return {"U": U_sb, "V": V_sb, "W": W_sb}
