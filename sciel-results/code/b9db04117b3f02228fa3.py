"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 3, 3, 6
    
    # Build the target multiplication tensor
    # T[a, b, c] where a = i*m+j, b = j*p+k, c = k*n+i
    # For matrix mult A(n,m) * B(m,p) = C(n,p): C[i,k] = sum_j A[i,j]B[j,k]
    # The tensor is T[i*m+j, j*p+k, k*n+i] = 1 (for valid i,j,k)
    # Note: output index c = k*n + i (trace convention W[k*n+i])
    
    T_target = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0
    
    def check_decomposition(U, V, W, tol=1e-9):
        """Check if U ⊗ V ⊗ W reconstructs the multiplication tensor exactly."""
        if isinstance(U, list):
            U_arr = np.array(U, dtype=float)
            V_arr = np.array(V, dtype=float)
            W_arr = np.array(W, dtype=float)
        else:
            U_arr = U.astype(float)
            V_arr = V.astype(float)
            W_arr = W.astype(float)
        
        r = U_arr.shape[1]
        T_recon = np.zeros((n*m, m*p, p*n))
        for t in range(r):
            T_recon += np.einsum('a,b,c->abc', U_arr[:,t], V_arr[:,t], W_arr[:,t])
        
        if np.allclose(T_recon, T_target, atol=tol, rtol=0):
            return True
        return False
    
    def schoolbook_decomposition():
        """Return the standard rank-54 schoolbook decomposition."""
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
        return U, V, W, rank
    
    # Start with schoolbook as baseline
    U_school, V_school, W_school, school_rank = schoolbook_decomposition()
    best_U, best_V, best_W = U_school.copy(), V_school.copy(), W_school.copy()
    best_rank = school_rank
    
    def objective_and_gradient(x, target_rank):
        """Compute objective and gradient for L-BFGS-B optimization."""
        U_flat = x[:n*m*target_rank].reshape(n*m, target_rank)
        V_flat = x[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
        W_flat = x[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
        
        T_recon = np.zeros((n*m, m*p, p*n))
        for t in range(target_rank):
            T_recon += np.einsum('a,b,c->abc', U_flat[:,t], V_flat[:,t], W_flat[:,t])
        
        error = T_target - T_recon
        obj_val = np.sum(error**2)
        
        dU = np.zeros_like(U_flat)
        dV = np.zeros_like(V_flat)
        dW = np.zeros_like(W_flat)
        for t in range(target_rank):
            dU[:,t] = np.einsum('abc,bc->a', error, V_flat[:,t] * W_flat[:,t])
            dV[:,t] = np.einsum('abc,ac->b', error, U_flat[:,t] * W_flat[:,t])
            dW[:,t] = np.einsum('abc,ab->c', error, U_flat[:,t] * V_flat[:,t])
        
        grad = np.concatenate([-2*dU.flatten(), -2*dV.flatten(), -2*dW.flatten()])
        return obj_val, grad
    
    def try_l_bfgs(target_rank, num_trials=3, max_iter=300):
        """Try to find a rank-target_rank decomposition using L-BFGS-B."""
        if target_rank >= best_rank:
            return None
        
        from scipy.optimize import minimize
        
        for trial in range(num_trials):
            if time.time() > deadline - 5:
                return None
            
            np.random.seed(trial * 7919 + 13)
            
            # Structured initialization exploiting block structure
            U_t = np.zeros((n*m, target_rank))
            V_t = np.zeros((m*p, target_rank))
            W_t = np.zeros((p*n, target_rank))
            
            # Initialize with a mix of schoolbook-like structure and random perturbation
            # For the first 27 columns, use a pattern that respects the block structure
            for col in range(min(27, target_rank)):
                i = col % n
                j = (col // n) % m
                k = (col // (n*m)) % p
                U_t[i*m+j, col] = 1.0
                V_t[j*p+k, col] = 1.0
                W_t[k*n+i, col] = 1.0
            
            # Add random perturbation to break symmetry and allow optimization to find better solutions
            U_t += np.random.randn(n*m, target_rank) * 0.1
            V_t += np.random.randn(m*p, target_rank) * 0.1
            W_t += np.random.randn(p*n, target_rank) * 0.1
            
            # Flatten parameters
            x0 = np.concatenate([U_t.flatten(), V_t.flatten(), W_t.flatten()])
            
            def objective(x):
                if time.time() > deadline - 2:
                    return 1e10
                val, _ = objective_and_gradient(x, target_rank)
                return val
            
            def gradient(x):
                if time.time() > deadline - 2:
                    return np.zeros_like(x)
                _, grad = objective_and_gradient(x, target_rank)
                return grad
            
            try:
                result = minimize(objective, x0, jac=gradient, method='L-BFGS-B', 
                                 options={'maxiter': max_iter, 'gtol': 1e-12, 'maxfun': 1000,
                                          'ftol': 1e-12})
            except Exception:
                continue
            
            if time.time() > deadline - 5:
                return None
            
            # Check if we found a good solution
            U_final = result.x[:n*m*target_rank].reshape(n*m, target_rank)
            V_final = result.x[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
            W_final = result.x[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
            
            # Polish with tighter tolerance
            if result.fun < 1e-4 and time.time() < deadline - 10:
                try:
                    result_polish = minimize(objective, result.x, jac=gradient, method='L-BFGS-B',
                                            options={'maxiter': 100, 'gtol': 1e-14, 'maxfun': 500,
                                                     'ftol': 1e-14})
                    if result_polish.fun < result.fun:
                        result = result_polish
                        U_final = result.x[:n*m*target_rank].reshape(n*m, target_rank)
                        V_final = result.x[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
                        W_final = result.x[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
                except Exception:
                    pass
            
            if check_decomposition(U_final, V_final, W_final, tol=1e-6):
                return U_final, V_final, W_final, target_rank
        
        return None
    
    # Try ranks from 45 down to 35, focusing on the most promising ranges
    for target_rank in range(45, 34, -1):
        if time.time() > deadline - 20:
            break
        
        result = try_l_bfgs(target_rank, num_trials=2, max_iter=200)
        if result is not None:
            U_t, V_t, W_t, r = result
            if check_decomposition(U_t, V_t, W_t, tol=1e-6):
                best_U, best_V, best_W = U_t, V_t, W_t
                best_rank = r
                break
    
    # If L-BFGS-B didn't find improvement, try alternating minimization as fallback
    if best_rank >= 46 and time.time() < deadline - 15:
        def try_alternating_min(target_rank, num_trials=3, max_iter=100):
            """Try to find a rank-target_rank decomposition using alternating minimization."""
            if target_rank >= best_rank:
                return None
            
            for trial in range(num_trials):
                if time.time() > deadline - 10:
                    return None
                
                np.random.seed(trial * 1000 + 42)
                
                # Initialize with structured pattern
                U_t = np.zeros((n*m, target_rank))
                V_t = np.zeros((m*p, target_rank))
                W_t = np.zeros((p*n, target_rank))
                
                for col in range(min(27, target_rank)):
                    i = col % n
                    j = (col // n) % m
                    k = (col // (n*m)) % p
                    U_t[i*m+j, col] = 1.0
                    V_t[j*p+k, col] = 1.0
                    W_t[k*n+i, col] = 1.0
                
                U_t += np.random.randn(n*m, target_rank) * 0.05
                V_t += np.random.randn(m*p, target_rank) * 0.05
                W_t += np.random.randn(p*n, target_rank) * 0.05
                
                # Alternating minimization
                for it in range(max_iter):
                    if time.time() > deadline - 10:
                        break
                    
                    # Update U
                    for t in range(target_rank):
                        vw = np.outer(V_t[:,t], W_t[:,t])
                        U_new_t = np.einsum('abc,bc->a', T_target, vw)
                        denom = np.dot(V_t[:,t], V_t[:,t]) * np.dot(W_t[:,t], W_t[:,t])
                        if denom > 1e-15:
                            U_t[:,t] = U_new_t / denom
                    
                    # Update V
                    for t in range(target_rank):
                        uw = np.outer(U_t[:,t], W_t[:,t])
                        V_new_t = np.einsum('abc,ac->b', T_target, uw)
                        denom = np.dot(U_t[:,t], U_t[:,t]) * np.dot(W_t[:,t], W_t[:,t])
                        if denom > 1e-15:
                            V_t[:,t] = V_new_t / denom
                    
                    # Update W
                    for t in range(target_rank):
                        uv = np.outer(U_t[:,t], V_t[:,t])
                        W_new_t = np.einsum('abc,ab->c', T_target, uv)
                        denom = np.dot(U_t[:,t], U_t[:,t]) * np.dot(V_t[:,t], V_t[:,t])
                        if denom > 1e-15:
                            W_t[:,t] = W_new_t / denom
                    
                    # Check convergence
                    if it % 10 == 9:
                        T_recon = np.zeros((n*m, m*p, p*n))
                        for t in range(target_rank):
                            T_recon += np.einsum('a,b,c->abc', U_t[:,t], V_t[:,t], W_t[:,t])
                        error = np.linalg.norm(T_recon - T_target)
                        if error < 1e-6:
                            return U_t, V_t, W_t, target_rank
                
                # Final check
                if check_decomposition(U_t, V_t, W_t, tol=1e-6):
                    return U_t, V_t, W_t, target_rank
            
            return None
        
        for target_rank in range(45, 42, -1):
            if time.time() > deadline - 10:
                break
            
            result = try_alternating_min(target_rank, num_trials=2, max_iter=80)
            if result is not None:
                U_t, V_t, W_t, r = result
                if check_decomposition(U_t, V_t, W_t, tol=1e-6):
                    best_U, best_V, best_W = U_t, V_t, W_t
                    best_rank = r
                    break
    
    # Convert to lists with integer/half-integer entries if possible
    def to_fraction_list(arr):
        """Convert numpy array to list of Fractions (integers or half-integers)."""
        result = []
        for row in arr:
            result_row = []
            for val in row:
                # Round to nearest half-integer
                val_rounded = round(val * 2) / 2
                if abs(val - val_rounded) < 1e-6:
                    result_row.append(Fraction(val_rounded))
                else:
                    # Keep as float if not close to half-integer
                    result_row.append(float(val))
            result.append(result_row)
        return result
    
    # For the return, use the best decomposition found
    # Convert to lists
    U_out = best_U.tolist()
    V_out = best_V.tolist()
    W_out = best_W.tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
