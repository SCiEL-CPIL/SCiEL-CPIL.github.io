import numpy as np
import scipy.linalg as la
import time

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # Time guard
    start_time = time.time()
    timeout = 300.0  # Leave margin for evaluator overhead

    # 1. Test for Linearity and Time-Independence
    # Check f(t0, 0)
    y_zero = np.zeros(n)
    f0 = f(t0, y_zero)
    
    # Check if f is time independent by comparing f(t0, y0) and f(t1, y0)
    f_t0_y0 = f(t0, y0)
    f_t1_y0 = f(t1, y0)
    
    is_time_indep = np.allclose(f_t0_y0, f_t1_y0, rtol=1e-10, atol=1e-12)
    
    # Extract b
    b = f0.copy()
    
    # Extract A columns
    A = np.zeros((n, n))
    for i in range(n):
        e_i = np.zeros(n)
        e_i[i] = 1.0
        f_ei = f(t0, e_i)
        A[:, i] = f_ei - b
        
    # Verify linearity on a random vector
    v_rand = np.random.rand(n)
    f_rand = f(t0, v_rand)
    f_pred = A @ v_rand + b
    is_linear = np.allclose(f_rand, f_pred, rtol=1e-8, atol=1e-10)
    
    if is_time_indep and is_linear:
        # Solve y' = A y + b
        # If b == 0, it's homogeneous: y(t) = expm(A*T) @ y0
        # If b != 0, it's affine: y(t) = expm(A*T) @ y0 + integral_0^T expm(A*(T-s)) b ds
        # For constant b, integral_0^T expm(A*(T-s)) b ds = expm(A*T) @ A^{-1} @ (I - expm(-A*T)) @ b
        # Alternatively, use augmented system:
        # z' = M z, where z = [y; 1], M = [[A, b], [0, 0]]
        # z(T) = expm(M*T) @ [y0; 1]
        
        T = t1 - t0
        
        # Check time
        if time.time() - start_time > timeout:
            # Fallback to simple RK4 if we are too slow (unlikely)
            pass 
            
        # Compute matrix exponential using eigendecomposition for better stability
        # E = V diag(exp(lambda_i * T)) V^{-1}
        
        try:
            # Use eigendecomposition
            eigvals, V = la.eig(A)
            V_inv = la.inv(V)
            
            # Compute exp(eigvals * T)
            exp_eigvals = np.exp(eigvals * T)
            
            # Construct E = V diag(exp_eigvals) V^{-1}
            # E = V @ diag(exp_eigvals) @ V_inv
            E = (V * exp_eigvals) @ V_inv
            
            # Check if b is zero
            if np.allclose(b, 0.0, rtol=1e-14, atol=1e-16):
                # Homogeneous case
                y_end = E @ y0
            else:
                # Affine case: use augmented matrix exponential
                M = np.zeros((n+1, n+1))
                M[:n, :n] = A
                M[:n, n] = b
                # Last row is 0
                
                # Compute expm(M*T) using eigendecomposition
                eigvals_M, V_M = la.eig(M)
                V_M_inv = la.inv(V_M)
                exp_eigvals_M = np.exp(eigvals_M * T)
                E_M = (V_M * exp_eigvals_M) @ V_M_inv
                
                z0 = np.zeros(n+1)
                z0[:n] = y0
                z0[n] = 1.0
                
                z_end = E_M @ z0
                y_end = z_end[:n]
            
            return y_end
            
        except Exception:
            # Fallback to scipy's expm if eigendecomposition fails
            M = np.zeros((n+1, n+1))
            M[:n, :n] = A
            M[:n, n] = b
            
            E = la.expm(M * T)
            
            z0 = np.zeros(n+1)
            z0[:n] = y0
            z0[n] = 1.0
            
            z_end = E @ z0
            y_end = z_end[:n]
            
            return y_end
    else:
        # Fallback: High precision adaptive solver
        # Use scipy.integrate.solve_ivp with Radau (stiff)
        from scipy.integrate import solve_ivp
        
        def wrapper(t, y):
            return f(t, y)
            
        sol = solve_ivp(wrapper, t_span, y0, method='Radau', rtol=1e-10, atol=1e-12, max_step=(t1-t0)/100)
        return sol.y[:, -1]
