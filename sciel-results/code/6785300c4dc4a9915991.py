import numpy as np
import scipy.linalg as la
import time

def integrate(f, t_span, y0):
        t0, t1 = t_span
        n = len(y0)
        y0 = np.array(y0, dtype=float)
        
        # Time guard
        start_time = time.time()
        timeout = 300.0 # Leave margin for evaluator overhead

        # 1. Test for Linearity and Time-Independence
        # Check f(t0, 0)
        y_zero = np.zeros(n)
        f0 = f(t0, y_zero)
        
        # Check if f is time independent by comparing f(t0, y0) and f(t1, y0)
        f_t0_y0 = f(t0, y0)
        f_t1_y0 = f(t1, y0)
        
        is_time_indep = np.allclose(f_t0_y0, f_t1_y0, rtol=1e-10, atol=1e-12)
        
        # Check linearity: f(t, a*x + b*y) == a*f(t,x) + b*f(t,y) ?
        # We need to check if f is linear map.
        # f(t, y) = A y + b
        # If f(0) != 0, b = f(0).
        # Then A y = f(y) - b.
        # Check if A is linear.
        
        # Let's assume it is linear/affine and time-independent if checks pass.
        # To be safe, we can try to extract A and verify on a random vector.
        
        # Extract b
        b = f0.copy()
        
        # Extract A columns
        A = np.zeros((n, n))
        for i in range(n):
            e_i = np.zeros(n)
            e_i[i] = 1.0
            # f(e_i) = A e_i + b = col_i(A) + b
            # col_i(A) = f(e_i) - b
            # But we need to be careful with scaling if we use small e_i? No, linearity holds for any scale.
            # Using 1.0 is fine.
            f_ei = f(t0, e_i)
            A[:, i] = f_ei - b
            
        # Verify linearity on a random vector
        v_rand = np.random.rand(n)
        f_rand = f(t0, v_rand)
        f_pred = A @ v_rand + b
        is_linear = np.allclose(f_rand, f_pred, rtol=1e-8, atol=1e-10)
        
        if is_time_indep and is_linear:
            # Solve y' = A y + b
            # Augmented system
            M = np.zeros((n+1, n+1))
            M[:n, :n] = A
            M[:n, n] = b
            # Last row is 0
            
            T = t1 - t0
            
            # Check time
            if time.time() - start_time > timeout:
                # Fallback to simple RK4 if we are too slow (unlikely)
                pass 
            
            # Compute matrix exponential
            # expm(M * T)
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
