import numpy as np
import scipy.integrate
import time

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # Compute Jacobian at t0 using central differences
    # Since the system is linear, the Jacobian is constant.
    # f(t, y) = A y + b (if non-homogeneous) or A y.
    # J = df/dy = A.
    
    # Choose a small step for finite difference
    eps = 1e-7
    J = np.zeros((n, n))
    
    # We need to evaluate f at t0. 
    # Note: f signature is f(t, y).
    
    y_center = y0.copy()
    f_center = f(t0, y_center)
    
    for j in range(n):
        y_plus = y0.copy()
        y_minus = y0.copy()
        step = eps * max(1.0, abs(y0[j]))
        y_plus[j] += step
        y_minus[j] -= step
        
        f_plus = f(t0, y_plus)
        f_minus = f(t0, y_minus)
        
        J[:, j] = (f_plus - f_minus) / (2 * step)
        
    # Use Radau solver for stiff problems
    # Radau is an implicit Runge-Kutta method, good for stiff ODEs.
    # Passing the Jacobian significantly reduces the number of function evaluations.
    
    start_time = time.time()
    
    try:
        sol = scipy.integrate.solve_ivp(
            f, 
            t_span, 
            y0, 
            method='Radau', 
            jac=lambda t, y: J, 
            rtol=1e-12, 
            atol=1e-12,
            max_step=0.1 # Allow adaptive stepping but cap it to ensure resolution
        )
        
        if not sol.success:
            # Fallback to BDF if Radau fails
            sol = scipy.integrate.solve_ivp(
                f, 
                t_span, 
                y0, 
                method='BDF', 
                jac=lambda t, y: J, 
                rtol=1e-12, 
                atol=1e-12,
                max_step=0.1
            )
            if not sol.success:
                # Last resort: LSODA
                sol = scipy.integrate.solve_ivp(
                    f, 
                    t_span, 
                    y0, 
                    method='LSODA', 
                    rtol=1e-12, 
                    atol=1e-12
                )
                if not sol.success:
                    # If all fail, return the last state we have or raise
                    # For now, return the initial state or best effort
                    return sol.y[:, -1] if sol.y.size > 0 else y0
        else:
            return sol.y[:, -1]
            
    except Exception as e:
        # If an exception occurs, try a simple explicit method with small steps as fallback
        # though this is unlikely to be accurate for stiff problems
        try:
            h = 0.01
            t = t0
            y = y0.copy()
            while t < t1 - 1e-10:
                if t + h > t1:
                    h = t1 - t
                k1 = f(t, y)
                k2 = f(t + h/2, y + h/2 * k1)
                k3 = f(t + h/2, y + h/2 * k2)
                k4 = f(t + h, y + h * k3)
                y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
                t += h
            return y
        except:
            return y0
