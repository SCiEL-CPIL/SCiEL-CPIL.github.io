import numpy as np
import time

def integrate(f, t_span, y0):
    start_time = time.time()
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    n = len(y)
    
    # Fixed step size. 
    # Robertson problem is stiff but smooth. 
    # With Radau IIA (3rd order), stability region is large.
    # We want high accuracy. 
    # Let's pick a step size that results in ~1000-5000 steps.
    # Total span 40. 
    # If we use 1000 steps, h = 0.04.
    # If we use 2000 steps, h = 0.02.
    # Let's try 2000 steps.
    num_steps = 2000
    h = (t1 - t0) / num_steps
    t = t0
    
    # Radau IIA 3rd order coefficients
    # c1 = 0, c2 = 1 - 1/(2*sqrt(3)), c3 = 1
    # a11 = 0
    # a21 = 1/(2*(3 - 2*sqrt(3)))
    # a22 = 1/2 - 1/(2*(3 - 2*sqrt(3)))
    # a31 = 1/(2*(3 + 2*sqrt(3)))
    # a32 = 1/2 - 1/(2*(3 + 2*sqrt(3)))
    # a33 = 1/2 + 1/(2*(3 + 2*sqrt(3)))
    # b1 = 1/6
    # b2 = 2/3
    # b3 = 1/6
    
    sqrt3 = np.sqrt(3)
    
    a21 = 1.0 / (2.0 * (3.0 - 2.0 * sqrt3))
    a22 = 0.5 - a21
    a31 = 1.0 / (2.0 * (3.0 + 2.0 * sqrt3))
    a32 = 0.5 - a31
    a33 = 0.5 + a31
    
    b1 = 1.0 / 6.0
    b2 = 2.0 / 3.0
    b3 = 1.0 / 6.0
    
    # Pre-compute Jacobian approximation? 
    # Since h is fixed and problem is small (n=3), we can use finite difference Jacobian 
    # or just solve the linear system (I - h*A) directly if we linearize? 
    # Radau IIA is implicit:
    # Y1 = y_n
    # Y2 = y_n + h * (a21*f(t, Y1) + a22*f(t+h*c2, Y2))
    # Y3 = y_n + h * (a31*f(t, Y1) + a32*f(t+h*c2, Y2) + a33*f(t+h, Y3))
    # y_{n+1} = y_n + h * (b1*f(t, Y1) + b2*f(t+h*c2, Y2) + b3*f(t+h, Y3))
    
    # For small n=3, solving 3x3 linear systems is cheap.
    # We can use Newton's method for the implicit stages.
    # Or, since the nonlinearity is mild over small h, we can use a few fixed-point iterations 
    # or just Newton.
    # Given n=3, Newton is robust.
    
    # Jacobian estimation function
    def get_jacobian(t_val, y_val):
        J = np.zeros((n, n))
        f0 = f(t_val, y_val)
        for j in range(n):
            e = np.zeros(n)
            eps = 1e-7 * max(1.0, abs(y_val[j]))
            e[j] = eps
            J[:, j] = (f(t_val, y_val + e) - f0) / eps
        return J

    for i in range(num_steps):
        # Check time limit
        if time.time() - start_time > 280:
            # Return best guess (current y) if we run out of time, 
            # though 2000 steps should be fast.
            break
        
        # Stage 1
        Y1 = y
        f1 = f(t, Y1)
        
        # Stage 2
        # Y2 = y + h * (a21*f1 + a22*f2)
        # (I - h*a22*J) * Y2 = y + h*a21*f1
        # We need J at some point. Let's estimate J at t+h*c2, y + h*a21*f1 (initial guess)
        c2 = 1.0 - 1.0/(2.0*sqrt3)
        t2 = t + h * c2
        Y2_guess = y + h * a21 * f1
        J2 = get_jacobian(t2, Y2_guess)
        
        A2 = np.eye(n) - h * a22 * J2
        rhs2 = y + h * a21 * f1
        Y2 = np.linalg.solve(A2, rhs2)
        
        # Newton correction for Y2
        for _ in range(3):
            f2 = f(t2, Y2)
            resid = Y2 - (y + h * (a21 * f1 + a22 * f2))
            J2_new = get_jacobian(t2, Y2)
            A2_new = np.eye(n) - h * a22 * J2_new
            try:
                delta_Y2 = np.linalg.solve(A2_new, -resid)
            except:
                break
            Y2 += delta_Y2
            if np.linalg.norm(delta_Y2) < 1e-12:
                break
        f2 = f(t2, Y2)
        
        # Stage 3
        # Y3 = y + h * (a31*f1 + a32*f2 + a33*f3)
        # (I - h*a33*J) * Y3 = y + h*(a31*f1 + a32*f2)
        c3 = 1.0
        t3 = t + h * c3
        Y3_guess = y + h * (a31 * f1 + a32 * f2)
        J3 = get_jacobian(t3, Y3_guess)
        
        A3 = np.eye(n) - h * a33 * J3
        rhs3 = y + h * (a31 * f1 + a32 * f2)
        Y3 = np.linalg.solve(A3, rhs3)
        
        # Newton correction for Y3
        for _ in range(3):
            f3 = f(t3, Y3)
            resid = Y3 - (y + h * (a31 * f1 + a32 * f2 + a33 * f3))
            J3_new = get_jacobian(t3, Y3)
            A3_new = np.eye(n) - h * a33 * J3_new
            try:
                delta_Y3 = np.linalg.solve(A3_new, -resid)
            except:
                break
            Y3 += delta_Y3
            if np.linalg.norm(delta_Y3) < 1e-12:
                break
        f3 = f(t3, Y3)
        
        # Update y
        y = y + h * (b1 * f1 + b2 * f2 + b3 * f3)
        t = t3
        
    return y

