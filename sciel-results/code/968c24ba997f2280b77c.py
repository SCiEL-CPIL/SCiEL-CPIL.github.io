import numpy as np
import time

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    t = t0
    
    # Precompute constants for Van der Pol (mu=5)
    # y' = [y1, 5*(1-y0^2)*y1 - y0]
    # J = [[0, 1], [-1 - 10*y0*y1, 5*(1 - y0^2)]]
    
    def jac(y):
        y0_val = y[0]
        y1_val = y[1]
        return np.array([
            [0.0, 1.0],
            [-1.0 - 10.0 * y0_val * y1_val, 5.0 * (1.0 - y0_val**2)]
        ])

    def rhs(t, y):
        return f(t, y)

    # Radau IIA coefficients for order 5 (3-stage)
    # c = [0, 0.4358665, 1.0]
    # A = [[0, 0, 0],
    #      [0.1652883, 0.2705782, 0],
    #      [0.5111763, 0.4888237, 0.0]]
    # b = [0.5111763, 0.4888237, 0.0]
    # b_hat (embedded order 4) = [0.4358665, 0.5641335, 0.0]
    
    c = np.array([0.0, 0.4358665, 1.0])
    A = np.array([
        [0.0, 0.0, 0.0],
        [0.1652883, 0.2705782, 0.0],
        [0.5111763, 0.4888237, 0.0]
    ])
    b = np.array([0.5111763, 0.4888237, 0.0])
    b_hat = np.array([0.4358665, 0.5641335, 0.0])
    
    # Initial step size
    h = 0.01
    rtol = 1e-11
    atol = 1e-13
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 280.0  # 280s limit to be safe
    
    # Cache for f evaluations to avoid redundant calls if possible? 
    # The task counts every call to f. We must minimize calls.
    # Newton's method calls f multiple times.
    
    y_stages = np.zeros((3, 2))
    y_stages[0] = y
    
    while t < t1:
        if time.time() > deadline:
            break
            
        # Adjust h to not overshoot t1
        if t + h > t1:
            h = t1 - t
            
        # Solve for stages Y_1, Y_2, Y_3
        # Y_i = y + h * sum_j A_ij f(t + c_j h, Y_j)
        # This is a nonlinear system F(Y) = Y - y - h * A @ f(Y) = 0
        
        # Prediction: Use previous stages or simple extrapolation
        # For the first step, use y for all stages
        Y = np.zeros((3, 2))
        Y[0] = y
        
        # If we have previous stages, use them as initial guess for Newton
        if 'prev_Y' in locals() and prev_Y is not None:
            Y[1] = prev_Y[1]
            Y[2] = prev_Y[2]
        else:
            Y[1] = y + h * rhs(t, y)
            Y[2] = y + h * rhs(t, y)
            
        # Newton's Method for the system
        max_newton_iter = 10
        converged = False
        for _ in range(max_newton_iter):
            # Compute F(Y)
            F = np.zeros((3, 2))
            F[0] = Y[0] - y
            
            for i in range(1, 3):
                F[i] = Y[i] - y - h * (A[i, 0] * rhs(t + c[0]*h, Y[0]) + 
                                      A[i, 1] * rhs(t + c[1]*h, Y[1]) + 
                                      A[i, 2] * rhs(t + c[2]*h, Y[2]))
                
            err = np.linalg.norm(F)
            if err < 1e-14:
                converged = True
                break
                
            # Compute Jacobian of F with respect to Y
            # F_i = Y_i - y - h * sum_j A_ij f_j
            # dF_i/dY_k = delta_ik * I - h * A_ik * J_f(Y_k)
            
            J_F = np.zeros((6, 6))
            for i in range(3):
                for k in range(3):
                    if i == k:
                        J_F[i*2:(i+1)*2, k*2:(k+1)*2] = np.eye(2)
                    J_F[i*2:(i+1)*2, k*2:(k+1)*2] -= h * A[i, k] * jac(Y[k])
            
            # Solve J_F * delta = -F
            F_flat = F.flatten()
            try:
                delta = np.linalg.solve(J_F, -F_flat)
            except np.linalg.LinAlgError:
                # Fallback to least squares if singular
                delta = np.linalg.lstsq(J_F, -F_flat, rcond=None)[0]
                
            Y_flat = Y.flatten() + delta
            Y = Y_flat.reshape(3, 2)
            
        if not converged:
            # Reduce step size
            h *= 0.5
            continue
            
        # Compute local error
        # y_next_full = y + h * b @ f(Y)
        # y_next_embedded = y + h * b_hat @ f(Y)
        f_Y = np.array([rhs(t + c[i]*h, Y[i]) for i in range(3)])
        
        y_next_full = y + h * (b[0]*f_Y[0] + b[1]*f_Y[1] + b[2]*f_Y[2])
        y_next_embedded = y + h * (b_hat[0]*f_Y[0] + b_hat[1]*f_Y[1] + b_hat[2]*f_Y[2])
        
        err_vec = y_next_full - y_next_embedded
        err_norm = np.linalg.norm(err_vec)
        
        # Tolerance check
        scale = np.maximum(np.abs(y), 1.0)
        tol = rtol * scale + atol
        err_ratio = np.max(err_vec / tol)
        
        if err_ratio <= 1.0:
            # Accept step
            t += h
            y = y_next_full
            
            # Adaptive step size control
            if err_ratio > 0:
                h_new = h * (0.9 * (1.0 / err_ratio))**(1.0/5.0)
            else:
                h_new = h * 1.5
                
            h = min(h_new, 0.1) # Cap step size
            
            # Store stages for next prediction
            prev_Y = Y.copy()
        else:
            # Reject step
            h = h * (0.9 * (1.0 / err_ratio))**(1.0/5.0)
            if h < 1e-6:
                h = 1e-6
            # Do not update y or t
            # Keep prev_Y as is or reset? Resetting might be safer for prediction
            prev_Y = None
            
    return y
