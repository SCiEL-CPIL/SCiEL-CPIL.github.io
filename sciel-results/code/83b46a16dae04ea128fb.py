import numpy as np
import time

# Coefficients for 3-stage Gauss-Legendre (6th order)
# Nodes c = [0.5 - sqrt(15)/6, 0.5, 0.5 + sqrt(15)/6]
# Weights b = [5/18, 4/9, 5/18]
# Matrix A:
# a_11 = 5/36 - 5*sqrt(15)/108
# a_12 = 2/9 - 5*sqrt(15)/108
# a_13 = 5/36 + 5*sqrt(15)/108
# a_21 = 5/36 + 5*sqrt(15)/108
# a_22 = 4/9
# a_23 = 5/36 - 5*sqrt(15)/108
# a_31 = 5/36 - 5*sqrt(15)/108
# a_32 = 2/9 + 5*sqrt(15)/108
# a_33 = 5/36 + 5*sqrt(15)/108

sqrt15 = np.sqrt(15.0)
c1 = 0.5 - sqrt15/6.0
c2 = 0.5
c3 = 0.5 + sqrt15/6.0

b1 = 5.0/18.0
b2 = 4.0/9.0
b3 = 5.0/18.0

a11 = 5.0/36.0 - 5.0*sqrt15/108.0
a12 = 2.0/9.0 - 5.0*sqrt15/108.0
a13 = 5.0/36.0 + 5.0*sqrt15/108.0

a21 = 5.0/36.0 + 5.0*sqrt15/108.0
a22 = 4.0/9.0
a23 = 5.0/36.0 - 5.0*sqrt15/108.0

a31 = 5.0/36.0 - 5.0*sqrt15/108.0
a32 = 2.0/9.0 + 5.0*sqrt15/108.0
a33 = 5.0/36.0 + 5.0*sqrt15/108.0

A = np.array([
    [a11, a12, a13],
    [a21, a22, a23],
    [a31, a32, a33]
])
B = np.array([b1, b2, b3])
C = np.array([c1, c2, c3])

def integrate(f, t_span, y0):
    """
    Integrate y' = f(t, y) from t_span[0] to t_span[1] with initial condition y0.
    
    Strategy: Custom 3-stage Gauss-Legendre Runge-Kutta (6th order) with 
    Newton-Raphson for stage resolution and step-doubling for error control.
    """
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    n = len(y)
    
    # Initial step size estimate
    # Robertson is stiff, start small
    h = 0.01
    
    rtol = 1e-10
    atol = 1e-14
    
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin
    
    def newton_step(t, y_n, h):
        """
        Solve for stage values Y = [y1, y2, y3] using Newton's method.
        Y_i = y_n + h * sum_j A_ij f(t + c_j h, Y_j)
        """
        # Initial guess: Euler prediction
        f0 = f(t, y_n)
        Y = np.vstack([y_n + h * f0] * 3) # Shape (3, n)
        
        for _ in range(10): # Max Newton iterations
            # Evaluate f at stages
            F = np.vstack([
                f(t + C[0]*h, Y[0]),
                f(t + C[1]*h, Y[1]),
                f(t + C[2]*h, Y[2])
            ]) # Shape (3, n)
            
            # Residual R = Y - y_n - h * (A @ F)
            # A @ F is matrix multiplication: (3x3) @ (3xn) -> (3xn)
            AF = A @ F
            R = Y - y_n - h * AF
            
            # Check convergence
            r_norm = np.max(np.abs(R))
            if r_norm < 1e-12:
                break
            
            # Compute Jacobian of R with respect to Y
            # R_i = Y_i - y_n - h * sum_j A_ij F_j(Y)
            # dR_i / dY_k = (delta_ik) I - h A_ik J_f(Y_k)
            # We approximate J_f using finite differences.
            # Since n is small (3 for Robertson), we can do this.
            
            J = np.zeros((3*n, 3*n))
            
            # We need J_f for each stage k
            # J_f_k is n x n
            # We compute J_f_k for k=0,1,2
            
            for k in range(3):
                y_k = Y[k]
                t_k = t + C[k]*h
                f_k = F[k]
                
                # Compute Jacobian of f at (t_k, y_k) using forward differences
                J_f_k = np.zeros((n, n))
                eps = 1e-8
                for j in range(n):
                    y_pert = y_k.copy()
                    y_pert[j] += eps
                    f_pert = f(t_k, y_pert)
                    J_f_k[:, j] = (f_pert - f_k) / eps
                
                # Now fill in the Jacobian block for R_i with respect to Y_k
                # R_i depends on Y_k via the term -h * A_ik * F_k(Y_k)
                # dR_i / dY_k = (delta_ik) I - h * A_ik * J_f_k
                
                for i in range(3):
                    block = np.zeros((n, n))
                    if i == k:
                        block = np.eye(n)
                    block -= h * A[i, k] * J_f_k
                    
                    # Place block in J
                    J[i*n:(i+1)*n, k*n:(k+1)*n] = block
            
            # Solve J * delta = -R
            R_flat = -R.flatten()
            try:
                delta = np.linalg.solve(J, R_flat)
            except np.linalg.LinAlgError:
                # If singular, use least squares
                delta = np.linalg.lstsq(J, R_flat, rcond=None)[0]
            
            # Update Y
            Y += delta.reshape(3, n)
            
            # Damping if divergence is detected (optional, but good for robustness)
            # For now, assume convergence.
            
        return Y, F

    t = t0
    while t < t1:
        if time.time() > deadline:
            break
            
        # Cap step size to not overshoot
        if t + h > t1:
            h = t1 - t
        if h <= 0:
            break
            
        # Attempt step with size h
        try:
            Y_full, F_full = newton_step(t, y, h)
            # New state
            y_new = y + h * (B @ F_full)
        except Exception:
            # If Newton fails, reduce step size
            h *= 0.5
            continue
        
        # Error estimation via step doubling
        # Take two steps of h/2
        try:
            Y_half1, F_half1 = newton_step(t, y, h/2.0)
            y_mid = y + (h/2.0) * (B @ F_half1)
            
            Y_half2, F_half2 = newton_step(t + h/2.0, y_mid, h/2.0)
            y_new_half = y_mid + (h/2.0) * (B @ F_half2)
        except Exception:
            h *= 0.5
            continue
            
        # Error estimate: difference between y_new and y_new_half
        # Scale by norm of y
        scale = np.maximum(atol, rtol * np.maximum(np.abs(y), np.abs(y_new)))
        err = np.max(np.abs(y_new - y_new_half) / scale)
        
        if err <= 1.0:
            # Accept step
            # Use the more accurate solution (step-doubled)
            y = y_new_half
            t += h
            
            # Adjust step size
            if err > 0.1:
                h *= 1.5
            else:
                h *= 1.2
        else:
            # Reject step
            h *= 0.5
            if h < 1e-12:
                break

    return y
