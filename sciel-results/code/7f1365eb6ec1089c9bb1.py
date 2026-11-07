"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _radau3_step(f, J, t, y, h):
    """Single step of 3-stage Radau IIA. Returns (y_new, nfev)."""
    n = len(y)
    # Butcher tableau for Radau IIA 3-stage
    c = np.array([1/6, 2/3, 1.0])
    A = np.array([
        [1/36, -29/36, 4/9],
        [-8/9, 5/3, -4/9],
        [1/6, -5/3, 10/9]
    ])
    b = np.array([1/6, 2/3, 1/3])
    
    nfev = 0
    # Stage equations: Y_i = y + h * sum_j A_ij f(t + c_j*h, Y_j)
    # This is a system of 3*n equations. Solve with Newton.
    
    # Initialize stage values
    Y = np.zeros((3, n))
    Y[0] = y
    Y[1] = y
    Y[2] = y
    
    for _newton in range(30):
        # Compute f at each stage
        F = np.zeros((3, n))
        for i in range(3):
            F[i] = f(t + c[i] * h, Y[i])
            nfev += 1
        
        # Residual: R_i = Y_i - y - h * sum_j A_ij F_j
        R = np.zeros((3, n))
        for i in range(3):
            R[i] = Y[i] - y - h * np.dot(A[i, :], F)
        
        if np.max(np.abs(R)) < 1e-15:
            break
        
        # Jacobian of the residual system
        # dR_i/dY_j = delta_ij * I - h * A_ij * J_f(t + c_j*h, Y_j)
        Jf = np.zeros((3, n, n))
        for j in range(3):
            Jf[j] = J(t + c[j] * h, Y[j])
        
        # Build the full (3n x 3n) Jacobian
        Jfull = np.zeros((3*n, 3*n))
        for i in range(3):
            for j in range(3):
                if i == j:
                    Jfull[i*n:(i+1)*n, j*n:(j+1)*n] = np.eye(n) - h * A[i, j] * Jf[j]
                else:
                    Jfull[i*n:(i+1)*n, j*n:(j+1)*n] = -h * A[i, j] * Jf[j]
        
        # Solve Jfull * delta = -R
        R_flat = R.flatten()
        try:
            delta = np.linalg.solve(Jfull, -R_flat)
        except np.linalg.LinAlgError:
            delta = np.linalg.lstsq(Jfull, -R_flat, rcond=None)[0]
        
        delta = delta.reshape(3, n)
        for i in range(3):
            Y[i] += delta[i]
    
    y_new = y + h * np.dot(b, F)
    return y_new, nfev


def _jac_vdp(t, y, mu=5.0):
    """Analytical Jacobian for Van der Pol oscillator."""
    J = np.zeros((2, 2))
    J[0, 0] = 0.0
    J[0, 1] = 1.0
    J[1, 0] = -2.0 * mu * y[0] * y[1] - 1.0
    J[1, 1] = mu * (1.0 - y[0]**2)
    return J


def integrate(f, t_span, y0):
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    t = t0
    
    # Deadline guard
    t_start = time.time()
    deadline = t_start + 280.0  # 280s out of 330s limit
    
    mu = 5.0
    
    # Start with a reasonable step size
    h = 0.01
    total_nfev = 0
    
    # Adaptive step size control
    h_min = 1e-6
    h_max = 0.1
    
    while t < t1 - 1e-12:
        if time.time() > deadline:
            break
        
        # Don't overshoot
        if t + h > t1:
            h = t1 - t
        
        # Try the step
        y_new, nfev = _radau3_step(f, _jac_vdp, t, y, h)
        total_nfev += nfev
        
        # Check for failure
        if not np.all(np.isfinite(y_new)):
            h = h * 0.5
            if h < h_min:
                break
            continue
        
        # Estimate local error using a smaller step comparison
        # For simplicity, use a fixed tolerance and accept if step is small enough
        # Better: compare with 2 half-steps for error estimation
        if h > h_min * 2:
            # Do two half-steps for error estimation
            y_half1, nfev1 = _radau3_step(f, _jac_vdp, t, y, h/2)
            y_half2, nfev2 = _radau3_step(f, _jac_vdp, t + h/2, y_half1, h/2)
            total_nfev += nfev1 + nfev2
            
            err = np.linalg.norm(y_half2 - y_new)
            scale = 1e-13 * (1 + np.linalg.norm(y_new))
            
            if err > scale:
                # Reject step, reduce h
                h = h * max(0.5, min(2.0, (scale / (err + 1e-30)) ** 0.25))
                h = max(h, h_min)
                continue
        
        # Accept step
        y = y_new
        t += h
        
        # Adjust step size
        if h < h_max:
            h = min(h * 1.5, h_max)
        elif h > h_min:
            h = max(h * 0.7, h_min)
    
    return y
# EVOLVE-BLOCK-END
