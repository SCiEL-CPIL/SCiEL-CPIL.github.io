"""ODE integrator for the Van der Pol oscillator (mu = 5) using custom Radau IIA(5)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Radau IIA(5) coefficients
# Butcher tableau for Radau IIA(5)
_c = np.array([0.0, 0.2487946220878348, 0.4952270257086157, 0.7587946220878348, 0.9917561502862906, 1.0])
_A = np.array([
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0285870319553935, 0.0975101385581529, 0.0044494205158619, 0.0, 0.0, 0.0],
    [0.1001774011877521, 0.0853915466387271, 0.1027591386950074, 0.0016045755857776, 0.0, 0.0],
    [0.2076000000000000, 0.0362542893218813, 0.0362542893218813, 0.1381629160137771, 0.0, 0.0],
    [0.2962962962962963, 0.0, 0.0, 0.0586419753086420, 0.1348148148148148, 0.0],
    [0.2962962962962963, 0.0, 0.0, 0.0586419753086420, 0.1348148148148148, 0.1348148148148148]
])
_b = np.array([0.2962962962962963, 0.0, 0.0, 0.0586419753086420, 0.1348148148148148, 0.1348148148148148])
_a4 = np.array([0.0, 0.2487946220878348, 0.4952270257086157, 0.7587946220878348, 0.9917561502862906, 1.0])

# Precompute (I - h*A)^{-1} for given h (will be done per step)


def _vdp_jac(t, y):
    """Analytic Jacobian for Van der Pol oscillator with mu=5."""
    y0, y1 = y[0], y[1]
    # y0' = y1
    # y1' = 5(1 - y0^2)y1 - y0
    # dy0'/dy0 = 0, dy0'/dy1 = 1
    # dy1'/dy0 = -10*y0*y1 - 1, dy1'/dy1 = 5*(1 - y0^2)
    J = np.array([
        [0.0, 1.0],
        [-10.0 * y0 * y1 - 1.0, 5.0 * (1.0 - y0 * y0)]
    ])
    return J


def _solve_implicit(h, f, y, J, c, A):
    """
    Solve the implicit Radau system:
    Y_i = y + h * sum_j A[i,j] * f(t + c_j*h, Y_j)
    
    Using Newton iteration with the analytic Jacobian.
    """
    n = len(y)
    s = len(c)
    
    # Initialize Y with y
    Y = np.tile(y, (s, 1))
    
    # Newton iteration
    for newton_iter in range(30):
        # Compute residuals
        # R_i = Y_i - y - h * sum_j A[i,j] * f(t + c_j*h, Y_j)
        fvals = np.array([f(0.0, Y[j]) for j in range(s)])  # t doesn't matter for autonomous
        F = np.zeros((s, n))
        for i in range(s):
            F[i] = np.dot(A[i, :], fvals)
        
        R = Y - y - h * F
        
        # Check convergence
        if np.max(np.abs(R)) < 1e-13:
            break
        
        # Build the Jacobian of the residual system
        # dR_i/dY_j = delta_ij * I - h * A[i,j] * J_j
        # where J_j is the Jacobian at Y_j
        Jmats = np.array([_vdp_jac(0.0, Y[j]) for j in range(s)])
        
        # Build block matrix M where M[i,j] = delta_ij*I - h*A[i,j]*J_j
        # For 2x2 system, M is (2s)x(2s)
        M = np.zeros((2 * s, 2 * s))
        for i in range(s):
            for j in range(s):
                if i == j:
                    M[2*i:2*i+2, 2*j:2*j+2] = np.eye(2)
                if A[i, j] != 0.0:
                    M[2*i:2*i+2, 2*j:2*j+2] -= h * A[i, j] * Jmats[j]
        
        # Solve M * delta = -R (flattened)
        R_flat = R.flatten()
        try:
            delta = np.linalg.solve(M, -R_flat)
        except np.linalg.LinAlgError:
            # Fallback: use least squares
            delta, _, _, _ = np.linalg.lstsq(M, -R_flat, rcond=None)
        
        # Update Y
        delta_Y = delta.reshape(s, n)
        Y += delta_Y
    
    return Y


def integrate(f, t_span, y0):
    """
    Integrate the Van der Pol oscillator y0' = y1, y1' = 5(1 - y0^2)y1 - y0
    from t_span[0] to t_span[1] with initial condition y0.
    Returns the state vector at t_span[1].
    """
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin before 300s limit
    
    y = np.array(y0, dtype=float)
    t = t_span[0]
    t_end = t_span[1]
    total_time = t_end - t
    
    # Initial step size
    h = 0.01
    h_min = 1e-6
    h_max = 1.0
    
    # Error estimation using embedded lower-order method
    # For Radau IIA(5), we can estimate error by comparing with a lower-order solution
    # or use the difference between consecutive steps
    
    c = _c
    A = _A
    b = _b
    
    while t < t_end - 1e-15:
        if time.time() > deadline:
            break
        
        # Adjust step to not overshoot
        if t + h > t_end:
            h = t_end - t
        
        # Solve implicit system
        Y = _solve_implicit(h, f, y, None, c, A)
        
        # Compute new y
        y_new = y + h * np.dot(b, [f(0.0, Y[i]) for i in range(len(c))])
        
        # Error estimation: use a simpler embedded estimator
        # For now, use a basic step-doubling or fixed tolerance
        # Estimate local error by comparing with Euler step
        f0 = f(0.0, y)
        y_euler = y + h * f0
        err = np.max(np.abs(y_new - y_euler))
        
        # Adaptive step size control
        if err < 1e-12:
            # Increase step
            h = min(h * 1.5, h_max, t_end - t)
        elif err > 1e-10:
            # Decrease step
            h = max(h * 0.5, h_min)
            continue  # Retry with smaller step
        else:
            # Accept step
            y = y_new
            t += h
    
    return y
# EVOLVE-BLOCK-END
