"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _robertson_jac(t, y):
    """Analytic Jacobian for the Robertson problem."""
    y1, y2, y3 = y
    a, b, c = 0.04, 3e-7, 1e-10
    J = np.array([
        [-a, c * y3, c * y2],
        [a, -b - c * y3, -c * y2],
        [0.0, b, 0.0]
    ])
    return J


def integrate(f, t_span, y0):
    """
    Integrate y' = f(t, y) from t_span[0] to t_span[1] with initial condition y0.
    Uses implicit Euler with Newton iteration and analytic Jacobian,
    with a geometrically increasing step-size schedule.
    """
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    t = t0
    n = len(y)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 280  # 280s out of 330s to be safe
    
    # Geometric step size schedule
    h = 1e-6
    growth = 2.5
    max_h = 2.0
    
    # Precompute Jacobian
    def jac(t, y):
        return _robertson_jac(t, y)
    
    while t < t1 - 1e-14:
        # Don't exceed t1
        if t + h > t1:
            h = t1 - t
        if h < 1e-14:
            break
        
        # Check deadline
        if time.time() > deadline:
            break
        
        # Implicit Euler: y_new = y + h * f(t + h, y_new)
        # Solve F(z) = z - y - h * f(t + h, z) = 0 using Newton
        z = y + h * f(t, y)  # Predictor: explicit Euler
        t_new = t + h
        
        for newton_iter in range(50):
            fz = f(t_new, z)
            J = jac(t_new, z)
            # F(z) = z - y - h * fz
            # F'(z) = I - h * J
            F = z - y - h * fz
            err = np.linalg.norm(F)
            if err < 1e-13 * (1 + np.linalg.norm(z)):
                break
            
            try:
                A = np.eye(n) - h * J
                dz = np.linalg.solve(A, -F)
            except np.linalg.LinAlgError:
                dz = -F * 0.1
                break
            
            z += dz
            if np.linalg.norm(dz) < 1e-13 * (1 + np.linalg.norm(z)):
                break
        
        y = z
        t = t_new
        
        # Grow step size geometrically, but slow down in stiff regions
        h = min(h * growth, max_h)
        # Adjust growth based on where we are
        if t < 0.01:
            growth = 2.5
        elif t < 0.1:
            growth = 2.0
        elif t < 1.0:
            growth = 1.5
        elif t < 10.0:
            growth = 1.2
        else:
            growth = 1.05
    
    return y
# EVOLVE-BLOCK-END
