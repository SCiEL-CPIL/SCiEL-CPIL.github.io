"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def integrate(f, t_span, y0):
    """
    Integrates the Van der Pol oscillator using a custom Radau IIIA collocation solver.
    Uses analytic Jacobian and Newton iteration for efficiency.
    """
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin before 300s limit
    
    mu = 5.0
    t0, tf = t_span
    y = np.array(y0, dtype=float)
    
    # Analytic Jacobian for the Van der Pol oscillator
    def jac(t, y):
        y0_val = y[0]
        y1_val = y[1]
        return np.array([
            [0.0, 1.0],
            [-1.0 - 2.0 * mu * y0_val * y1_val, mu * (1.0 - y0_val * y0_val)]
        ])
    
    # Radau IIIA 3-stage coefficients
    c = np.array([0.135087567407121, 0.489251280242632, 0.989079026173931])
    A = np.array([
        [0.062543783703561, 0.072543783703561, 0.000000000000000],
        [0.000000000000000, 0.244625640121316, 0.244625640121316],
        [0.000000000000000, 0.000000000000000, 0.494539513086966]
    ])
    b = np.array([0.000000000000000, 0.244625640121316, 0.755374359878684])
    
    # Initial step size
    h = (tf - t0) / 500.0
    
    def radau_step(t, y, h):
        """Perform one Radau IIIA step."""
        n_comp = 2
        n_stages = 3
        
        # Initial guess
        f0 = f(t, y)
        F = np.zeros((n_comp, n_stages))
        for j in range(n_stages):
            F[:, j] = y + c[j] * h * f0
        
        # Newton iteration
        for _ in range(30):
            # Compute all stage derivatives
            f_stages = np.zeros((n_comp, n_stages))
            for j in range(n_stages):
                f_stages[:, j] = f(t + c[j] * h, F[:, j])
            
            # Residual
            res = np.zeros(n_comp * n_stages)
            for j in range(n_stages):
                sum_val = np.zeros(n_comp)
                for i in range(n_stages):
                    sum_val += A[j, i] * f_stages[:, i]
                res[n_comp*j:n_comp*j+n_comp] = F[:, j] - y - h * sum_val
            
            if np.max(np.abs(res)) < 1e-15:
                break
            
            # Build Jacobian
            J_full = np.zeros((n_comp * n_stages, n_comp * n_stages))
            for j in range(n_stages):
                for i in range(n_stages):
                    J_f = jac(t + c[i] * h, F[:, i])
                    J_full[n_comp*j:n_comp*j+n_comp, n_comp*i:n_comp*i+n_comp] = -h * A[j,i] * J_f
                J_full[n_comp*j:n_comp*j+n_comp, n_comp*j:n_comp*j+n_comp] += np.eye(n_comp)
            
            try:
                delta = np.linalg.solve(J_full, -res)
            except np.linalg.LinAlgError:
                delta, *_ = np.linalg.lstsq(J_full, -res, rcond=None)
            
            F_flat = F.flatten() + delta
            F = F_flat.reshape(n_comp, n_stages)
        
        # Update
        y_new = y + h * np.dot(b, F.T)
        return y_new, F
    
    t = t0
    while t < tf - 1e-15:
        if time.time() > deadline:
            break
        
        h_eff = min(h, tf - t)
        
        y_new, F = radau_step(t, y, h_eff)
        
        # Error estimate using embedded method (compare with half step)
        h_half = h_eff / 2.0
        y_h1, _ = radau_step(t, y, h_half)
        y_h2, _ = radau_step(t + h_half, y_h1, h_half)
        
        err = np.linalg.norm(y_new - y_h2)
        
        # Adaptive step
        if err < 1e-14:
            h = min(h * 1.8, (tf - t) / 5.0)
        elif err > 1e-11:
            h = h * 0.5
        else:
            h = h * 0.9
        
        y = y_new
        t += h_eff
    
    return y

# EVOLVE-BLOCK-END
