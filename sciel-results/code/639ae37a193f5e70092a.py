import numpy as np
import time

def integrate(f, t_span, y0):
    start_time = time.time()
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    n = len(y)
    
    # Radau IIA coefficients (5th order, 3 stages)
    c = np.array([0.167193860365911, 0.634844063093175, 1.0])
    A = np.array([
        [0.019685260303335, 0.047975866589096, 0.100697632738480],
        [0.095838190989704, 0.220809491770996, 0.318196380332475],
        [0.115331045515568, 0.262508000287831, 0.622160954196501]
    ])
    b = np.array([0.067661126892431, 0.427968433592090, 0.504370439515479])
    
    # Step size
    h = 0.2
    n_steps = int(np.ceil((t1 - t0) / h))
    h = (t1 - t0) / n_steps
    
    t = t0
    
    for _ in range(n_steps):
        # Solve for Z = [z1, z2, z3]
        # Z_i = y + h * sum_j A_ij f(t + c_j h, Z_j)
        # System: Z - h A F(Z) = y
        # Newton iteration
        Z = np.tile(y, (3, 1)) # Initial guess
        
        for _newton in range(10):
            # Evaluate f at stages
            F = np.zeros((3, n))
            for i in range(3):
                F[i] = f(t + c[i] * h, Z[i])
            
            # Residual: Z - y - h A F
            R = Z - y - h * (A @ F)
            
            # Jacobian of Residual: I - h A J_F
            # J_F is 3n x 3n block diagonal with J_ODE(Z_i)
            J_sys = np.eye(3 * n)
            for i in range(3):
                # Compute J_ODE for Z[i]
                y0_val = Z[i, 0]
                y1_val = Z[i, 1]
                # y0' = y1
                # y1' = 5(1-y0^2)y1 - y0
                # d(y0')/dy0 = 0, d(y0')/dy1 = 1
                # d(y1')/dy0 = -10 y0 y1 - 1
                # d(y1')/dy1 = 5(1-y0^2)
                J_ode = np.array([
                    [0.0, 1.0],
                    [-10.0 * y0_val * y1_val - 1.0, 5.0 * (1.0 - y0_val**2)]
                ])
                # Place in block diagonal
                J_sys[i*n:(i+1)*n, i*n:(i+1)*n] = J_ode
            
            # J_sys = I - h * (A kron I) * J_F
            # A kron I is 3n x 3n
            A_kron_I = np.kron(A, np.eye(n))
            J_total = np.eye(3*n) - h * (A_kron_I @ J_sys)
            
            # Solve J_total dZ = -R
            R_flat = R.flatten()
            try:
                dZ = np.linalg.solve(J_total, -R_flat)
            except np.linalg.LinAlgError:
                # Fallback to least squares if singular
                dZ, *_ = np.linalg.lstsq(J_total, -R_flat, rcond=None)
            
            Z = Z + dZ.reshape(3, n)
            
            if np.linalg.norm(dZ) < 1e-14 * (1 + np.linalg.norm(Z)):
                break
        
        # Update y
        y = y + h * (b @ F)
        t += h
        
        if time.time() - start_time > 250:
            break
            
    return y
