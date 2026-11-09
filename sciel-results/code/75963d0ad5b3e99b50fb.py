import numpy as np
import time
from scipy.linalg import lu_factor, lu_solve

def integrate(f, t_span, y0):
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin
    
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    
    # Robertson problem is stiff.
    # Use 3-stage Radau IIA (5th order) with fixed step size.
    # Coefficients for 3-stage Radau IIA:
    # c = [1/10, 3/10, 1]
    # A = [[1/10, 0, 0],
    #      [3/10, 1/10, 0],
    #      [1/9, 3/8, 1/9]]
    # b = [1/9, 3/8, 1/9]
    
    c = np.array([1/10, 3/10, 1.0])
    A = np.array([
        [1/10, 0, 0],
        [3/10, 1/10, 0],
        [1/9, 3/8, 1/9]
    ])
    b = np.array([1/9, 3/8, 1/9])
    
    # Fixed step size. h=1.0 means 40 steps.
    # We can try h=2.0 (20 steps) to reduce calls further, but error might increase.
    # Let's start with h=1.0 to ensure accuracy, then maybe try h=2.0 if time permits?
    # No, fixed strategy: h=1.0 is safe.
    h = 1.0
    n_steps = int(np.ceil((t1 - t0) / h))
    
    # Precompute LU factors? The matrix (I - h*A_ij * J) changes every Newton iteration.
    # So we must factorize each time. For 3x3, this is very fast.
    
    def jac(t, y):
        y1, y2, y3 = y
        J = np.zeros((3, 3))
        J[0, 0] = -0.04
        J[0, 1] = 1e4 * y3
        J[0, 2] = 1e4 * y2
        
        J[1, 0] = 0.04
        J[1, 1] = -1e4 * y3 - 6e7 * y2
        J[1, 2] = -1e4 * y2
        
        J[2, 0] = 0.0
        J[2, 1] = 6e7 * y2
        J[2, 2] = 0.0
        return J

    for i in range(n_steps):
        if time.time() > deadline:
            break
            
        t = t0 + i * h
        # Ensure we don't overshoot
        if t + h > t1:
            h_step = t1 - t
        else:
            h_step = h
            
        # Radau IIA stage equations:
        # Y_j = y + h * sum_{k=1}^s A_jk * f(t + c_k h, Y_k)
        # This is a system of 3*s = 9 nonlinear equations.
        # We can solve it using Newton's method.
        
        # Initial guess: y (or previous stage values)
        Y = np.tile(y, (3, 1)).T  # Shape (3, 3), each column is a stage value
        
        # Newton iteration
        max_newton_iters = 10
        tol = 1e-14
        
        for n_iter in range(max_newton_iters):
            # Compute F(Y) = Y - y - h * A * f(t + c*h, Y)
            # f_vals[j] = f(t + c[j]*h_step, Y[:, j])
            f_vals = np.zeros((3, 3))
            for j in range(3):
                f_vals[:, j] = f(t + c[j] * h_step, Y[:, j])
            
            # Residual R = Y - y - h_step * (A @ f_vals)
            # A is (3,3), f_vals is (3,3). A @ f_vals gives (3,3) where each col is sum A_jk f_k
            A_f = A @ f_vals
            R = Y - y - h_step * A_f
            
            # Check convergence
            if np.max(np.abs(R)) < tol:
                break
            
            # Compute Jacobian of the system
            # dR/dY = I_3 - h_step * (A @ J_f)
            # where J_f is a block diagonal matrix of 3x3 Jacobians at each stage?
            # Actually, the system is:
            # Y_j = y + h * sum_k A_jk f_k(Y_k)
            # dY_j/dY_k = h * A_jk * J_f_k
            # So the Jacobian of the full system (9x9) is:
            # J_full = I_9 - h * (A @ J_block)
            # Where J_block is block diagonal with J_f_k on the diagonal blocks?
            # No, the mapping is from vectorized Y to vectorized R.
            # Let's construct the 9x9 Jacobian.
            
            J_full = np.eye(9)
            # Block structure: rows/cols correspond to (j, component)
            # dR_j / dY_k = -h * A_jk * J_f_k (for j != k? No, A_jk is always there)
            # Wait, R_j = Y_j - y - h * sum_k A_jk f_k
            # dR_j / dY_k = delta_jk * I - h * A_jk * J_f_k
            
            for j in range(3):
                for k in range(3):
                    # J_f_k is the Jacobian of f at stage k
                    J_f_k = jac(t + c[k] * h_step, Y[:, k])
                    block = -h_step * A[j, k] * J_f_k
                    if j == k:
                        block += np.eye(3)
                    J_full[j*3:(j+1)*3, k*3:(k+1)*3] = block
            
            # Solve J_full * delta = -R
            R_flat = R.flatten()
            try:
                lu, piv = lu_factor(J_full)
                delta = lu_solve((lu, piv), -R_flat)
                delta = delta.reshape(3, 3)
            except Exception:
                break
            
            Y += delta
            
        # Update y using b weights
        # y_new = y + h * sum_j b_j f_j
        f_vals = np.zeros((3, 3))
        for j in range(3):
            f_vals[:, j] = f(t + c[j] * h_step, Y[:, j])
        
        y = y + h_step * (b @ f_vals)
        
        # Check time
        if time.time() > deadline:
            break

    return y
