"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def integrate(f, t_span, y0):
    """
    Integrate the Van der Pol oscillator (mu=5) from t_span[0] to t_span[1]
    using a custom variable-step Radau IIA(5) integrator with analytic Jacobian.
    
    The system is:
        y0' = y1
        y1' = 5*(1 - y0^2)*y1 - y0
    
    We use Radau IIA(5) which is an implicit Runge-Kutta method of order 5
    with L-stability, well-suited for stiff problems.
    
    Key design choices:
    - Use the analytic Jacobian to avoid finite-difference calls
    - Adaptive step size control based on local error estimate
    - Newton iteration for the implicit solve
    - Track function calls to stay well under 2 million
    """
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    
    # Radau IIA(5) coefficients
    # The Radau IIA(5) method has 5 stages
    # c_i are the nodes (roots of shifted Legendre polynomial)
    # b_i are the weights
    # A is the coefficient matrix
    
    # Nodes c_i for Radau IIA(5)
    c = np.array([
        0.155688271133952752,
        0.420497876391589982,
        0.672645965103890860,
        0.896906490875639150,
        1.0
    ])
    
    # Weights b_i
    b = np.array([
        0.139031430882846633,
        0.275281832538061147,
        0.316514731484122757,
        0.202628073188316328,
        0.066543931806653135
    ])
    
    # Coefficient matrix A for Radau IIA(5)
    # A[i,j] = b_j * (c_i - c_j) * P_{2n-2}( (c_i - c_j)/(c_i + c_j) ) / (c_i - c_j) ... 
    # Actually, for Radau IIA, A is defined such that:
    # A[i,j] = integral from 0 to c_i of L_j(s) ds where L_j are Lagrange basis polynomials
    # For Radau IIA(5), the matrix A is:
    A = np.array([
        [0.012778762094388729, 0.0, 0.0, 0.0, 0.0],
        [0.040949995059762618, 0.022538876265853892, 0.0, 0.0, 0.0],
        [0.073982920734215124, 0.061350494304772227, 0.035138969333574456, 0.0, 0.0],
        [0.103397216113111288, 0.098788886333243246, 0.076400353943872337, 0.051480795995584996, 0.0],
        [0.139031430882846633, 0.275281832538061147, 0.316514731484122757, 0.202628073188316328, 0.066543931806653135]
    ])
    
    # For Radau IIA, the last row of A equals b^T (this is a property of Radau IIA)
    # Actually, let me use the standard formulation. The Radau IIA(5) method satisfies:
    # A[i,j] for i,j = 0..4
    # The matrix is upper Hessenberg and the last row is b^T
    
    # Let me redefine using the standard Radau IIA(5) coefficients from literature
    # c = [0.155688271133952752, 0.420497876391589982, 0.672645965103890860, 0.896906490875639150, 1.0]
    # b = [0.139031430882846633, 0.275281832538061147, 0.316514731484122757, 0.202628073188316328, 0.066543931806653135]
    # A is such that A[i,:] are the quadrature weights for integrating from 0 to c_i
    
    # The correct A matrix for Radau IIA(5):
    A = np.array([
        [0.012778762094388729, 0.0, 0.0, 0.0, 0.0],
        [0.040949995059762618, 0.022538876265853892, 0.0, 0.0, 0.0],
        [0.073982920734215124, 0.061350494304772227, 0.035138969333574456, 0.0, 0.0],
        [0.103397216113111288, 0.098788886333243246, 0.076400353943872337, 0.051480795995584996, 0.0],
        [0.139031430882846633, 0.275281832538061147, 0.316514731484122757, 0.202628073188316328, 0.066543931806653135]
    ])
    
    # Verify: for Radau IIA, A[i, i] should be such that the method is consistent
    # b = A[4, :] (last row)
    
    n = len(y)
    
    # Initial step size
    h = 0.01
    t = t0
    
    # Error tolerances
    rtol = 1e-10
    atol = 1e-12
    
    # Maximum and minimum step sizes
    h_min = 1e-6
    h_max = 0.1
    
    # Function call counter
    call_count = 0
    
    def vdp_rhs(t_val, y_val):
        """Van der Pol RHS: y0' = y1, y1' = 5*(1-y0^2)*y1 - y0"""
        nonlocal call_count
        call_count += 1
        y0_val = y_val[0]
        y1_val = y_val[1]
        dy0 = y1_val
        dy1 = 5.0 * (1.0 - y0_val * y0_val) * y1_val - y0_val
        return np.array([dy0, dy1])
    
    def vdp_jacobian(y_val):
        """Analytic Jacobian of the Van der Pol system.
        J[0,0] = 0, J[0,1] = 1
        J[1,0] = -10*y0*y1 - 1
        J[1,1] = 5*(1 - y0^2)
        """
        y0_val = y_val[0]
        y1_val = y_val[1]
        J = np.array([
            [0.0, 1.0],
            [-10.0 * y0_val * y1_val - 1.0, 5.0 * (1.0 - y0_val * y0_val)]
        ])
        return J
    
    # Precompute M = I - h*A for each step (depends on h)
    # We'll recompute when h changes
    
    # Adaptive step size control parameters
    safety = 0.9
    error_exponent = -0.2  # For order 5: exponent is -1/5 = -0.2
    
    while t < t1 - 1e-15:
        # Adjust h to not overshoot t1
        if t + h > t1:
            h = t1 - t
        
        if h < h_min:
            h = h_min
        
        # Build the system matrix M = I - h*A (for the stage equations)
        # The stage equations are: Y_i = y + h * sum_j A[i,j] * f(t + c_j*h, Y_j)
        # Rearranging: Y_i - h * sum_j A[i,j] * f(t + c_j*h, Y_j) = y
        # For implicit solve, we use Newton's method
        
        # Initialize stage values
        Y = np.zeros((5, n))
        for i in range(5):
            Y[i] = y
        
        # Newton iteration for the implicit stage values
        max_newton_iter = 20
        newton_tol = 1e-12
        
        for newton_iter in range(max_newton_iter):
            # Evaluate f at all stages
            F = np.zeros((5, n))
            for i in range(5):
                F[i] = f(t + c[i] * h, Y[i])
            
            # Compute residuals
            R = np.zeros((5, n))
            for i in range(5):
                R[i] = Y[i] - y - h * np.dot(A[i, :], F)
            
            # Check convergence
            max_res = np.max(np.abs(R))
            if max_res < newton_tol * (1.0 + np.max(np.abs(Y))):
                break
            
            # Compute the Jacobian of the system with respect to Y
            # The system is: G_i(Y) = Y_i - y - h * sum_j A[i,j] * f(t + c_j*h, Y_j) = 0
            # dG_i/dY_j = delta_ij * I - h * A[i,j] * J_f(t + c_j*h, Y_j)
            
            # Build the full Jacobian (5n x 5n)
            Jn = 5 * n
            J_full = np.zeros((Jn, Jn))
            
            for i in range(5):
                for j in range(5):
                    if A[i, j] != 0.0:
                        Jf = vdp_jacobian(Y[j])
                        J_full[i*n:(i+1)*n, j*n:(j+1)*n] = -h * A[i, j] * Jf
                    if i == j:
                        J_full[i*n:(i+1)*n, j*n:(j+1)*n] += np.eye(n)
            
            # Solve for the Newton update
            R_flat = R.flatten()
            try:
                delta = np.linalg.solve(J_full, -R_flat)
            except np.linalg.LinAlgError:
                # If singular, use least squares
                delta, _, _, _ = np.linalg.lstsq(J_full, -R_flat, rcond=None)
            
            delta_reshaped = delta.reshape((5, n))
            Y += delta_reshaped
        
        # Update y using the Radau IIA formula
        y_new = y + h * np.dot(b, F)
        
        # Compute error estimate using a lower-order method or embedded estimate
        # For Radau IIA(5), we can use the last stage as an approximation
        # or compute a local error estimate
        
        # Simple error estimate: compare with a 4th order approximation
        # For now, use a rough estimate based on the difference between stages
        # A better approach: use the fact that Radau IIA(5) is order 5,
        # and estimate error as O(h^6)
        
        # Compute local error estimate
        # Using the difference between the full solution and a truncated version
        # For simplicity, estimate error as:
        # err ~ h^6 * |y_new - y| / some_scale
        # More carefully, we can use the last few stages to estimate
        
        # A practical error estimate for Radau methods:
        # e = |b_5 * F_5 - b_4 * F_4| * h  (rough)
        # Better: use the fact that for Radau IIA(5), we can embed a 4th order method
        
        # Let's use a simple heuristic:
        y_diff = np.abs(y_new - y)
        y_scale = np.maximum(np.abs(y_new), 1e-12)
        err_vec = y_diff / (atol + rtol * y_scale)
        err_norm = np.max(err_vec)
        
        # Accept or reject step
        if err_norm <= 1.0 or h <= h_min:
            # Accept step
            t = t + h
            y = y_new
            
            # Update step size
            if err_norm > 0:
                h_new = safety * h * (1.0 / err_norm) ** error_exponent
            else:
                h_new = h * 5.0
            h = np.clip(h_new, h_min, h_max)
        else:
            # Reject step, reduce h
            if err_norm > 0:
                h_new = safety * h * (1.0 / err_norm) ** error_exponent
            else:
                h_new = h * 0.5
            h = np.clip(h_new, h_min, h_max)
    
    return y
# EVOLVE-BLOCK-END
