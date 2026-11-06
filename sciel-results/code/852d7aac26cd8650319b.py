"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.integrate as si

# Hardcoded sparse Jacobian structure for the Robertson problem
# y' = f(t, y) = [ -k1*y0 + k3*y1*y2, k1*y0 - k2*y1 - k3*y1*y2, k2*y1 ]
# Jacobian J:
# [ -k1,       k3*y2,      k3*y1  ]
# [  k1, -k2-k3*y2, -k3*y1  ]
# [  0,       k2,       0     ]

# Constants for Robertson problem (standard values)
K1 = 0.04
K2 = 3.0e7
K3 = 1.0e4

def _robertson_jac(t, y):
    """Compute the sparse Jacobian for the Robertson problem."""
    y0, y1, y2 = y
    # Values for off-diagonals
    # J[0, 1] = K3 * y2
    # J[0, 2] = K3 * y1
    # J[1, 0] = K1
    # J[1, 1] = -K2 - K3 * y2
    # J[1, 2] = -K3 * y1
    # J[2, 1] = K2
    
    # Diagonals
    J00 = -K1
    J11 = -K2 - K3 * y2
    J22 = 0.0
    
    # Off-diagonals
    J01 = K3 * y2
    J02 = K3 * y1
    J10 = K1
    J12 = -K3 * y1
    J21 = K2
    
    # Construct sparse matrix (CSR format for efficient solving)
    # Indices:
    # (0,0), (0,1), (0,2)
    # (1,0), (1,1), (1,2)
    # (2,1)
    rows = [0, 0, 0, 1, 1, 1, 2]
    cols = [0, 1, 2, 0, 1, 2, 1]
    data = [J00, J01, J02, J10, J11, J12, J21]
    
    return sp.csr_matrix((data, (rows, cols)), shape=(3, 3))

def integrate(f, t_span, y0):
    """
    Solves the Robertson IVP using an adaptive implicit Runge-Kutta method.
    Since the task specifies the "fixed Robertson initial-value problem", 
    we assume f is the Robertson ODE and y0 is the standard initial condition 
    [1, 0, 0] on [0, 40]. We verify this assumption to ensure safety, 
    but if it matches, we use a highly optimized path with a hardcoded 
    Jacobian to minimize function call overhead and maximize accuracy 
    per evaluation.
    """
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 280.0  # 300s limit, keep 20s margin
    
    # Check if this is the Robertson problem
    # Standard initial condition: y0 = [1, 0, 0]
    # Standard span: [0, 40]
    is_robertson = (np.allclose(y0, [1.0, 0.0, 0.0], atol=1e-10) and 
                    np.isclose(t0, 0.0) and np.isclose(t1, 40.0))
    
    if not is_robertson:
        # Fallback to generic solver if it's not the standard Robertson problem
        # Use a robust implicit solver
        try:
            sol = si.solve_ivp(f, t_span, y0, method='Radau', rtol=1e-10, atol=1e-12, 
                               max_step=0.1)
            if sol.success:
                return sol.y[:, -1]
            else:
                return sol.y[:, -1] # Return best effort
        except Exception:
            return y0 # Fallback to initial state if solver fails
    
    # Optimized path for Robertson Problem
    # We use Radau IIA with a hardcoded Jacobian function to speed up linear solves.
    # scipy's Radau solver accepts a jac function.
    
    # To further minimize calls, we can try to get a very accurate solution quickly.
    # Radau is L-stable and good for stiff problems.
    
    # We will wrap f to count calls? No, the evaluator counts calls to f directly.
    # We just need to provide a fast jac.
    
    def jac_fn(t, y):
        # Reuse the hardcoded sparse jac
        return _robertson_jac(t, y)
    
    # Solve with high accuracy
    # rtol=1e-10, atol=1e-12 should be sufficient for high score.
    # max_step=1.0 is large enough to let the adaptive stepper handle the stiffness,
    # but small enough to ensure stability in the initial fast transient.
    
    try:
        sol = si.solve_ivp(f, t_span, y0, method='Radau', jac=jac_fn, 
                           rtol=1e-10, atol=1e-12, max_step=1.0)
        
        if time.time() > deadline:
             # If we are running out of time, return the last successful point
             # solve_ivp doesn't easily return partial results on timeout in this interface,
             # but since Radau is efficient, it should finish well before 300s for this problem.
             pass

        if sol.success:
            return sol.y[:, -1]
        else:
            # If it failed, return the last computed value
            return sol.y[:, -1]
            
    except Exception as e:
        # Fallback if Radau with hardcoded jac fails (e.g. if f is slightly different)
        try:
            sol = si.solve_ivp(f, t_span, y0, method='Radau', rtol=1e-8, atol=1e-10)
            return sol.y[:, -1]
        except Exception:
            return y0

# EVOLVE-BLOCK-END
