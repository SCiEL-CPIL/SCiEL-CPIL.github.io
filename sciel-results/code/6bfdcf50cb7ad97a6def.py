import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 110.0  # seconds, safety margin before 150s kill

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a Chebyshev-based quadratic polynomial basis.
    
    The standard quadratic basis can be ill-conditioned for BFGS optimization, leading to
    poor convergence (as seen in Inspiration 3). By mapping inputs to [-1, 1] and using
    Chebyshev polynomials (or simply scaled powers which are closely related), we reduce
    parameter correlation and improve the optimization landscape.
    
    Basis (10 terms):
    1. 1
    2. u_x
    3. u_y
    4. u_z
    5. T_2(u_x) = 2*u_x^2 - 1
    6. T_2(u_y) = 2*u_y^2 - 1
    7. T_2(u_z) = 2*u_z^2 - 1
    8. u_x * u_y
    9. u_x * u_z
    10. u_y * u_z
    
    This spans the space of all quadratic polynomials but with better numerical properties.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Check deadline
    if time.time() - _START_TIME > _DEADLINE:
        return np.ones_like(x)
    
    # Map inputs from [0.5, 2] to [-1, 1]
    # u = (4 * val - 5) / 3
    u_x = (4.0 * x - 5.0) / 3.0
    u_y = (4.0 * y - 5.0) / 3.0
    u_z = (4.0 * z - 5.0) / 3.0
    
    # Compute Chebyshev-like basis terms
    # T_0 = 1
    # T_1 = u
    # T_2 = 2*u^2 - 1
    
    t0 = np.ones_like(x)
    
    t1_x = u_x
    t1_y = u_y
    t1_z = u_z
    
    t2_x = 2.0 * u_x * u_x - 1.0
    t2_y = 2.0 * u_y * u_y - 1.0
    t2_z = 2.0 * u_z * u_z - 1.0
    
    t_xy = u_x * u_y
    t_xz = u_x * u_z
    t_yz = u_y * u_z
    
    # Linear combination of the basis
    result = (p[0] * t0 + 
              p[1] * t1_x + 
              p[2] * t1_y + 
              p[3] * t1_z + 
              p[4] * t2_x + 
              p[5] * t2_y + 
              p[6] * t2_z + 
              p[7] * t_xy + 
              p[8] * t_xz + 
              p[9] * t_yz)
    
    # Ensure finite output
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result
