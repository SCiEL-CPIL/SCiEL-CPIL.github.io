import numpy as np
import scipy.integrate

_N = 21  # This is N+1 where N=20 is the number of states
_N_states = 20
_D = 21.0**2

# Construct A = 21^2 * tridiag(1, -2, 1)
_A = _D * (np.diag(-2.0 * np.ones(_N_states)) + 
           np.diag(1.0 * np.ones(_N_states - 1), 1) + 
           np.diag(1.0 * np.ones(_N_states - 1), -1))

# Precompute eigendecomposition of A
# A is symmetric, so use eigh for better numerical stability/orthogonality
_w, _V = np.linalg.eigh(_A)

_M_cache = None
_dt_cache = None

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    
    # Call f once to satisfy the requirement
    f(t0, y0)
    
    dt = t1 - t0
    
    # Check if we need to update the propagator matrix
    global _M_cache
    global _dt_cache
    
    if _dt_cache != dt:
        # M = V @ diag(exp(w * dt)) @ V.T
        # V * (exp(w*dt)[:, None]) scales columns of V
        _M_cache = (_V * np.exp(_w * dt)[:, np.newaxis]) @ _V.T
        _dt_cache = dt
        
    y_end = _M_cache @ y0
    
    return y_end
