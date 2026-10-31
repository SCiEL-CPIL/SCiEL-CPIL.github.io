import numpy as np
import time
from scipy.optimize import minimize, differential_evolution, root

_DEADLINE = time.time() + 200

# Precompute I[n] = integral_{-inf}^{inf} x^{2n} exp(-x^2) dx
I = np.zeros(30)
I[0] = np.sqrt(np.pi)
for n in range(1, 30):
    I[n] = I[n-1] * (2.0*n - 1.0) / (2.0*n)

# Hermite polynomial coefficients in power basis x^{2k}
H0_c = np.array([1, 0, 0, 0, 0, 0, 0])
H4_c = np.array([3, -6, 1, 0, 0, 0, 0])
H8_c = np.array([105, -420, 210, -28, 1, 0, 0])
H12_c = np.array([945, -7560, 13860, -6930, 1155, -66, 1])

# a12 = -(a0 + 3*a4 + 105*a8) / 945
c12 = np.array([-1.0/945.0, -3.0/945.0, -105.0/945.0])

col0 = H0_c + c12[0]*H12_c
col1 = H4_c + c12[1]*H12_c
col2 = H8_c + c12[2]*H12_c

A_mat = np.column_stack((col0, col1, col2))

signs = np.array([1, -1, 1, -1, 1, -1, 1])

def _get_p(a):
    return A_mat @ a

def _compute_bound_and_grad(a):
    """Compute the uncertainty bound and its gradient for given coefficients a = [a0, a4, a8]."""
    p = _get_p(a)
    q = p * signs
    
    # Compute moments
    Mx0 = 0.0
    Mx1 = 0.0
    Mxi0 = 0.0
    Mxi1 = 0.0
    
    for i in range(7):
        pi = p[i]
        qi = q[i]
        if pi == 0 and qi == 0:
            continue
        for j in range(7):
            Mx0 += pi * p[j] * I[i+j]
            Mx1 += pi * p[j] * I[i+j+1]
            Mxi0 += qi * q[j] * I[i+j]
            Mxi1 += qi * q[j] * I[i+j+1]
            
    if Mx0 <= 1e-15 or Mxi0 <= 1e-15:
        return 1e18, np.zeros(3)
        
    vx = Mx1 / Mx0
    vxi = Mxi1 / Mxi0
    
    if vx <= 1e-15 or vxi <= 1e-15:
        return 1e18, np.zeros(3)
        
    bound = np.sqrt(vx * vxi)
    
    if not np.isfinite(bound):
        return 1e18, np.zeros(3)
    
    # Compute gradients
    Sx = np.zeros((2, 7))
    Sxi = np.zeros((2, 7))
    
    for n in range(2):
        for i in range(7):
            Sx[n, i] = np.dot(p, I[i+n:i+n+7])
            Sxi[n, i] = np.dot(q, I[i+n:i+n+7])
            
    dMx = np.zeros((2, 3))
    dMxi = np.zeros((2, 3))
    
    for m in range(3):
        for n in range(2):
            dMx[n, m] = 2.0 * np.dot(A_mat[:, m], Sx[n, :])
            dMxi[n, m] = 2.0 * np.dot(A_mat[:, m], Sxi[n, :])
            
    grad_vx = np.zeros(3)
    grad_vxi = np.zeros(3)
    
    for m in range(3):
        grad_vx[m] = (dMx[1, m]*Mx0 - Mx1*dMx[0, m]) / (Mx0*Mx0)
        grad_vxi[m] = (dMxi[1, m]*Mxi0 - Mxi1*dMxi[0, m]) / (Mxi0*Mxi0)
        
    grad_bound = (vxi * grad_vx + vx * grad_vxi) / (2.0 * bound)
    
    return bound, grad_bound

def construct():
    """
    Find three coefficients (a0, a4, a8) that minimize the uncertainty product.
    Returns array of shape (3,).
    """
    
    seed_params = np.array([-1.13, 0.04, 0.000308642])
    seed_val, _ = _compute_bound_and_grad(seed_params)
    
    best_val = seed_val
    best_params = seed_params.copy()
    
    deadline_margin = 15  # seconds
    
    # Define bounds for the coefficients. 
    # Based on the seed and typical scales, we can search a reasonable range.
    # a0 is around -1.13, a4 around 0.04, a8 around 0.0003.
    bounds = [(-2.0, 0.0), (-0.1, 0.1), (-0.001, 0.001)]
    
    # Phase 1: Global Optimization using Differential Evolution
    # This helps escape local minima which L-BFGS-B might get stuck in.
    if time.time() < _DEADLINE - deadline_margin - 30:
        try:
            # Use a relatively small population for speed, but enough to explore.
            # maxiter is controlled by time.
            de_result = differential_evolution(
                lambda x: _compute_bound_and_grad(x)[0], 
                bounds, 
                seed=42, 
                maxiter=50, 
                popsize=15,
                tol=1e-10,
                polish=False # We will polish manually
            )
            if np.all(np.isfinite(de_result.x)):
                val, _ = _compute_bound_and_grad(de_result.x)
                if val < best_val:
                    best_val = val
                    best_params = de_result.x.copy()
        except Exception:
            pass

    # Phase 2: Trust-Region Polish
    # Use 'trust-ncg' which uses Hessian information (or approximation) and is a trust-region method.
    # This aligns with the directive's suggestion of a trust-region subproblem.
    if time.time() < _DEADLINE - deadline_margin:
        try:
            res = minimize(
                lambda x: _compute_bound_and_grad(x)[0], 
                best_params, 
                jac=lambda x: _compute_bound_and_grad(x)[1],
                method='trust-ncg',
                options={'maxiter': 100, 'gtol': 1e-10}
            )
            if np.all(np.isfinite(res.x)):
                val, _ = _compute_bound_and_grad(res.x)
                if val < best_val:
                    best_val = val
                    best_params = res.x.copy()
        except Exception:
            pass

    # Phase 3: Final L-BFGS-B polish for high precision
    if time.time() < _DEADLINE - deadline_margin:
        try:
            res = minimize(
                lambda x: _compute_bound_and_grad(x)[0], 
                best_params, 
                jac=lambda x: _compute_bound_and_grad(x)[1],
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            if np.all(np.isfinite(res.x)):
                val, _ = _compute_bound_and_grad(res.x)
                if val < best_val:
                    best_val = val
                    best_params = res.x.copy()
        except Exception:
            pass

    if not np.all(np.isfinite(best_params)):
        return seed_params

    return np.array(best_params)
