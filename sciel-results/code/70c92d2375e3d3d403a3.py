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
    
    # Strategy: Use spherical coordinates on the constraint plane g(0)=0
    # The constraint is a0 + 3*a4 + 105*a8 = 0 (since a12 is determined)
    # This is a plane through the origin. We parameterize points on this plane
    # using two angles (theta, phi) and a radius r.
    
    # Basis for the plane a0 + 3*a4 + 105*a8 = 0
    # v1 = [3, -1, 0] (dot with [1,3,105] = 3-3+0 = 0)
    # v2 = [105, 0, -1] (dot with [1,3,105] = 105+0-105 = 0)
    # Orthonormalize
    v1 = np.array([3.0, -1.0, 0.0])
    v2 = np.array([105.0, 0.0, -1.0])
    
    v1_norm = np.linalg.norm(v1)
    v1_unit = v1 / v1_norm
    
    v2_proj = v2 - np.dot(v2, v1_unit) * v1_unit
    v2_norm = np.linalg.norm(v2_proj)
    v2_unit = v2_proj / v2_norm
    
    def _spherical_to_params(theta, phi, r):
        """Convert spherical coordinates to (a0, a4, a8) on the constraint plane."""
        # theta in [0, pi], phi in [0, 2*pi]
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        return x * v1_unit + y * v2_unit
    
    def _spherical_obj(params):
        """Objective in spherical coordinates: [theta, phi, r]."""
        theta, phi, r = params
        a = _spherical_to_params(theta, phi, r)
        val, _ = _compute_bound_and_grad(a)
        return val
    
    # Generate initial guesses in spherical coordinates
    rng = np.random.default_rng(42)
    
    # Start near the seed
    seed_theta = np.arccos(np.clip(np.dot(seed_params, v1_unit) / (np.linalg.norm(seed_params) + 1e-300), -1, 1))
    seed_phi = np.arctan2(np.dot(seed_params, v2_unit), np.dot(seed_params, v1_unit))
    seed_r = np.linalg.norm(seed_params)
    
    initial_spherical = [np.array([seed_theta, seed_phi, seed_r])]
    
    # Add perturbations
    for scale in [0.01, 0.05, 0.1, 0.5]:
        for _ in range(5):
            theta = seed_theta + rng.normal(0, scale)
            phi = seed_phi + rng.normal(0, scale)
            r = seed_r * np.exp(rng.normal(0, scale/3))
            if 0 < theta < np.pi and r > 0:
                initial_spherical.append(np.array([theta, phi, r]))
    
    # Add some random points
    for _ in range(20):
        theta = rng.uniform(0.1, np.pi - 0.1)
        phi = rng.uniform(-np.pi, np.pi)
        r = rng.uniform(0.1, 5.0)
        initial_spherical.append(np.array([theta, phi, r]))
    
    # Phase 1: Optimize in spherical coordinates
    for start in initial_spherical:
        if time.time() > _DEADLINE - deadline_margin:
            break
        
        try:
            res = minimize(_spherical_obj, start, method='L-BFGS-B',
                          bounds=[(1e-6, np.pi-1e-6), (-np.pi, np.pi), (1e-6, 20.0)],
                          options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12})
            if res.success and np.all(np.isfinite(res.x)):
                a = _spherical_to_params(*res.x)
                val, _ = _compute_bound_and_grad(a)
                if val < best_val:
                    best_val = val
                    best_params = a.copy()
        except Exception:
            pass
    
    # Phase 2: Gradient-based minimization from best point in original coordinates
    if time.time() < _DEADLINE - deadline_margin:
        try:
            res = minimize(_compute_bound_and_grad, best_params, method='L-BFGS-B',
                          jac=True, options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-13})
            if res.success and np.all(np.isfinite(res.x)):
                val, _ = _compute_bound_and_grad(res.x)
                if val < best_val:
                    best_val = val
                    best_params = res.x.copy()
        except Exception:
            pass
    
    # Phase 3: Local search with Nelder-Mead
    if time.time() < _DEADLINE - deadline_margin - 5:
        try:
            res = minimize(_spherical_obj, start, method='Nelder-Mead',
                          bounds=[(1e-6, np.pi-1e-6), (-np.pi, np.pi), (1e-6, 20.0)],
                          options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-14})
            if np.all(np.isfinite(res.x)):
                a = _spherical_to_params(*res.x)
                val, _ = _compute_bound_and_grad(a)
                if val < best_val:
                    best_val = val
                    best_params = a.copy()
        except Exception:
            pass
    
    # Phase 4: Final polish with tight tolerances
    if time.time() < _DEADLINE - deadline_margin - 2:
        try:
            res = minimize(_compute_bound_and_grad, best_params, method='L-BFGS-B',
                          jac=True, options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-14})
            if res.success and np.all(np.isfinite(res.x)):
                val, _ = _compute_bound_and_grad(res.x)
                if val < best_val:
                    best_val = val
                    best_params = res.x.copy()
        except Exception:
            pass
    
    if not np.all(np.isfinite(best_params)):
        return seed_params

    return np.array(best_params)
