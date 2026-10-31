"""Analytic uncertainty bound with spherical reparameterization + L-BFGS-B."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, brentq

_DEADLINE = time.time() + 195
_START_TIME = time.time()

# Precompute I_n = integral x^{2n} e^{-x^2} dx = (2n)! / (4^n n!) * sqrt(pi)
def _precompute_I():
    """Precompute I_n for n=0..13"""
    I = np.zeros(14)
    sp = np.sqrt(np.pi)
    I[0] = sp
    for n in range(1, 14):
        I[n] = (2*n - 1) * I[n-1] / (4.0 * n)
    return I

_I = _precompute_I()

def _spherical_to_coefficients(theta, phi):
    """Convert spherical coordinates to (a0, a4, a8) on unit sphere."""
    a0 = np.cos(theta)
    a4 = np.sin(theta) * np.cos(phi)
    a8 = np.sin(theta) * np.sin(phi)
    return np.array([a0, a4, a8])

def _coefficients_to_spherical(a0, a4, a8):
    """Convert (a0, a4, a8) to (theta, phi) on unit sphere."""
    r = np.sqrt(a0**2 + a4**2 + a8**2)
    if r < 1e-15:
        return (np.pi/2, 0.0)
    a0n, a4n, a8n = a0/r, a4/r, a8/r
    theta = np.arccos(np.clip(a0n, -1.0, 1.0))
    phi = np.arctan2(a8n, a4n)
    return (theta, phi)


def _bound_and_grad(params):
    """
    Compute uncertainty product and its analytic gradient.
    g(x) = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * exp(-x^2/2)
    a12 = -(a0 + 3*a4 + 105*a8)/945  (enforces g(0)=0)
    """
    a0, a4, a8 = params[0], params[1], params[2]
    a12 = -(a0 + 3.0*a4 + 105.0*a8) / 945.0
    
    p = np.empty(7)
    p[0] = 0.0
    p[1] = -6.0*a4 - 420.0*a8 - 7560.0*a12
    p[2] = a4 + 210.0*a8 + 13860.0*a12
    p[3] = -28.0*a8 - 6930.0*a12
    p[4] = a8 + 1155.0*a12
    p[5] = -66.0*a12
    p[6] = a12
    
    I = _I
    pp = np.outer(p, p)
    m = np.zeros(3)
    for k in range(3):
        idx_mat = k + np.add.outer(np.arange(7), np.arange(7))
        m[k] = np.sum(pp * I[idx_mat])
    
    signs = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
    q = p * signs
    qq = np.outer(q, q)
    n = np.zeros(3)
    for k in range(3):
        n[k] = 2.0 * np.pi * np.sum(qq * I[k + np.add.outer(np.arange(7), np.arange(7))])
    
    m0, m2 = m[0], m[1]
    n0, n2 = n[0], n[1]
    
    if m0 <= 0 or n0 <= 0 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    vf = m2 / m0
    vF = n2 / n0
    
    if vf <= 0 or vF <= 0:
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    bound = np.sqrt(vf * vF)
    
    S_p = np.zeros((3, 7))
    S_q = np.zeros((3, 7))
    for k in range(3):
        for i in range(7):
            s_p = 0.0
            s_q = 0.0
            for j in range(7):
                idx = k + i + j
                if idx < 14:
                    s_p += p[j] * I[idx]
                    s_q += q[j] * I[idx]
            S_p[k, i] = s_p
            S_q[k, i] = s_q
    
    dpda = np.zeros((3, 7))
    dpda[0, 1] = 8.0
    dpda[1, 1] = -30.0
    dpda[2, 1] = -1260.0
    dpda[0, 2] = -13860.0/945.0
    dpda[1, 2] = 45.0
    dpda[2, 2] = 1750.0
    dpda[0, 3] = 6930.0/945.0
    dpda[1, 3] = -22.0
    dpda[2, 3] = -798.0
    dpda[0, 4] = -1155.0/945.0
    dpda[1, 4] = 3465.0/945.0
    dpda[2, 4] = 1 + 1155.0*105.0/945.0
    dpda[0, 5] = 66.0/945.0
    dpda[1, 5] = -198.0/945.0
    dpda[2, 5] = -6930.0/945.0
    dpda[0, 6] = -1.0/945.0
    dpda[1, 6] = -3.0/945.0
    dpda[2, 6] = -105.0/945.0
    
    grad = np.zeros(3)
    for c in range(3):
        dp = dpda[c]
        dm = 2.0 * (dp @ S_p)
        dq = dp * signs
        dn = 4.0 * np.pi * (dq @ S_q)
        dvm2 = (dm[1]*m0 - m2*dm[0]) / (m0*m0)
        dvn2 = (dn[1]*n0 - n2*dn[0]) / (n0*n0)
        grad[c] = (vF * dvm2 + vf * dvn2) / (2.0 * bound)
    
    return bound, grad


def _bound_from_spherical(theta, phi):
    """Compute bound from spherical coordinates (theta, phi) on unit sphere."""
    coeffs = _spherical_to_coefficients(theta, phi)
    val, _ = _bound_and_grad(coeffs)
    return val


def _bound_and_grad_spherical(params):
    """
    Compute bound and gradient in spherical coordinates (theta, phi).
    Uses chain rule: d/d(theta,phi) = (d/d a0, d/d a4, d/d a8) @ (d(a0,a4,a8)/d(theta,phi))
    """
    theta, phi = params[0], params[1]
    coeffs = _spherical_to_coefficients(theta, phi)
    bound, grad_coeff = _bound_and_grad(coeffs)
    
    st = np.sin(theta)
    ct = np.cos(theta)
    sp = np.sin(phi)
    cp = np.cos(phi)
    
    jac = np.array([
        [-st, 0.0],
        [ct*cp, -st*sp],
        [ct*sp, st*cp]
    ])
    
    grad_spherical = jac.T @ grad_coeff
    return bound, grad_spherical


def _objective_spherical(params):
    val, _ = _bound_and_grad_spherical(params)
    return val


def _objective_grad_spherical(params):
    val, grad = _bound_and_grad_spherical(params)
    return val, grad


def _objective(params):
    val, _ = _bound_and_grad(params)
    return val

def _objective_grad(params):
    val, grad = _bound_and_grad(params)
    return val, grad


def construct():
    # Convert seed to spherical coordinates
    seed_coeffs = np.array([-1.13, 0.04, 0.000308642])
    seed_theta, seed_phi = _coefficients_to_spherical(*seed_coeffs)
    
    # Generate diverse starting points in spherical coordinates
    rng = np.random.default_rng(42)
    starts = []
    
    # Seed point
    starts.append(np.array([seed_theta, seed_phi]))
    
    # Grid of starting points on the sphere - high resolution for global coverage
    for theta in np.linspace(0.05, np.pi - 0.05, 30):
        for phi in np.linspace(0.05, 2*np.pi - 0.05, 30):
            starts.append(np.array([theta, phi]))
    
    # Random points
    for _ in range(100):
        theta = rng.uniform(0.02, np.pi - 0.02)
        phi = rng.uniform(0.02, 2*np.pi - 0.02)
        starts.append(np.array([theta, phi]))
    
    best_val = np.inf
    best_params_sph = None
    best_params_linear = None
    
    # Phase 1: Multi-start L-BFGS-B in spherical coordinates
    for start in starts:
        if time.time() > _DEADLINE - 80:
            break
        try:
            res = minimize(_objective_grad_spherical, start, method='L-BFGS-B',
                          jac=True, bounds=[(0.001, np.pi - 0.001), (0.001, 2*np.pi - 0.001)],
                          options={'maxiter': 5000, 'ftol': 1e-14, 'gtol': 1e-13, 'maxls': 30})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params_sph = res.x.copy()
                best_params_linear = _spherical_to_coefficients(res.x[0], res.x[1])
        except Exception:
            pass
    
    # Phase 2: Refine in linear coordinates from best spherical result
    if best_params_linear is not None and time.time() < _DEADLINE - 50:
        try:
            res = minimize(_objective_grad, best_params_linear, method='L-BFGS-B',
                          jac=True, options={'maxiter': 10000, 'ftol': 1e-15, 'gtol': 1e-14, 'maxls': 50})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params_linear = res.x.copy()
        except Exception:
            pass
    
    # Phase 3: Nelder-Mead polish in linear coordinates
    if best_params_linear is not None and time.time() < _DEADLINE - 30:
        try:
            res = minimize(_objective, best_params_linear, method='Nelder-Mead',
                          options={'maxiter': 50000, 'xatol': 1e-16, 'fatol': 1e-18, 'adaptive': True})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params_linear = res.x.copy()
        except Exception:
            pass
    
    # Phase 4: Convert back to spherical and do high-precision polish there
    if best_params_linear is not None and time.time() < _DEADLINE - 15:
        theta, phi = _coefficients_to_spherical(*best_params_linear)
        try:
            res = minimize(_objective_grad_spherical, np.array([theta, phi]), method='L-BFGS-B',
                          jac=True, bounds=[(0.0001, np.pi - 0.0001), (0.0001, 2*np.pi - 0.0001)],
                          options={'maxiter': 20000, 'ftol': 1e-17, 'gtol': 1e-16, 'maxls': 50})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params_sph = res.x.copy()
                best_params_linear = _spherical_to_coefficients(res.x[0], res.x[1])
        except Exception:
            pass
    
    # Phase 5: Final ultra-precise Nelder-Mead in linear coords
    if best_params_linear is not None and time.time() < _DEADLINE - 5:
        try:
            res = minimize(_objective, best_params_linear, method='Nelder-Mead',
                          options={'maxiter': 100000, 'xatol': 1e-19, 'fatol': 1e-21, 'adaptive': True})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params_linear = res.x.copy()
        except Exception:
            pass
    
    # Phase 6: One more L-BFGS-B with tightest tolerances
    if best_params_linear is not None and time.time() < _DEADLINE - 2:
        try:
            res = minimize(_objective_grad, best_params_linear, method='L-BFGS-B',
                          jac=True, options={'maxiter': 30000, 'ftol': 1e-19, 'gtol': 1e-17, 'maxls': 50})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params_linear = res.x.copy()
        except Exception:
            pass
    
    # Phase 7: 1D ratio-based global search exploiting scale invariance
    # The uncertainty product is scale-invariant, so we can fix one coefficient
    # and search over the ratios of the others.
    if time.time() < _DEADLINE - 10:
        # Fix a0 = 1, search over (a4/a0, a8/a0)
        # But we need to handle the case where a0 might be zero, so use spherical
        # Instead, let's do a finer grid in the 2D parameter space
        
        # High-resolution grid search to find better starting points
        theta_grid = np.linspace(0.01, np.pi - 0.01, 50)
        phi_grid = np.linspace(0.01, 2*np.pi - 0.01, 50)
        
        for t in theta_grid:
            if time.time() > _DEADLINE - 10:
                break
            for p in phi_grid:
                try:
                    val = _bound_from_spherical(t, p)
                    if val < best_val and np.isfinite(val):
                        # Found a better point, refine locally
                        start = np.array([t, p])
                        res = minimize(_objective_grad_spherical, start, method='L-BFGS-B',
                                      jac=True, bounds=[(0.0001, np.pi - 0.0001), (0.0001, 2*np.pi - 0.0001)],
                                      options={'maxiter': 3000, 'ftol': 1e-15, 'gtol': 1e-14, 'maxls': 30})
                        if res.fun < best_val and np.isfinite(res.fun):
                            best_val = res.fun
                            best_params_sph = res.x.copy()
                            best_params_linear = _spherical_to_coefficients(res.x[0], res.x[1])
                except Exception:
                    pass
    
    # Phase 8: Final polish if we found something better
    if best_params_linear is not None and time.time() < _DEADLINE - 2:
        try:
            res = minimize(_objective_grad, best_params_linear, method='L-BFGS-B',
                          jac=True, options={'maxiter': 50000, 'ftol': 1e-20, 'gtol': 1e-18, 'maxls': 100})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params_linear = res.x.copy()
        except Exception:
            pass
    
    if best_params_linear is None:
        return np.array([-1.13, 0.04, 0.000308642])
    
    return np.array(best_params_linear)
# EVOLVE-BLOCK-END
