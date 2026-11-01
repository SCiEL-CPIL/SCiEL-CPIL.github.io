"""
Tune a Fourier-invariant Hermite-polynomial times Gaussian test function to reduce
the upper bound on A(f)*A(Fourier(f)).

Reparameterized on the unit sphere with smooth L1 approximation for stable gradients.
"""

import numpy as np
from scipy import optimize
from scipy.special import factorial
from scipy.integrate import quad
import time

# Precompute constants
_H0_0 = 1.0
_H4_0 = 12.0
_H8_0 = 1190.0
_H12_0 = 113880.0

def _hermite_h(n, x):
    """Compute physicist's Hermite polynomial H_n(x)."""
    if n == 0:
        return np.ones_like(x) if np.ndim(x) else 1.0
    if n == 1:
        return 2 * x if np.ndim(x) else 2.0 * x
    if np.ndim(x) == 0:
        x = np.array([x])
    x = np.asarray(x, dtype=float)
    h0 = np.ones_like(x)
    h1 = 2.0 * x
    for k in range(2, n + 1):
        h2 = 2.0 * x * h1 - 2.0 * (k - 1) * h0
        h0, h1 = h1, h2
    return h1

def _compute_c12(c0, c4, c8):
    """Compute c12 from the constraint g(0) = 0."""
    return -(c0 * _H0_0 + c4 * _H4_0 + c8 * _H8_0) / _H12_0

def _smooth_l1_quad(c0, c4, c8, is_ft=False, sigma=0.01):
    """
    Compute smooth L1 norm of g or F(g) using adaptive quadrature.
    Uses sqrt(x^2 + sigma^2) instead of |x| for differentiability.
    """
    c12 = _compute_c12(c0, c4, c8)
    coeffs = np.array([c0, c4, c8, c12])
    degrees = [0, 4, 8, 12]
    two_pi = 2.0 * np.pi
    
    if not is_ft:
        def integrand(x):
            H_vals = np.array([_hermite_h(d, x) for d in degrees])
            gauss = np.exp(-x**2 / 2.0)
            val = (coeffs @ H_vals) * gauss
            return np.sqrt(val**2 + sigma**2)
    else:
        ft_coeffs = np.zeros(4)
        for idx, (d, c) in enumerate(zip(degrees, coeffs)):
            ft_coeffs[idx] = c * np.sqrt(two_pi) * (two_pi ** d) * ((-1j) ** d)
        
        def integrand(x):
            H_ft_vals = np.array([_hermite_h(d, two_pi * x) for d in degrees])
            gauss = np.exp(-two_pi**2 * x**2)
            val = (ft_coeffs @ H_ft_vals) * gauss
            return np.sqrt(val**2 + sigma**2)
    
    result, error = quad(integrand, -10, 10, limit=300, points=[0])
    return result

def _l1_norm_quad(c0, c4, c8, is_ft=False):
    """
    Compute L1 norm of g or F(g) using adaptive quadrature (exact, non-smooth).
    """
    c12 = _compute_c12(c0, c4, c8)
    coeffs = np.array([c0, c4, c8, c12])
    degrees = [0, 4, 8, 12]
    two_pi = 2.0 * np.pi
    
    if not is_ft:
        def integrand(x):
            H_vals = np.array([_hermite_h(d, x) for d in degrees])
            gauss = np.exp(-x**2 / 2.0)
            val = (coeffs @ H_vals) * gauss
            return np.abs(val)
    else:
        ft_coeffs = np.zeros(4)
        for idx, (d, c) in enumerate(zip(degrees, coeffs)):
            ft_coeffs[idx] = c * np.sqrt(two_pi) * (two_pi ** d) * ((-1j) ** d)
        
        def integrand(x):
            H_ft_vals = np.array([_hermite_h(d, two_pi * x) for d in degrees])
            gauss = np.exp(-two_pi**2 * x**2)
            val = (ft_coeffs @ H_ft_vals) * gauss
            return np.abs(val)
    
    result, error = quad(integrand, -10, 10, limit=300, points=[0])
    return result

def _l2_norm_sq_analytic(c0, c4, c8):
    """
    Compute L2 norm squared analytically using orthogonality.
    int H_m(x) H_n(x) exp(-x^2) dx = sqrt(pi) * 2^n * n! * delta_{mn}
    """
    c12 = _compute_c12(c0, c4, c8)
    coeffs = np.array([c0, c4, c8, c12])
    degrees = [0, 4, 8, 12]
    
    norm_sq = 0.0
    for d, c in zip(degrees, coeffs):
        norm_sq += c**2 * np.sqrt(np.pi) * (2.0 ** d) * float(factorial(d, exact=True))
    return norm_sq

def _bound_value(c0, c4, c8, smooth=False, sigma=0.01):
    """Compute the bound A(f)*A(F(f)) for given coefficients."""
    c12 = _compute_c12(c0, c4, c8)
    
    l2_sq = _l2_norm_sq_analytic(c0, c4, c8)
    if l2_sq < 1e-15:
        return 1e10
    l2 = np.sqrt(l2_sq)
    
    if smooth:
        l1_g = _smooth_l1_quad(c0, c4, c8, is_ft=False, sigma=sigma)
        l1_Fg = _smooth_l1_quad(c0, c4, c8, is_ft=True, sigma=sigma)
    else:
        l1_g = _l1_norm_quad(c0, c4, c8, is_ft=False)
        l1_Fg = _l1_norm_quad(c0, c4, c8, is_ft=True)
    
    if l1_g < 1e-15 or l1_Fg < 1e-15:
        return 1e10
    
    A_f = l1_g / l2
    A_F = l1_Fg / l2
    return A_f * A_F

def _bound_and_grad(c0, c4, c8, smooth=True, sigma=0.01):
    """
    Compute the bound and its gradient using finite differences.
    """
    coeffs = np.array([c0, c4, c8])
    val = _bound_value(c0, c4, c8, smooth=smooth, sigma=sigma)
    
    if val >= 1e10:
        return val, np.zeros(3)
    
    eps = 1e-6
    grad = np.zeros(3)
    for i in range(3):
        c_plus = coeffs.copy()
        c_minus = coeffs.copy()
        c_plus[i] += eps
        c_minus[i] -= eps
        bp = _bound_value(c_plus[0], c_plus[1], c_plus[2], smooth=smooth, sigma=sigma)
        bm = _bound_value(c_minus[0], c_minus[1], c_minus[2], smooth=smooth, sigma=sigma)
        grad[i] = (bp - bm) / (2.0 * eps)
    
    return val, grad

def _sphere_to_cartesian(theta, phi):
    """Convert spherical coordinates (theta in [0,pi], phi in [0,2pi]) to unit vector."""
    return np.array([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta)
    ])

def _cartesian_to_sphere(v):
    """Convert unit vector to spherical coordinates."""
    v = np.asarray(v)
    norm = np.linalg.norm(v)
    if norm < 1e-15:
        return np.array([np.pi/2, 0.0])
    v = v / norm
    theta = np.arccos(np.clip(v[2], -1, 1))
    phi = np.arctan2(v[1], v[0])
    return np.array([theta, phi])

def construct():
    """
    Find optimal coefficients c0, c4, c8 to minimize the upper bound.
    Returns np.array([c0, c4, c8]).
    """
    start_time = time.time()
    deadline = start_time + 195.0  # Safety margin before 210s limit
    
    # Starting point from seed
    x0 = np.array([-1.13, 0.04, 0.000308642])
    
    def objective(coeffs):
        elapsed = time.time() - start_time
        if elapsed > deadline - 5.0:
            return 1e10
        try:
            c0, c4, c8 = coeffs
            return _bound_value(c0, c4, c8, smooth=True, sigma=0.01)
        except Exception:
            return 1e10
    
    def objective_grad(coeffs):
        elapsed = time.time() - start_time
        if elapsed > deadline - 5.0:
            return 1e10, np.zeros(3)
        try:
            c0, c4, c8 = coeffs
            return _bound_and_grad(c0, c4, c8, smooth=True, sigma=0.01)
        except Exception:
            return 1e10, np.zeros(3)
    
    # Phase 1: L-BFGS-B with smooth objective
    best_coeffs = x0.copy()
    best_val = objective(x0)
    
    elapsed = time.time() - start_time
    if elapsed < deadline - 60.0:
        try:
            result = optimize.minimize(
                objective, x0, method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 1500}
            )
            if result.fun < best_val:
                best_val = result.fun
                best_coeffs = result.x.copy()
        except Exception:
            pass
    
    # Phase 2: Nelder-Mead on the sphere (reparameterized)
    elapsed = time.time() - start_time
    if elapsed < deadline - 50.0:
        # Normalize to sphere
        v = best_coeffs / np.linalg.norm(best_coeffs)
        theta_phi = _cartesian_to_sphere(v)
        
        def sphere_objective(tp):
            elapsed = time.time() - start_time
            if elapsed > deadline - 5.0:
                return 1e10
            try:
                v = _sphere_to_cartesian(tp[0], tp[1])
                scale = np.linalg.norm(best_coeffs)
                c = v * scale
                return _bound_value(c[0], c[1], c[2], smooth=True, sigma=0.01)
            except Exception:
                return 1e10
        
        try:
            result = optimize.minimize(
                sphere_objective, theta_phi, method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            v = _sphere_to_cartesian(result.x[0], result.x[1])
            scale = np.linalg.norm(best_coeffs)
            c = v * scale
            val = _bound_value(c[0], c[1], c[2], smooth=True, sigma=0.01)
            if val < best_val:
                best_val = val
                best_coeffs = c
        except Exception:
            pass
    
    # Phase 3: Coordinate descent polish on the sphere with adaptive steps
    elapsed = time.time() - start_time
    if elapsed < deadline - 30.0:
        v = best_coeffs / np.linalg.norm(best_coeffs)
        scale = np.linalg.norm(best_coeffs)
        
        # Try small perturbations in each coordinate direction
        for iteration in range(50):
            if time.time() - start_time > deadline - 15.0:
                break
            improved = False
            for i in range(3):
                for step in [0.01, 0.001, 0.0001]:
                    for sign in [1, -1]:
                        v_test = v.copy()
                        v_test[i] += sign * step
                        v_test = v_test / np.linalg.norm(v_test)
                        c_test = v_test * scale
                        val_test = _bound_value(c_test[0], c_test[1], c_test[2], smooth=True, sigma=0.01)
                        if val_test < best_val - 1e-12:
                            v = v_test
                            best_val = val_test
                            best_coeffs = c_test
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                break
    
    # Phase 4: Final polish with exact (non-smooth) L1 and tight tolerances
    elapsed = time.time() - start_time
    if elapsed < deadline - 10.0:
        def objective_exact(coeffs):
            elapsed = time.time() - start_time
            if elapsed > deadline - 3.0:
                return 1e10
            try:
                c0, c4, c8 = coeffs
                return _bound_value(c0, c4, c8, smooth=False)
            except Exception:
                return 1e10
        
        try:
            result = optimize.minimize(
                objective_exact, best_coeffs, method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10, 'maxfun': 800}
            )
            if result.fun < best_val:
                best_val = result.fun
                best_coeffs = result.x.copy()
        except Exception:
            pass
    
    # Verify with exact L1
    final_val = _bound_value(best_coeffs[0], best_coeffs[1], best_coeffs[2], smooth=False)
    
    return np.array(best_coeffs)

if __name__ == "__main__":
    coeffs = construct()
    print(f"Optimal coefficients: {coeffs}")
    val, grad = _bound_and_grad(coeffs[0], coeffs[1], coeffs[2], smooth=False)
    print(f"Upper bound on C4: {val}")
