"""Analytic Hermite-Gaussian uncertainty bound with global search and high-precision polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = time.time() + 195  # 15s safety margin before the 210s kill

def _compute_moments(a0, a4, a8):
    """
    Compute m0, m2, n0, n2 exactly using Hermite polynomial properties.
    f(x) = sum a_k H_k(x) e^{-x^2/2}
    m_k = int x^k |f|^2 dx
    n_k = int xi^k |hat{f}|^2 dxi
    """
    # Coefficients for H0, H4, H8, H12
    a12 = -(a0 + 3.0 * a4 + 105.0 * a8) / 945.0
    coeffs = [a0, a4, a8, a12]
    degrees = [0, 4, 8, 12]
    
    # m0 = sum a_k^2 * 2^k * k! * sqrt(pi)
    m0 = 0.0
    m2 = 0.0
    n0 = 0.0
    n2 = 0.0
    
    sqrt_pi = np.sqrt(np.pi)
    
    # Precompute factorials and powers of 2
    # H_k orthogonality: int H_n H_m e^{-x^2} dx = sqrt(pi) 2^n n! delta_nm
    # For m2: int x^2 H_n H_m e^{-x^2} dx
    # x^2 = 1/4 H_2 + 1/2 H_0
    # int x^2 H_n H_m e^{-x^2} = 1/4 int H_2 H_n H_m e^{-x^2} + 1/2 int H_0 H_n H_m e^{-x^2}
    
    # We need 3-term integrals of Hermite polynomials.
    # This is complex. Let's use the recurrence relation for moments directly.
    # Alternative: Numerical integration with high precision Gauss-Hermite quadrature.
    # Gauss-Hermite with 100 points is exact for polynomials up to degree 199.
    # Our polynomial degree is 24 (12*2). So 20 points would suffice, 50 is safe.
    
    x, w = np.polynomial.hermite.hermgauss(100)
    # The weight function for hermgauss is e^{-x^2}.
    # So int P(x) e^{-x^2} dx = sum w_i P(x_i)
    
    g = np.zeros_like(x)
    for a, k in zip(coeffs, degrees):
        g += a * np.polynomial.hermite.hermval(x, np.zeros(100)) # This is wrong, hermval needs coeffs
        
    # Correct evaluation of g(x)
    g = np.zeros_like(x)
    for a, k in zip(coeffs, degrees):
        # H_k(x)
        # Use np.polynomial.hermite.hermval with coefficients [0,0,...,a]
        c = np.zeros(100)
        c[k] = a
        g += np.polynomial.hermite.hermval(x, c)
        
    # m0 = int g^2 e^{-x^2} dx
    m0 = np.sum(w * g**2)
    
    # m2 = int x^2 g^2 e^{-x^2} dx
    m2 = np.sum(w * x**2 * g**2)
    
    # For n0, n2:
    # hat{f}(xi) = 1/sqrt(2pi) int f(x) e^{-ixxi} dx
    # Since f is a sum of H_k(x) e^{-x^2/2}, hat{f} is a sum of (-i)^k H_k(xi) e^{-xi^2/2} / sqrt(2pi) * sqrt(pi) * 2^k k! ?
    # Fourier transform of H_n(x) e^{-x^2/2} is (-i)^n H_n(xi) e^{-xi^2/2}.
    # So hat{f}(xi) = 1/sqrt(2pi) sum a_k (-i)^k H_k(xi) e^{-xi^2/2}
    # |hat{f}|^2 = 1/(2pi) (sum a_k (-i)^k H_k(xi) e^{-xi^2/2}) (sum a_j (i)^j H_j(xi) e^{-xi^2/2})
    # n0 = int |hat{f}|^2 dxi = 1/(2pi) sum a_k a_j (-i)^k (i)^j int H_k H_j e^{-xi^2} dxi
    # = 1/(2pi) sum a_k^2 (-1)^k sqrt(pi) 2^k k!
    # Wait, the cross terms vanish.
    # n0 = sqrt(pi)/(2pi) sum a_k^2 (-1)^k 2^k k!
    
    n0 = 0.0
    for a, k in zip(coeffs, degrees):
        n0 += a**2 * ((-1)**k) * (2.0**k) * (np.math.factorial(k))
    n0 *= sqrt_pi / (2.0 * np.pi)
    
    # n2 = int xi^2 |hat{f}|^2 dxi
    # Similar to m2, but with the coefficients modified by (-i)^k.
    # Let b_k = a_k (-i)^k.
    # n2 = 1/(2pi) int xi^2 (sum b_k H_k(xi) e^{-xi^2/2})^2 dxi
    # = 1/(2pi) int xi^2 |g_hat(xi)|^2 e^{-xi^2} dxi
    # This is the same form as m2, but with coefficients b_k.
    
    g_hat_coeffs = [a * ((-1j)**k) for a, k in zip(coeffs, degrees)]
    # We can reuse the logic for m2 by computing the integral with complex coefficients?
    # No, |hat{f}|^2 is real.
    # g_hat(xi) = sum b_k H_k(xi) e^{-xi^2/2}
    # |g_hat|^2 = g_hat * conj(g_hat)
    # Since H_k are real, conj(b_k) = a_k (i)^k.
    # So we are integrating xi^2 (sum b_k H_k) (sum conj(b_j) H_j) e^{-xi^2}
    
    # Let's just compute it numerically on the same grid.
    g_hat = np.zeros_like(x)
    for a, k in zip(coeffs, degrees):
        b = a * ((-1j)**k)
        c = np.zeros(100)
        c[k] = b
        g_hat += np.polynomial.hermite.hermval(x, c)
        
    # Note: The grid x and weights w are for e^{-x^2}.
    # n2 = 1/(2pi) * sum w_i * x_i^2 * |g_hat(x_i)|^2
    n2 = np.sum(w * x**2 * np.abs(g_hat)**2)
    n2 /= (2.0 * np.pi)
    
    return m0, m2, n0, n2

def _bound_and_grad(a0, a4, a8):
    m0, m2, n0, n2 = _compute_moments(a0, a4, a8)
    if m0 <= 0 or n0 <= 0 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18, np.array([0.0, 0.0, 0.0])
    
    vf = m2 / m0
    vF = n2 / n0
    
    if vf <= 0 or vF <= 0:
        return 1e18, np.array([0.0, 0.0, 0.0])
        
    bound = np.sqrt(vf * vF)
    
    # Gradient via finite differences
    eps = 1e-7
    grad = np.zeros(3)
    base = bound
    for i in range(3):
        p = np.array([a0, a4, a8])
        p[i] += eps
        m0_p, m2_p, n0_p, n2_p = _compute_moments(p[0], p[1], p[2])
        if m0_p > 0 and n0_p > 0:
            vf_p = m2_p / m0_p
            vF_p = n2_p / n0_p
            if vf_p > 0 and vF_p > 0:
                bound_p = np.sqrt(vf_p * vF_p)
                grad[i] = (bound_p - base) / eps
            else:
                grad[i] = 0
        else:
            grad[i] = 0
            
    return bound, grad

def _objective_unit(params):
    norm = np.linalg.norm(params)
    if norm < 1e-15:
        return 1e18, np.array([0.0, 0.0, 0.0])
    unit = params / norm
    val, grad = _bound_and_grad(unit[0], unit[1], unit[2])
    if not np.isfinite(val):
        return 1e18, np.array([0.0, 0.0, 0.0])
    # Project gradient onto tangent space
    grad_tangent = grad - np.dot(grad, unit) * unit
    return val, grad_tangent

def construct():
    # Grid search on the sphere
    # Parameterize unit sphere: 
    # a0 = sin(theta) cos(phi)
    # a4 = sin(theta) sin(phi)
    # a8 = cos(theta)
    
    best_val = 1e18
    best_params = np.array([1.0, 0.0, 0.0])
    
    n_theta = 200
    n_phi = 200
    
    thetas = np.linspace(0, np.pi, n_theta)
    phis = np.linspace(0, 2*np.pi, n_phi)
    
    # Coarse grid
    for th in thetas:
        if time.time() > _DEADLINE - 60: break
        st, ct = np.sin(th), np.cos(th)
        for ph in phis:
            sp, cp = np.sin(ph), np.cos(ph)
            a0 = st * cp
            a4 = st * sp
            a8 = ct
            val, _ = _bound_and_grad(a0, a4, a8)
            if val < best_val:
                best_val = val
                best_params = np.array([a0, a4, a8])
                
    # Local polish
    if time.time() < _DEADLINE - 30:
        try:
            res = minimize(_objective_unit, best_params, method='L-BFGS-B',
                           options={'maxiter': 5000, 'gtol': 1e-12, 'ftol': 1e-12})
            if res.fun < best_val and np.isfinite(res.fun):
                norm_res = np.linalg.norm(res.x)
                if norm_res > 1e-15:
                    best_params = res.x / norm_res
                    best_val = res.fun
        except Exception:
            pass
            
    # Second round of polish with different start points if time permits
    if time.time() < _DEADLINE - 10:
        # Try a few random points
        rng = np.random.default_rng(42)
        for _ in range(10):
            v = rng.normal(0, 1, 3)
            v /= np.linalg.norm(v)
            try:
                res = minimize(_objective_unit, v, method='L-BFGS-B',
                               options={'maxiter': 2000, 'gtol': 1e-12, 'ftol': 1e-12})
                if res.fun < best_val and np.isfinite(res.fun):
                    norm_res = np.linalg.norm(res.x)
                    if norm_res > 1e-15:
                        best_params = res.x / norm_res
                        best_val = res.fun
            except Exception:
                pass

    return np.array(best_params)
# EVOLVE-BLOCK-END
