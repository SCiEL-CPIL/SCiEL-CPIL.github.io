import numpy as np
from scipy.special import eval_hermite, hermgauss
from scipy.optimize import minimize
import time

def _compute_a12(a0, a4, a8):
    H12_0 = 498960.0
    return -(a0 + 12.0 * a4 + 1680.0 * a8) / H12_0

def _hermite_coeffs(a0, a4, a8, a12):
    """Return coefficients for H0, H4, H8, H12."""
    return np.array([a0, a4, a8, a12])

def _compute_norm4_squared(coeffs, degrees):
    """
    Compute ||sum c_n H_n(x) e^{-x^2/2}||_4^4 = integral |sum c_n H_n(x) e^{-x^2/2}|^4 dx
    using Gauss-Hermite quadrature.
    
    We use the substitution: let psi_n(x) = (2^n n! sqrt(pi))^{-1/2} H_n(x) e^{-x^2/2}
    be the orthonormal Hermite functions.
    
    Actually, let's work directly. g(x) = P(x) e^{-x^2/2} where P is polynomial degree <= 12.
    |g(x)|^4 = P(x)^4 e^{-2x^2}
    
    Using Gauss-Hermite: integral f(x) e^{-x^2} dx ≈ sum w_i f(x_i)
    We need integral P(x)^4 e^{-2x^2} dx. Let t = sqrt(2) x, dx = dt/sqrt(2).
    integral P(t/sqrt(2))^4 e^{-t^2} dt / sqrt(2)
    
    This is a polynomial in t of degree 48 times e^{-t^2}, so Gauss-Hermite with
    enough nodes (>= 25) gives exact result.
    """
    # Gauss-Hermite quadrature nodes and weights
    n_quad = 30  # 30 nodes is exact for polynomials up to degree 59
    x_nodes, w_nodes = hermgauss(n_quad)
    
    # Evaluate P(x) = sum c_d H_d(x) at scaled nodes
    # g(x) = P(x) e^{-x^2/2}, |g|^4 = P(x)^4 e^{-2x^2}
    # integral P(x)^4 e^{-2x^2} dx
    # With t = sqrt(2)*x: x = t/sqrt(2), dx = dt/sqrt(2)
    # integral P(t/sqrt(2))^4 e^{-t^2} dt / sqrt(2)
    
    x_scaled = x_nodes / np.sqrt(2.0)
    
    # Compute P(x) = sum c_d H_d(x)
    P_vals = np.zeros(len(x_nodes))
    for c, d in zip(coeffs, degrees):
        if abs(c) > 1e-15:
            P_vals += c * eval_hermite(d, x_scaled)
    
    # |g|^4 integral = integral P(x)^4 e^{-2x^2} dx = integral P(t/sqrt(2))^4 e^{-t^2} dt / sqrt(2)
    integrand = (P_vals ** 4) * w_nodes
    result = np.sum(integrand) / np.sqrt(2.0)
    
    return max(result, 0.0)

def _objective(coeffs):
    """Compute ||g||_4 * ||F(g)||_4 using analytical Hermite quadrature."""
    a0, a4, a8 = coeffs
    a12 = _compute_a12(a0, a4, a8)
    
    coeff_array = np.array([a0, a4, a8, a12])
    degrees = [0, 4, 8, 12]
    
    # ||g||_4^4
    norm_g4_sq = _compute_norm4_squared(coeff_array, degrees)
    if norm_g4_sq <= 1e-300:
        return np.inf
    norm_g4 = norm_g4_sq ** 0.25
    
    # For Fourier transform: F(g)(xi) = integral g(x) e^{-ixi} dx
    # For g(x) = sum c_n H_n(x) e^{-x^2/2}:
    # The Fourier transform of H_n(x) e^{-x^2/2} is (-i)^n H_n(xi) e^{-xi^2/2} * sqrt(2*pi)
    # Wait, let me be more careful with conventions.
    # 
    # Standard: F(f)(xi) = integral f(x) e^{-2*pi*i*x*xi} dx
    # But in physics/math, often: F(f)(xi) = (1/sqrt(2*pi)) integral f(x) e^{-i*x*xi} dx
    #
    # The task says "A(f)*A(Fourier(f))" which suggests the uncertainty principle constant.
    # Let's use the convention where the Hermite functions are eigenfunctions of FT.
    #
    # With F(f)(xi) = (1/sqrt(2*pi)) integral f(x) e^{-i*x*xi} dx:
    # F[H_n(x) e^{-x^2/2}](xi) = (-i)^n H_n(xi) e^{-xi^2/2}
    #
    # So F(g)(xi) = sum c_n (-i)^n H_n(xi) e^{-xi^2/2}
    
    # Fourier coefficients get phase factors (-i)^n
    fourier_coeffs = np.array([
        coeff_array[0] * (-1j)**0,  # H0: phase 1
        coeff_array[1] * (-1j)**4,  # H4: phase 1
        coeff_array[2] * (-1j)**8,  # H8: phase 1
        coeff_array[3] * (-1j)**12  # H12: phase 1
    ])
    
    # All even degrees, so (-i)^n = 1 for all n in {0,4,8,12}
    # Thus F(g) has the same functional form as g!
    
    norm_Fg4_sq = _compute_norm4_squared(fourier_coeffs, degrees)
    if norm_Fg4_sq <= 1e-300:
        return np.inf
    norm_Fg4 = norm_Fg4_sq ** 0.25
    
    return norm_g4 * norm_Fg4

def construct():
    start_time = time.time()
    deadline = 195.0  # 15s safety margin before 210s kill
    
    # Seed coefficients
    seed = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    
    # Evaluate seed
    best_val = _objective(seed)
    best_coeffs = seed.copy()
    
    # Phase 1: Nelder-Mead from seed
    try:
        if time.time() - start_time < deadline - 30:
            res = minimize(_objective, seed, method='Nelder-Mead',
                          options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-14,
                                   'adaptive': True})
            if np.isfinite(res.fun) and res.fun < best_val:
                best_val = res.fun
                best_coeffs = res.x.copy()
    except Exception:
        pass
    
    # Phase 2: BFGS refinement
    try:
        if time.time() - start_time < deadline - 20:
            res = minimize(_objective, best_coeffs, method='BFGS',
                          options={'maxiter': 5000, 'gtol': 1e-13})
            if np.isfinite(res.fun) and res.fun < best_val:
                best_val = res.fun
                best_coeffs = res.x.copy()
    except Exception:
        pass
    
    # Phase 3: L-BFGS-B for additional refinement
    try:
        if time.time() - start_time < deadline - 15:
            res = minimize(_objective, best_coeffs, method='L-BFGS-B',
                          options={'maxiter': 5000, 'gtol': 1e-13, 'ftol': 1e-14})
            if np.isfinite(res.fun) and res.fun < best_val:
                best_val = res.fun
                best_coeffs = res.x.copy()
    except Exception:
        pass
    
    # Phase 4: Random restarts with multiple scales
    rng = np.random.RandomState(42)
    scales = [0.5, 0.1, 0.01, 0.001]
    for scale in scales:
        for i in range(8):
            if time.time() - start_time > deadline - 5:
                break
            perturbation = rng.normal(0, scale, 3)
            start_point = best_coeffs + perturbation
            try:
                res = minimize(_objective, start_point, method='Nelder-Mead',
                              options={'maxiter': 3000, 'xatol': 1e-11, 'fatol': 1e-13,
                                       'adaptive': True})
                if np.isfinite(res.fun) and res.fun < best_val:
                    best_val = res.fun
                    best_coeffs = res.x.copy()
            except Exception:
                pass
    
    # Phase 5: Final BFGS polish
    try:
        if time.time() - start_time < deadline - 2:
            res = minimize(_objective, best_coeffs, method='BFGS',
                          options={'maxiter': 2000, 'gtol': 1e-14})
            if np.isfinite(res.fun) and res.fun < best_val:
                best_val = res.fun
                best_coeffs = res.x.copy()
    except Exception:
        pass
    
    return best_coeffs
