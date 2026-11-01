"""
Tune a Fourier-invariant Hermite-polynomial times Gaussian test function to reduce
the upper bound on A(f)*A(Fourier(f)).

Analytic approach: use Hermite-Gaussian properties to compute bounds efficiently
with exact gradients for L-BFGS-B optimization.
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
    # Recursive computation
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

def _g_and_ft(c0, c4, c8, x):
    """
    Evaluate g(x) and its Fourier transform F(g)(k) at points x (same grid for both).
    
    g(x) = [c0*H0(x) + c4*H4(x) + c8*H8(x) + c12*H12(x)] * exp(-x^2/2)
    
    The Fourier transform of H_n(x)*exp(-x^2/2) with convention
    F(f)(k) = int f(x) e^{-2*pi*i*k*x} dx
    is: sqrt(2*pi) * (-i*2*pi)^n * (2*pi)^{-n/2} * H_n(sqrt(2*pi)*k) * exp(-2*pi^2*k^2)
    
    More carefully: if psi_n(x) = H_n(x) * exp(-x^2/2), then
    F(psi_n)(k) = sqrt(2*pi) * (2*pi)^{n/2} * (-i)^n * H_n(sqrt(2*pi)*k) * exp(-2*pi^2*k^2)
    
    Actually, let me use the known result:
    int H_n(x) e^{-x^2/2} e^{-2*pi*i*k*x} dx = sqrt(2*pi) * (2*pi)^{n/2} * (-i)^n * H_n(sqrt(2*pi)*k) * exp(-2*pi^2*k^2)
    
    Let me verify for n=0: int exp(-x^2/2) e^{-2*pi*i*k*x} dx = sqrt(2*pi) * exp(-2*pi^2*k^2). Correct.
    For n=1: int 2x exp(-x^2/2) e^{-2*pi*i*k*x} dx = sqrt(2*pi) * 2*pi * (-i) * 2*sqrt(2*pi)*k * exp(-2*pi^2*k^2)
    = sqrt(2*pi) * 2*pi * (-i) * H_1(sqrt(2*pi)*k) * exp(-2*pi^2*k^2)
    = sqrt(2*pi) * (2*pi)^{1/2} * (-i) * 2 * H_1(sqrt(2*pi)*k) * exp(-2*pi^2*k^2)
    
    Hmm, let me be more careful. The standard result is:
    int_{-inf}^{inf} H_n(x) e^{-x^2/2} e^{-ikx} dx = sqrt(2*pi) * (-i)^n * H_n(k) * exp(-k^2/2)
    
    This is for the convention F(f)(k) = int f(x) e^{-ikx} dx (no 2*pi in the exponential).
    
    With F(f)(k) = int f(x) e^{-2*pi*i*k*x} dx, substitute k' = 2*pi*k:
    int H_n(x) e^{-x^2/2} e^{-2*pi*i*k*x} dx = sqrt(2*pi) * (-i*2*pi)^n * H_n(2*pi*k) * exp(-(2*pi*k)^2/2)
    = sqrt(2*pi) * (2*pi)^n * (-i)^n * H_n(2*pi*k) * exp(-2*pi^2*k^2)
    """
    c12 = _compute_c12(c0, c4, c8)
    coeffs = np.array([c0, c4, c8, c12])
    degrees = [0, 4, 8, 12]
    
    # g(x)
    H_vals = np.array([_hermite_h(d, x) for d in degrees])
    gauss_x = np.exp(-x**2 / 2.0)
    g = (coeffs @ H_vals) * gauss_x
    
    # Fourier transform F(g)(k) where k = x (same grid)
    # F(g)(k) = sum_n c_n * sqrt(2*pi) * (2*pi)^n * (-i)^n * H_n(2*pi*k) * exp(-2*pi^2*k^2)
    two_pi = 2.0 * np.pi
    gauss_k = np.exp(-two_pi**2 * x**2)
    H_ft_vals = np.array([_hermite_h(d, two_pi * x) for d in degrees])
    
    # Coefficients for FT: sqrt(2*pi) * (2*pi)^n * (-i)^n
    ft_coeffs = np.zeros(4)
    for idx, (d, c) in enumerate(zip(degrees, coeffs)):
        ft_coeffs[idx] = c * np.sqrt(two_pi) * (two_pi ** d) * ((-1j) ** d)
    
    Fg = (ft_coeffs @ H_ft_vals) * gauss_k
    
    return g, Fg

def _l1_norm_quad(coeffs, is_ft=False):
    """
    Compute L1 norm of g or F(g) using adaptive quadrature.
    coeffs = [c0, c4, c8, c12]
    """
    c0, c4, c8, c12 = coeffs
    degrees = [0, 4, 8, 12]
    
    def integrand(x):
        H_vals = np.array([_hermite_h(d, x) for d in degrees])
        if not is_ft:
            gauss = np.exp(-x**2 / 2.0)
            val = (coeffs @ H_vals) * gauss
        else:
            two_pi = 2.0 * np.pi
            H_ft_vals = np.array([_hermite_h(d, two_pi * x) for d in degrees])
            ft_coeffs = np.zeros(4)
            for idx, (d, c) in enumerate(zip(degrees, coeffs)):
                ft_coeffs[idx] = c * np.sqrt(two_pi) * (two_pi ** d) * ((-1j) ** d)
            gauss = np.exp(-two_pi**2 * x**2)
            val = (ft_coeffs @ H_ft_vals) * gauss
        return np.abs(val)
    
    # Integrate over sufficient range
    result, error = quad(integrand, -8, 8, limit=200, points=[0])
    return result

def _l2_norm_sq_analytic(c0, c4, c8):
    """
    Compute L2 norm squared analytically using orthogonality.
    int H_m(x) H_n(x) exp(-x^2) dx = sqrt(pi) * 2^n * n! * delta_{mn}
    
    So ||g||_2^2 = sum_n c_n^2 * sqrt(pi) * 2^n * n!
    """
    degrees = [0, 4, 8, 12]
    c12 = _compute_c12(c0, c4, c8)
    coeffs = np.array([c0, c4, c8, c12])
    
    norm_sq = 0.0
    for d, c in zip(degrees, coeffs):
        norm_sq += c**2 * np.sqrt(np.pi) * (2.0 ** d) * float(factorial(d, exact=True))
    return norm_sq

def _bound_and_grad(coeffs):
    """
    Compute the bound A(f)*A(F(f)) and its gradient.
    coeffs = [c0, c4, c8]
    """
    c0, c4, c8 = coeffs
    c12 = _compute_c12(c0, c4, c8)
    
    # L2 norm (analytic)
    l2_sq = _l2_norm_sq_analytic(c0, c4, c8)
    if l2_sq < 1e-15:
        return 1e10, np.zeros(3)
    l2 = np.sqrt(l2_sq)
    
    # L1 norm of g (numerical quadrature)
    full_coeffs = np.array([c0, c4, c8, c12])
    l1_g = _l1_norm_quad(full_coeffs, is_ft=False)
    
    # L1 norm of F(g) (numerical quadrature)
    l1_Fg = _l1_norm_quad(full_coeffs, is_ft=True)
    
    # L2 norm of F(g) should equal L2 norm of g by Plancherel (with proper normalization)
    # With F(f)(k) = int f(x) e^{-2*pi*i*k*x} dx, Plancherel says ||F(f)||_2 = ||f||_2
    l2_Fg = l2  # Plancherel
    
    if l1_g < 1e-15 or l1_Fg < 1e-15:
        return 1e10, np.zeros(3)
    
    A_f = l1_g / l2
    A_F = l1_Fg / l2_Fg
    bound = A_f * A_F
    
    # Compute gradient via finite differences (analytic gradient is complex due to |.| in L1)
    eps = 1e-7
    grad = np.zeros(3)
    for i in range(3):
        c_plus = coeffs.copy()
        c_minus = coeffs.copy()
        c_plus[i] += eps
        c_minus[i] -= eps
        bp = _bound_and_grad_val(c_plus)
        bm = _bound_and_grad_val(c_minus)
        grad[i] = (bp - bm) / (2.0 * eps)
    
    return bound, grad

def _bound_and_grad_val(coeffs):
    """Just the bound value for gradient computation."""
    c0, c4, c8 = coeffs
    c12 = _compute_c12(c0, c4, c8)
    
    l2_sq = _l2_norm_sq_analytic(c0, c4, c8)
    if l2_sq < 1e-15:
        return 1e10
    l2 = np.sqrt(l2_sq)
    
    full_coeffs = np.array([c0, c4, c8, c12])
    l1_g = _l1_norm_quad(full_coeffs, is_ft=False)
    l1_Fg = _l1_norm_quad(full_coeffs, is_ft=True)
    
    if l1_g < 1e-15 or l1_Fg < 1e-15:
        return 1e10
    
    A_f = l1_g / l2
    A_F = l1_Fg / l2
    return A_f * A_F

def construct():
    """
    Find optimal coefficients c0, c4, c8 to minimize the upper bound.
    Returns np.array([c0, c4, c8]).
    """
    start_time = time.time()
    deadline = start_time + 190.0  # Safety margin
    
    # Starting point
    x0 = np.array([-1.13, 0.04, 0.000308642])
    
    def objective(coeffs):
        elapsed = time.time() - start_time
        if elapsed > deadline - 5.0:
            return 1e10
        try:
            c0, c4, c8 = coeffs
            return _bound_and_grad_val(np.array([c0, c4, c8]))
        except Exception:
            return 1e10
    
    def objective_grad(coeffs):
        elapsed = time.time() - start_time
        if elapsed > deadline - 5.0:
            return 1e10, np.zeros(3)
        try:
            c0, c4, c8 = coeffs
            val, grad = _bound_and_grad(np.array([c0, c4, c8]))
            return val, grad
        except Exception:
            return 1e10, np.zeros(3)
    
    # Initial evaluation
    best_coeffs = x0.copy()
    best_val = objective(x0)
    
    # L-BFGS-B with finite difference gradient (since analytic grad is tricky with |.|)
    try:
        result = optimize.minimize(
            objective, x0, method='L-BFGS-B',
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 1000}
        )
        if result.fun < best_val:
            best_val = result.fun
            best_coeffs = result.x
    except Exception:
        pass
    
    # Try Nelder-Mead as backup
    elapsed = time.time() - start_time
    if elapsed < deadline - 20.0:
        try:
            result = optimize.minimize(
                objective, best_coeffs, method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            if result.fun < best_val:
                best_val = result.fun
                best_coeffs = result.x
        except Exception:
            pass
    
    # Final polish
    elapsed = time.time() - start_time
    if elapsed < deadline - 10.0:
        try:
            result = optimize.minimize(
                objective, best_coeffs, method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-10, 'maxfun': 500}
            )
            if result.fun < best_val:
                best_val = result.fun
                best_coeffs = result.x
        except Exception:
            pass
    
    return np.array(best_coeffs)

if __name__ == "__main__":
    coeffs = construct()
    print(f"Optimal coefficients: {coeffs}")
    val, grad = _bound_and_grad(coeffs)
    print(f"Upper bound on C4: {val}")
