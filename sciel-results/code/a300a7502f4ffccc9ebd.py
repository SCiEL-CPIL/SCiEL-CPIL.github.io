"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.special import eval_hermite, factorial
from math import sqrt, pi


def _hermite_values_at_zero(max_n=24):
    """Compute H_n(0) for even n up to max_n."""
    h0 = np.zeros(max_n + 1)
    for n in range(0, max_n + 1, 2):
        # H_n(0) for even n: (-1)^(n/2) * (n-1)!! = (-1)^(n/2) * n! / (2^(n/2) * (n/2)!)
        h0[n] = (-1) ** (n // 2) * factorial(n) / (2 ** (n // 2) * factorial(n // 2))
    return h0


def _precompute_basis(x):
    """Precompute the Hermite-Gaussian basis functions for degrees 0, 4, 8, 12."""
    basis = []
    for deg in [0, 4, 8, 12]:
        H = eval_hermite(deg, x)
        basis.append(np.exp(-x ** 2 / 2) * H)
    return basis


def _compute_objective_and_gradient(c, x, dx, basis, h0_vec):
    """
    Compute the objective function (upper bound on C4) and its gradient.
    
    c = [c0, c4, c8] are the free coefficients.
    c12 is determined by the constraint g(0) = 0.
    
    The objective is: (||g||_1 / ||g||_2) * (||F(g)||_1 / ||F(g)||_2)
    
    For g(x) = e^{-x^2/2} P(x), the Fourier transform is:
    F(g)(xi) = sqrt(2*pi) * sum_k c_k * i^{2k} * e^{-xi^2/2} H_{2k}(xi)
             = sqrt(2*pi) * sum_k c_k * (-1)^k * e^{-xi^2/2} H_{2k}(xi)
    
    So F(g) is also a Hermite-Gaussian combination with alternating signs.
    """
    # Get c12 from constraint: c0*H0(0) + c4*H4(0) + c8*H8(0) + c12*H12(0) = 0
    c12 = -(c[0] * h0_vec[0] + c[1] * h0_vec[1] + c[2] * h0_vec[2]) / h0_vec[3]
    
    # Build full coefficient vector
    coeffs = np.array([c[0], c[1], c[2], c12])
    
    # Compute g(x) = sum_k coeffs[k] * basis[k](x)
    g = np.zeros_like(x)
    for k in range(4):
        g += coeffs[k] * basis[k]
    
    # Compute ||g||_2^2 and ||g||_1
    g_sq = g ** 2
    norm2_sq = np.sum(g_sq) * dx
    if norm2_sq < 1e-15:
        return 1e10, np.zeros(3)
    norm1 = np.sum(np.abs(g)) * dx
    
    # For the Fourier transform, F(g)(xi) = sqrt(2*pi) * sum_k coeffs[k] * (-1)^k * basis[k](xi)
    # (since i^{2k} = (-1)^k for even degree 2k)
    fourier_coeffs = coeffs * np.array([1, -1, 1, -1])  # (-1)^0, (-1)^1, (-1)^2, (-1)^3
    
    # Compute F(g) on the same grid (by symmetry, we can use the same x values)
    fg = np.zeros_like(x)
    for k in range(4):
        fg += fourier_coeffs[k] * basis[k]
    
    fg_sq = fg ** 2
    fnorm2_sq = np.sum(fg_sq) * dx
    if fnorm2_sq < 1e-15:
        return 1e10, np.zeros(3)
    fnorm1 = np.sum(np.abs(fg)) * dx
    
    # The objective is (||g||_1 / ||g||_2) * (||F(g)||_1 / ||F(g)||_2)
    # = (norm1 / sqrt(norm2_sq)) * (fnorm1 / sqrt(fnorm2_sq))
    obj = (norm1 / sqrt(norm2_sq)) * (fnorm1 / sqrt(fnorm2_sq))
    
    # Compute gradient numerically (finite differences) for robustness
    # This is acceptable given the 180s budget and 3D problem
    grad = np.zeros(3)
    eps = 1e-7
    for i in range(3):
        c_plus = c.copy()
        c_plus[i] += eps
        obj_plus, _ = _compute_objective_and_gradient(c_plus, x, dx, basis, h0_vec)
        
        c_minus = c.copy()
        c_minus[i] -= eps
        obj_minus, _ = _compute_objective_and_gradient(c_minus, x, dx, basis, h0_vec)
        
        grad[i] = (obj_plus - obj_minus) / (2 * eps)
    
    return obj, grad


def construct():
    t0 = time.time()
    deadline = t0 + 170  # Safety margin
    
    # Grid for numerical integration - use enough points for accuracy
    x = np.linspace(-12, 12, 10000)
    dx = x[1] - x[0]
    
    # Precompute basis functions
    basis = _precompute_basis(x)
    
    # H_n(0) values
    h0 = _hermite_values_at_zero(12)
    h0_vec = np.array([h0[0], h0[4], h0[8], h0[12]])
    
    # Initial guess: try several starting points and pick the best
    initial_guesses = [
        np.array([1.0, -0.5, 0.2]),
        np.array([0.5, -0.3, 0.1]),
        np.array([1.0, -0.8, 0.3]),
        np.array([0.8, -0.6, 0.25]),
        np.array([1.0, -0.4, 0.15]),
    ]
    
    best_result = None
    best_obj = np.inf
    
    for guess in initial_guesses:
        if time.time() > deadline - 30:
            break
        
        try:
            result = minimize(
                _compute_objective_and_gradient,
                guess,
                args=(x, dx, basis, h0_vec),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if result.success and result.fun < best_obj:
                best_obj = result.fun
                best_result = result.x
                
        except Exception as e:
            continue
    
    # If all optimizations failed, return a reasonable default
    if best_result is None:
        return np.array([1.0, -0.5, 0.2])
    
    return best_result

# EVOLVE-BLOCK-END
