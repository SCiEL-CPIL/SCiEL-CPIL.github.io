"""Optimized program for uncertainty inequality task."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

# Precompute constants for Hermite polynomials of degrees 0, 4, 8, 12
_DEGREES = np.array([0, 4, 8, 12], dtype=np.int64)
_N_VALUES = [0, 4, 8, 12]

# H_n(0) values for even n
_H0_0 = 1.0
_H4_0 = -12.0
_H8_0 = 1680.0
_H12_0 = -103950.0

# Orthogonality weights: w_n = sqrt(pi) * 2^n * n!
def _weight(n):
    return np.sqrt(np.pi) * (2.0 ** n) * float(np.math.factorial(n))

_W = np.array([_weight(n) for n in _N_VALUES])

# M_kl = ∫ x² H_k(x) H_l(x) exp(-x²) dx
# Using: x² H_k = (1/4)H_{k+2} + (2k+1)/2 H_k + k(k-1)/2 H_{k-2}
# M_kl = w_l * [ (1/4)δ_{l,k+2} + (2k+1)/2 δ_{k,l} + k(k-1)/2 δ_{l,k-2} ]
_M = np.zeros((4, 4))
for i, k in enumerate(_N_VALUES):
    for j, l in enumerate(_N_VALUES):
        term = 0.0
        if l == k + 2:
            term += 0.25
        if k == l:
            term += (2 * k + 1) / 2.0
        if l == k - 2:
            term += k * (k - 1) / 2.0
        _M[i, j] = _W[j] * term

# a12 = -(a0*H0(0) + a4*H4(0) + a8*H8(0)) / H12(0)
# = -(a0*1 + a4*(-12) + a8*1680) / (-103950)
# = (a0 - 12*a4 + 1680*a8) / 103950
_A12_COEFFS = np.array([1.0, -12.0, 1680.0]) / 103950.0


def _get_all_coeffs(a0, a4, a8):
    """Get all four coefficients [a0, a4, a8, a12] from the three free ones."""
    a12 = a0 * _A12_COEFFS[0] + a4 * _A12_COEFFS[1] + a8 * _A12_COEFFS[2]
    return np.array([a0, a4, a8, a12])


def _compute_variance_and_grad(x):
    """
    Compute the variance (uncertainty product) and its gradient with respect to (a0, a4, a8).
    
    Variance = (a^T M a) / (a^T D a) where D = diag(W)
    """
    a0, a4, a8 = x
    a = _get_all_coeffs(a0, a4, a8)
    
    # a^T D a = sum of a_i^2 * W_i
    norm_sq = np.dot(a * a, _W)
    
    if norm_sq <= 1e-30:
        return np.inf, np.zeros(3)
    
    # a^T M a
    second_moment = a @ _M @ a
    
    variance = second_moment / norm_sq
    
    # Gradient: d/dx_j [ (a^T M a) / (a^T D a) ]
    # = [ (da/dx_j)^T (M + M^T) a * norm_sq - (a^T M a) * 2 * (da/dx_j)^T D a ] / norm_sq^2
    # Simplified: let N = a^T D a, S = a^T M a
    # dS/dx_j = 2 * (da/dx_j)^T M a (since M is symmetric)
    # dN/dx_j = 2 * (da/dx_j)^T D a
    # d(S/N)/dx_j = (dS/dx_j * N - S * dN/dx_j) / N^2
    
    # da/dx_j for j=0,1,2 (a0, a4, a8)
    # a = [a0, a4, a8, a12] where a12 = a0*c0 + a4*c1 + a8*c2
    # da/da0 = [1, 0, 0, c0]
    # da/da4 = [0, 1, 0, c1]
    # da/da8 = [0, 0, 1, c2]
    
    c0, c1, c2 = _A12_COEFFS
    da_dx = np.array([
        [1.0, 0.0, 0.0, c0],
        [0.0, 1.0, 0.0, c1],
        [0.0, 0.0, 1.0, c2]
    ])
    
    # Precompute M @ a and D @ a
    Ma = _M @ a
    Da = _W * a  # D @ a
    
    S = second_moment
    N = norm_sq
    
    grad = np.zeros(3)
    for j in range(3):
        dadx_j = da_dx[j]
        dS_dx = 2.0 * dadx_j @ Ma
        dN_dx = 2.0 * dadx_j @ Da
        grad[j] = (dS_dx * N - S * dN_dx) / (N * N)
    
    return variance, grad


def _objective(x):
    """Objective function for optimization."""
    val, _ = _compute_variance_and_grad(x)
    return val


def _objective_grad(x):
    """Objective function with gradient for L-BFGS-B."""
    val, grad = _compute_variance_and_grad(x)
    return val, grad


def construct():
    """
    Find the coefficients that minimize the uncertainty product.
    Returns array of three coefficients [a0, a4, a8].
    """
    deadline = time.time() + 170.0  # Safety margin before 180s limit
    
    # Initial guess from the seed program
    x0 = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    
    best_x = x0.copy()
    best_val = _objective(x0)
    
    # Try multiple starting points and optimization methods
    start_points = [
        x0,
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, -0.1, 0.001]),
        np.array([-1.0, 0.1, -0.001]),
        np.array([0.5, -0.05, 0.0005]),
        np.array([-0.5, 0.05, -0.0005]),
        np.array([2.0, -0.2, 0.002]),
        np.array([-2.0, 0.2, -0.002]),
        np.array([0.1, -0.01, 0.0001]),
        np.array([-0.1, 0.01, -0.0001]),
    ]
    
    for sp in start_points:
        if time.time() > deadline:
            break
        
        try:
            result = minimize(
                _objective_grad,
                sp,
                method='L-BFGS-B',
                jac=True,
                bounds=[(-100, 100), (-100, 100), (-100, 100)],
                options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            
            if result.success and result.fun < best_val:
                best_val = result.fun
                best_x = result.x
        except Exception:
            continue
    
    # Final polish with tighter tolerance if time remains
    if time.time() < deadline - 10.0:
        try:
            result = minimize(
                _objective_grad,
                best_x,
                method='L-BFGS-B',
                jac=True,
                bounds=[(-1000, 1000), (-1000, 1000), (-1000, 1000)],
                options={'maxiter': 5000, 'ftol': 1e-16, 'gtol': 1e-14}
            )
            
            if result.success and result.fun < best_val:
                best_val = result.fun
                best_x = result.x
        except Exception:
            pass
    
    # Verify the result is valid
    if not np.all(np.isfinite(best_x)):
        best_x = x0
    
    return best_x
# EVOLVE-BLOCK-END
