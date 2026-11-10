# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.special import lpmv
from scipy.optimize import minimize

_DEADLINE = 620.0  # safety margin before 630s kill
_START_TIME = None


def _precompute_basis():
    """
    Precompute the 70x130 quadrature grid and the 21 basis functions evaluated
    at every grid point. Returns (X, Y, Z, weights, basis) where basis has shape
    (21, 70, 130).
    """
    n_theta = 70
    n_phi = 130

    # Gauss-Legendre nodes and weights on [-1, 1]
    x_gl, w_gl = np.polynomial.legendre.leggauss(n_theta)
    cos_theta = x_gl
    sin_theta = np.sqrt(np.maximum(1.0 - cos_theta**2, 0.0))

    # Weights for surface integral: dOmega = sin(theta) dtheta dphi
    # With t = cos(theta), dtheta = -dt/sin(theta), so sin(theta)dtheta = -dt
    # Integral over sphere = sum over phi of sum over t of w(t) * f(t, phi)
    # The total measure should be 4*pi.
    # sum(w_gl) = 2 (for [-1,1]), sum over phi of dphi = 2*pi
    # So total = 2 * 2*pi = 4*pi. Good.
    weights = np.tile(w_gl, (n_phi, 1)) * (2.0 * np.pi / n_phi)
    weights = weights.T  # shape (n_theta, n_phi)

    # Phi grid
    phi = np.linspace(0, 2.0 * np.pi, n_phi, endpoint=False)

    # Build 3D grid
    theta = np.arccos(cos_theta)
    Theta, Phi = np.meshgrid(theta, phi, indexing='ij')
    X = sin_theta[:, None] * np.cos(Phi)
    Y = sin_theta[:, None] * np.sin(Phi)
    Z = cos_theta[:, None] * np.ones((1, n_phi))

    # Evaluate basis functions
    k = 10
    basis = np.zeros((2 * k + 1, n_theta, n_phi), dtype=np.float64)

    for m in range(-k, k + 1):
        am = abs(m)
        # Associated Legendre P_l^m(x) with Condon-Shortley phase
        # scipy.special.lpmv(m, l, x) includes (-1)^m factor
        P = lpmv(am, k, cos_theta)  # shape (n_theta,)

        # Normalization constant N_m
        # N_m = sqrt(21/(4*pi) * (10-m)!/(10+m)!)
        # Using log-gamma for numerical stability
        log_N = 0.5 * (np.log(21.0) - np.log(4.0 * np.pi)
                       + np.math.lgamma(k - am + 1) - np.math.lgamma(k + am + 1))
        N = np.exp(log_N)

        if m == 0:
            basis[m + k] = N * P[:, None] * np.ones((1, n_phi))
        elif m > 0:
            basis[m + k] = np.sqrt(2.0) * N * P[:, None] * np.cos(m * Phi)
        else:
            basis[m + k] = np.sqrt(2.0) * N * P[:, None] * np.sin(am * Phi)

    return X, Y, Z, weights, basis


def _compute_R_from_precomputed(coeffs, weights, basis):
    """
    Compute R = ||f||_4 / ||f||_2 using precomputed basis.
    """
    # f = sum_m c_m * basis[m], shape (n_theta, n_phi)
    f = np.tensordot(coeffs, basis, axes=(0, 0))

    f2 = f * f
    f4 = f2 * f2

    # Integrate
    int_f2 = np.sum(f2 * weights)
    int_f4 = np.sum(f4 * weights)

    norm2 = np.sqrt(int_f2)
    norm4 = np.power(int_f4, 0.25)

    if norm2 < 1e-15:
        return np.inf

    return norm4 / norm2


def _compute_R_and_grad(coeffs, weights, basis):
    """
    Compute R and its gradient with respect to coeffs.
    R = ||f||_4 / ||f||_2

    Let I2 = ∫f², I4 = ∫f⁴.
    ||f||_2 = I2^{1/2}, ||f||_4 = I4^{1/4}.
    R = I4^{1/4} / I2^{1/2}.

    dR/dc_m = (1/R) * [ (1/4) I4^{-3/4} * dI4/dc_m / I2^{1/2}
                       - I4^{1/4} * (1/2) I2^{-3/2} * dI2/dc_m ]
            = (1/R) * [ (1/4) I4^{-1} dI4/dc_m * (I4/I2)^{1/2} * ... ]

    Let's do it more carefully:
    R = I4^{1/4} * I2^{-1/2}
    dR/dc_m = (1/4) I4^{-3/4} * dI4/dc_m * I2^{-1/2}
             + I4^{1/4} * (-1/2) I2^{-3/2} * dI2/dc_m

    dI2/dc_m = ∫ 2f * basis_m * dOmega = 2 ∫ f * basis_m * w
    dI4/dc_m = ∫ 4f³ * basis_m * dOmega = 4 ∫ f³ * basis_m * w
    """
    f = np.tensordot(coeffs, basis, axes=(0, 0))
    f2 = f * f
    f3 = f2 * f
    f4 = f2 * f2

    int_f2 = np.sum(f2 * weights)
    int_f4 = np.sum(f4 * weights)

    if int_f2 < 1e-30:
        return np.inf, np.zeros_like(coeffs)

    norm2 = np.sqrt(int_f2)
    norm4 = np.power(int_f4, 0.25)
    R = norm4 / norm2

    # Gradient
    # dI2/dc_m = 2 * sum(f * basis_m * w)
    # dI4/dc_m = 4 * sum(f^3 * basis_m * w)

    # For efficiency, compute dot products
    # grad_I2[m] = 2 * sum(f * basis[m] * weights)
    # grad_I4[m] = 4 * sum(f^3 * basis[m] * weights)

    # f * weights shape (n_theta, n_phi)
    fw = f * weights
    f3w = f3 * weights

    grad_I2 = 2.0 * np.array([np.sum(fw * basis[m]) for m in range(len(coeffs))])
    grad_I4 = 4.0 * np.array([np.sum(f3w * basis[m]) for m in range(len(coeffs))])

    # dR/dc_m = (1/4) I4^{-3/4} * I2^{-1/2} * grad_I4[m]
    #           - (1/2) I4^{1/4} * I2^{-3/2} * grad_I2[m]
    term1 = 0.25 * np.power(int_f4, -0.75) * np.power(int_f2, -0.5) * grad_I4
    term2 = 0.5 * np.power(int_f4, 0.25) * np.power(int_f2, -1.5) * grad_I2

    grad_R = term1 - term2

    return R, grad_R


def _objective_and_grad(coeffs, weights, basis):
    """Objective and gradient for scipy.optimize.minimize."""
    R, grad = _compute_R_and_grad(coeffs, weights, basis)
    if not np.isfinite(R):
        return 1e10, np.zeros_like(coeffs)
    return R, grad


def rudin_problem10() -> np.ndarray:
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()

    k = 10
    n_coeffs = 2 * k + 1

    # Precompute basis
    X, Y, Z, weights, basis = _precompute_basis()

    # Normalize basis rows so that each basis function has unit L2 norm
    # (they should already be orthonormal, but let's verify)
    # Actually, the basis is orthonormal by construction, so sum_m c_m^2 = ||f||_2^2 / (4*pi)?
    # No. The basis functions are orthonormal: ∫ Y_m Y_n dOmega = delta_mn.
    # So ||f||_2^2 = sum_m c_m^2.
    # And R = ||f||_4 / ||f||_2.

    best_R = np.inf
    best_coeffs = np.ones(n_coeffs, dtype=np.float64)

    # Generate diverse initial points
    rng = np.random.RandomState(42)

    # Strategy: multiple restarts with L-BFGS-B
    # Use symmetry: R is invariant under sign flips of individual coefficients
    # and under the symmetry of the spherical harmonics.
    # We can restrict to c_m >= 0 for all m, and also use the fact that
    # the problem is invariant under m -> -m (swap cos and sin terms).

    # Let's try several initialization strategies:
    # 1. All ones
    # 2. Random with fixed seed
    # 3. Based on known optimal structures (e.g., only even m, or specific patterns)

    init_list = []

    # All ones
    init_list.append(np.ones(n_coeffs))

    # Random initializations
    for i in range(20):
        c = rng.randn(n_coeffs)
        c = np.abs(c)  # exploit sign symmetry
        if np.max(c) > 0:
            c /= np.max(c)
        init_list.append(c)

    # Symmetric patterns: only m=0, only even m, etc.
    # m=0 only
    c0 = np.zeros(n_coeffs)
    c0[k] = 1.0
    init_list.append(c0)

    # Even m only: m = -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10
    ce = np.zeros(n_coeffs)
    for m in range(-k, k + 1, 2):
        ce[m + k] = 1.0
    init_list.append(ce)

    # Odd m only
    co = np.zeros(n_coeffs)
    for m in range(-k + 1, k + 1, 2):
        co[m + k] = 1.0
    init_list.append(co)

    # m=0 and m=±10
    c10 = np.zeros(n_coeffs)
    c10[0] = 1.0
    c10[k] = 1.0
    c10[-1] = 1.0
    init_list.append(c10)

    # m=±10 only
    c10b = np.zeros(n_coeffs)
    c10b[0] = 1.0
    c10b[-1] = 1.0
    init_list.append(c10b)

    # m=0 and m=±8
    c8 = np.zeros(n_coeffs)
    c8[k - 8] = 1.0
    c8[k] = 1.0
    c8[k + 8] = 1.0
    init_list.append(c8)

    # m=±4 only
    c4 = np.zeros(n_coeffs)
    c4[k - 4] = 1.0
    c4[k + 4] = 1.0
    init_list.append(c4)

    # m=0, ±2, ±4, ±6, ±8 (even, excluding ±10)
    ce2 = np.zeros(n_coeffs)
    for m in range(-8, 9, 2):
        ce2[m + k] = 1.0
    init_list.append(ce2)

    # m=±1, ±3, ±5, ±7, ±9 (odd, excluding 0)
    co2 = np.zeros(n_coeffs)
    for m in range(-9, 10, 2):
        if m != 0:
            co2[m + k] = 1.0
    init_list.append(co2)

    # Run L-BFGS-B from each initialization
    for init in init_list:
        if time.time() - _START_TIME > _DEADLINE - 10:
            break

        try:
            res = minimize(
                _objective_and_grad,
                init,
                args=(weights, basis),
                method='L-BFGS-B',
                jac=True,
                options={
                    'maxiter': 2000,
                    'ftol': 1e-15,
                    'gtol': 1e-12,
                    'maxls': 50,
                    'maxfun': 10000,
                }
            )

            if res.success or res.fun < best_R:
                R_val = res.fun
                if np.isfinite(R_val) and R_val < best_R:
                    best_R = R_val
                    best_coeffs = res.x.copy()

        except Exception:
            continue

    # Final polish: run more iterations from the best found
    if time.time() - _START_TIME < _DEADLINE - 20:
        try:
            res = minimize(
                _objective_and_grad,
                best_coeffs,
                args=(weights, basis),
                method='L-BFGS-B',
                jac=True,
                options={
                    'maxiter': 5000,
                    'ftol': 1e-18,
                    'gtol': 1e-15,
                    'maxls': 100,
                    'maxfun': 50000,
                }
            )
            if np.isfinite(res.fun) and res.fun < best_R:
                best_R = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass

    # Ensure validity
    if not np.all(np.isfinite(best_coeffs)):
        best_coeffs = np.ones(n_coeffs)

    max_abs = np.max(np.abs(best_coeffs))
    if max_abs < 1e-3:
        best_coeffs = np.ones(n_coeffs)
    elif max_abs > 1e5:
        best_coeffs = best_coeffs / max_abs * 1e5

    return best_coeffs


# EVOLVE-BLOCK-END
