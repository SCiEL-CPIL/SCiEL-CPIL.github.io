# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.special import lpmv
import math


def _build_quadrature(k=10):
    """Build 70-point Gauss-Legendre in cos(theta) x 130 equally spaced phi."""
    # Gauss-Legendre nodes and weights for 70 points on [-1, 1]
    x, w = np.polynomial.legendre.leggauss(70)
    # x = cos(theta), w are the weights
    # phi: 130 equally spaced
    n_phi = 130
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    dphi = 2 * np.pi / n_phi
    # Surface element: dOmega = dphi * d(cos theta)
    # So weight for each (x_i, phi_j) is w_i * dphi
    # Total mass should be 4*pi: sum(w) * dphi = 2 * dphi * 130 = 2 * 2*pi = 4*pi. Good.
    return x, w, phi, dphi


def _compute_harmonics(x, phi, k=10):
    """
    Compute real orthonormal spherical harmonics Y_m for m = -k..k at points (x=cos theta, phi).
    Y_0 = N_0 * P_k^0
    Y_m = sqrt(2) * N_m * P_k^m * cos(m*phi) for m > 0
    Y_m = sqrt(2) * N_|m| * P_k^|m| * sin(|m|*phi) for m < 0
    N_m = sqrt((2k+1)/(4*pi) * (k-m)!/(k+m)!)
    
    Returns array of shape (n_x * n_phi, 2*k+1)
    """
    n_x = len(x)
    n_phi = len(phi)
    n_m = 2 * k + 1
    Y = np.zeros((n_x, n_phi, n_m), dtype=np.float64)
    
    for m in range(-k, k + 1):
        abs_m = abs(m)
        # Normalization: N_m = sqrt((2k+1)/(4*pi) * (k-m)!/(k+m)!)
        # Use log-gamma for stability
        log_num = math.lgamma(k + 1 - abs_m)
        log_den = math.lgamma(k + 1 + abs_m)
        N_m = math.sqrt((2 * k + 1) / (4 * np.pi) * math.exp(log_num - log_den))
        
        # Associated Legendre with Condon-Shortley phase
        # scipy.special.lpmv(m, l, x) includes Condon-Shortley phase (-1)^m
        P = lpmv(abs_m, k, x)  # shape (n_x,)
        
        if m == 0:
            Y[:, :, 10] = N_m * P[:, np.newaxis]
        elif m > 0:
            Y[:, :, 10 + m] = math.sqrt(2) * N_m * P[:, np.newaxis] * np.cos(m * phi)[np.newaxis, :]
        else:
            Y[:, :, 10 + m] = math.sqrt(2) * N_m * P[:, np.newaxis] * np.sin(abs_m * phi)[np.newaxis, :]
    
    return Y.reshape(n_x * n_phi, n_m)


def _compute_ratio(coeffs, Y, w, dphi):
    """Compute ||f||_4 / ||f||_2 using quadrature."""
    # f = Y @ coeffs, shape (n_points,)
    f = Y @ coeffs
    # Weights for quadrature: w_i * dphi for each (x_i, phi_j)
    # Y is (n_x*n_phi, n_m), so reshape w
    n_x = len(w)
    n_phi = Y.shape[0] // n_x
    weights_full = np.tile(w * dphi, n_phi)
    
    f2 = f * f
    f4 = f2 * f2
    
    norm2_sq = np.sum(weights_full * f2)
    norm4_sq = np.sum(weights_full * f4)
    
    if norm2_sq <= 0:
        return np.inf
    
    norm2 = np.sqrt(norm2_sq)
    norm4 = np.sqrt(np.sqrt(norm4_sq))  # ||f||_4 = (integral |f|^4)^{1/4}
    
    return norm4 / norm2


def _objective(coeffs, Y, w, dphi):
    """Objective to minimize: the L4/L2 ratio."""
    return _compute_ratio(coeffs, Y, w, dphi)


def rudin_problem10() -> np.ndarray:
    k = 10
    n_m = 2 * k + 1
    
    # Build quadrature
    x, w, phi, dphi = _build_quadrature(k)
    
    # Compute harmonics at all quadrature points
    Y = _compute_harmonics(x, phi, k)
    
    best_coeffs = None
    best_ratio = np.inf
    
    # Approach: Global Search (Differential Evolution) followed by Local Polish (L-BFGS-B)
    # Differential Evolution is a population-based stochastic method that explores the
    # parameter space more broadly than simple random restarts of local optimizers.
    
    # Define bounds for Differential Evolution
    # The scale of coefficients doesn't matter for the ratio, but DE needs finite bounds.
    # We can constrain the sum of squares to 1 or just use a large box.
    # Using a box of [-10, 10] is usually sufficient as the ratio is scale-invariant.
    bounds = [(-10.0, 10.0)] * n_m
    
    # Run Differential Evolution
    # maxiter=100, popsize=15 is a reasonable tradeoff for 21 dims.
    try:
        de_result = differential_evolution(
            _objective,
            bounds,
            args=(Y, w, dphi),
            maxiter=100,
            popsize=15,
            tol=1e-8,
            seed=42,
            polish=True,  # DE will run a local polish at the end by default if polish=True
            init='latinhypercube'
        )
        
        if de_result.success or de_result.fun < best_ratio:
            best_ratio = de_result.fun
            best_coeffs = de_result.x.copy()
    except Exception:
        pass

    # Additional Local Polish from the best point found by DE, just in case DE's polish wasn't aggressive enough
    if best_coeffs is not None:
        result = minimize(
            _objective,
            best_coeffs,
            args=(Y, w, dphi),
            method='L-BFGS-B',
            options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-14}
        )
        if result.fun < best_ratio:
            best_ratio = result.fun
            best_coeffs = result.x.copy()

    # Safety net: If DE failed or didn't find anything, try a few random starts
    if best_coeffs is None:
        rng = np.random.default_rng(42)
        for _ in range(10):
            coeffs0 = rng.standard_normal(n_m)
            result = minimize(
                _objective,
                coeffs0,
                args=(Y, w, dphi),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-14}
            )
            if result.fun < best_ratio:
                best_ratio = result.fun
                best_coeffs = result.x.copy()

    if best_coeffs is None:
        return np.ones(n_m, dtype=np.float64)
    
    coeffs = best_coeffs
    
    # Validate
    max_abs = np.max(np.abs(coeffs))
    if max_abs < 1e-3 or max_abs > 1e5:
        # Rescale
        if max_abs > 0:
            coeffs = coeffs / max_abs
        else:
            return np.ones(n_m, dtype=np.float64)
    
    return coeffs.astype(np.float64)


# EVOLVE-BLOCK-END
