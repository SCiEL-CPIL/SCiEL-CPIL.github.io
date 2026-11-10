# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.special import lpmv, factorial
from scipy.optimize import minimize

K = 10
N = 2 * K + 1  # 21 coefficients


def _build_quadrature():
    """Build 70-point Gauss-Legendre in cos(theta) x 130 phi grid."""
    x_gl, w_gl = np.polynomial.legendre.leggauss(70)
    # x_gl are in [-1, 1], these are cos(theta) values
    n_phi = 130
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    # Weights: for cos(theta) GL, weight is w_gl; for phi, weight is 2*pi/n_phi
    # Surface measure: dOmega = d(cos theta) * d(phi)
    # So total weight for point (i,j) is w_gl[i] * (2*pi/n_phi)
    w_phi = 2 * np.pi / n_phi
    # Build full weight matrix
    weights = np.outer(w_gl, np.full(n_phi, w_phi))
    return x_gl, phi, weights


def _build_harmonics(x_gl, phi):
    """
    Build the 21 real orthonormal degree-10 spherical harmonics evaluated
    on the quadrature grid. Returns Y of shape (n_points, 21).
    """
    n_theta = len(x_gl)
    n_phi = len(phi)
    n_pts = n_theta * n_phi
    
    # Y_m for m = -10..10
    # Y_0 = N_0 P_10^0
    # Y_m = sqrt(2) N_m P_10^m cos(m*phi) for m > 0
    # Y_m = sqrt(2) N_|m| P_10^|m| sin(|m|*phi) for m < 0
    # N_m = sqrt(21/(4*pi) * (10-m)!/(10+m)!)
    
    Y = np.zeros((n_pts, N), dtype=np.float64)
    
    for m_idx, m in enumerate(range(-K, K + 1)):
        am = abs(m)
        # Normalization
        N_m = np.sqrt(21.0 / (4.0 * np.pi) * factorial(10 - am) / factorial(10 + am))
        
        # Associated Legendre function with Condon-Shortley phase
        # lpmv(m, l, x) includes the Condon-Shortley phase (-1)^m
        P = lpmv(am, 10, x_gl)  # shape (n_theta,)
        
        if m == 0:
            val_theta = N_m * P  # shape (n_theta,)
            val = np.tile(val_theta[:, np.newaxis], (1, n_phi))
        elif m > 0:
            val_theta = np.sqrt(2.0) * N_m * P  # shape (n_theta,)
            cos_m_phi = np.cos(m * phi)  # shape (n_phi,)
            val = np.outer(val_theta, cos_m_phi)
        else:
            val_theta = np.sqrt(2.0) * N_m * P  # shape (n_theta,)
            sin_am_phi = np.sin(am * phi)  # shape (n_phi,)
            val = np.outer(val_theta, sin_am_phi)
        
        Y[:, m_idx] = val.ravel()
    
    return Y


# Precompute quadrature and harmonics at module level
_x_gl, _phi, _weights = _build_quadrature()
_Y = _build_harmonics(_x_gl, _phi)
_W = _weights.ravel()  # shape (n_pts,)


def _compute_ratio(coeffs):
    """
    Compute R = ||f||_4 / ||f||_2 where f = sum c_m Y_m.
    """
    f = _Y @ coeffs  # shape (n_pts,)
    f2 = f ** 2
    f4 = f2 ** 2
    
    norm2_sq = np.sum(_W * f2)  # = ||f||_2^2
    norm4_sq = np.sum(_W * f4)  # = ||f||_4^2
    
    if norm2_sq <= 0:
        return np.inf
    
    norm2 = np.sqrt(norm2_sq)
    norm4 = np.sqrt(np.sqrt(norm4_sq))  # ||f||_4 = (||f||_4^2)^(1/2) = (sum w f^4)^(1/4)
    
    return norm4 / norm2


def _objective_and_grad(coeffs):
    """
    Objective: R = ||f||_4 / ||f||_2.
    We minimize R.
    
    Gradient derivation:
    Let A = ||f||_4^2 = sum w f^4, B = ||f||_2^2 = sum w f^2
    R = A^{1/4} / B^{1/2}
    
    dR/df = (1/4) A^{-3/4} * 4 w f^3 / B^{1/2} - A^{1/4} * (1/2) B^{-3/2} * 2 w f
          = A^{-3/4} B^{-1/2} w f^3 - A^{1/4} B^{-3/2} w f
    
    dc_m = dR/df * Y_m
    """
    f = _Y @ coeffs
    f2 = f ** 2
    f4 = f2 ** 2
    
    A = np.sum(_W * f4)  # ||f||_4^2
    B = np.sum(_W * f2)  # ||f||_2^2
    
    if A <= 0 or B <= 0:
        return np.inf, np.zeros_like(coeffs)
    
    A_quarter = A ** 0.25
    B_half = np.sqrt(B)
    
    R = A_quarter / B_half
    
    # dR/df
    dR_df = (f3_coeff := f ** 3) * _W * (A ** (-0.75)) / B_half - f * _W * A_quarter * (B ** (-1.5))
    
    # Actually let me redo:
    # R = A^(1/4) * B^(-1/2)
    # dR/dA = (1/4) A^(-3/4) B^(-1/2)
    # dR/dB = -(1/2) A^(1/4) B^(-3/2)
    # dA/df = 4 w f^3
    # dB/df = 2 w f
    # dR/df = (1/4) A^(-3/4) B^(-1/2) * 4 w f^3 + (-(1/2) A^(1/4) B^(-3/2)) * 2 w f
    #        = A^(-3/4) B^(-1/2) w f^3 - A^(1/4) B^(-3/2) w f
    
    dR_df = A ** (-0.75) * B ** (-0.5) * _W * f ** 3 - A ** 0.25 * B ** (-1.5) * _W * f
    
    grad = _Y.T @ dR_df
    
    return R, grad


def rudin_problem10() -> np.ndarray:
    """
    Find coefficients c_m for m = -10..10 minimizing R = ||f||_4 / ||f||_2
    for real degree-10 spherical harmonics.
    """
    t_start = time.time()
    deadline = t_start + 600.0  # safety margin
    
    # Start with all-ones
    best_coeffs = np.ones(N, dtype=np.float64)
    best_R = _compute_ratio(best_coeffs)
    
    # Try several initializations and use L-BFGS-B with analytic gradient
    initializations = []
    
    # 1. All ones
    initializations.append(np.ones(N))
    
    # 2. Only m=0
    c0 = np.zeros(N)
    c0[K] = 1.0
    initializations.append(c0)
    
    # 3. Random (fixed seed for reproducibility)
    rng = np.random.default_rng(42)
    for _ in range(5):
        initializations.append(rng.standard_normal(N))
    
    # 4. Alternating signs
    c_alt = np.ones(N)
    c_alt[::2] = -1.0
    initializations.append(c_alt)
    
    # 5. Cosine-weighted
    m_vals = np.arange(-K, K + 1)
    c_cos = np.cos(m_vals * 0.5)
    initializations.append(c_cos)
    
    # 6. Sine-weighted
    c_sin = np.sin(m_vals * 0.3)
    initializations.append(c_sin)
    
    for init in initializations:
        if time.time() > deadline - 30:
            break
        
        try:
            res = minimize(
                _objective_and_grad,
                init,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
            )
            if res.success or res.fun < best_R:
                R_val = res.fun
                if np.isfinite(R_val) and R_val < best_R:
                    best_R = R_val
                    best_coeffs = res.x.copy()
        except Exception:
            pass
    
    # Final polish with tighter tolerance
    if time.time() < deadline - 10:
        try:
            res = minimize(
                _objective_and_grad,
                best_coeffs,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-10, 'maxls': 100}
            )
            if res.fun < best_R and np.isfinite(res.fun):
                best_R = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass
    
    # Validate
    if (best_coeffs.size != 21 or 
        not np.all(np.isfinite(best_coeffs)) or
        np.max(np.abs(best_coeffs)) < 1e-3 or
        np.max(np.abs(best_coeffs)) > 1e5):
        # Fallback to all ones
        best_coeffs = np.ones(N, dtype=np.float64)
    
    return best_coeffs.astype(np.float64)


# EVOLVE-BLOCK-END
