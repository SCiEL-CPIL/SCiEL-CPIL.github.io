import numpy as np
from scipy.special import lpmv, factorial
from scipy.optimize import minimize, differential_evolution
import time

def _build_harmonics():
    """Precompute real orthonormal degree-10 spherical harmonics on the quadrature grid."""
    k = 10
    # 70-point Gauss-Legendre in cos(theta), 130 equally spaced phi
    x, w = np.polynomial.legendre.leggauss(70)
    # x in [-1,1] is cos(theta), w are weights
    # phi equally spaced
    phi = np.linspace(0, 2*np.pi, 130, endpoint=False)
    
    # Build 2D grid
    cos_theta, phi_grid = np.meshgrid(x, phi, indexing='ij')
    # Surface measure: dOmega = dphi * d(cos_theta), so weights are w_i * (2*pi/130)
    # Actually the quadrature: integral = sum_i sum_j f(x_i, phi_j) * w_i * (2*pi/130)
    dphi = 2*np.pi / 130
    weights_2d = np.outer(w, np.ones(130)) * dphi
    
    # Compute P_10^m(cos_theta) with Condon-Shortley phase
    # scipy.special.lpmv(m, l, x) includes Condon-Shortley phase (-1)^m
    P = {}
    for m in range(0, 11):
        P[m] = lpmv(m, 10, cos_theta)  # shape (70, 130)
    
    # Normalization: N_m = sqrt(21/(4*pi) * (10-m)!/(10+m)!)
    N = {}
    for m in range(0, 11):
        N[m] = np.sqrt(21.0/(4*np.pi) * factorial(10-m)/factorial(10+m))
    
    # Build real orthonormal harmonics Y_m for m = -10..10
    # Y_0 = N_0 * P_10^0
    # Y_m = sqrt(2) * N_m * P_10^m * cos(m*phi) for m > 0
    # Y_m = sqrt(2) * N_|m| * P_10^|m| * sin(|m|*phi) for m < 0
    Y = np.zeros((21, 70, 130))
    for idx, m in enumerate(range(-10, 11)):
        if m == 0:
            Y[idx] = N[0] * P[0]
        elif m > 0:
            Y[idx] = np.sqrt(2) * N[m] * P[m] * np.cos(m * phi_grid)
        else:
            am = abs(m)
            Y[idx] = np.sqrt(2) * N[am] * P[am] * np.sin(am * phi_grid)
    
    return Y, weights_2d

def _objective(coeffs, Y, weights_2d):
    """Compute ||f||_4 / ||f||_2."""
    f = np.tensordot(coeffs, Y, axes=(0, 0))  # shape (70, 130)
    w = weights_2d
    norm2_sq = np.sum(f**2 * w)
    if norm2_sq < 1e-30:
        return 1e10
    norm2 = np.sqrt(norm2_sq)
    norm4 = np.sum(np.abs(f)**4 * w)**0.25
    return norm4 / norm2

def _objective_and_grad(coeffs, Y, weights_2d):
    """Compute objective and gradient."""
    f = np.tensordot(coeffs, Y, axes=(0, 0))
    w = weights_2d
    norm2_sq = np.sum(f**2 * w)
    if norm2_sq < 1e-30:
        return 1e10, np.zeros_like(coeffs)
    norm2 = np.sqrt(norm2_sq)
    norm4 = np.sum(np.abs(f)**4 * w)**0.25
    
    # d||f||4/dcoeffs = (1/4) * ||f||4^(-3) * 4 * sum(f^3 * Y_m * w) = ||f||4^(-3) * sum(f^3 * Y_m * w)
    # d||f||2/dcoeffs = (1/2) * ||f||2^(-1) * 2 * sum(f * Y_m * w) = ||f||2^(-1) * sum(f * Y_m * w)
    # d(norm4/norm2)/dc = (1/norm2) * dnorm4/dc - (norm4/norm2^2) * dnorm2_dc
    
    f3Y = np.tensordot(f**3, Y, axes=(0, 0))  # shape (21, 70, 130)
    fY = np.tensordot(f, Y, axes=(0, 0))       # shape (21, 70, 130)
    
    dnorm4_dc = np.sum(f3Y * w[None, :, :], axis=(1, 2)) / (norm4**3)
    dnorm2_dc = np.sum(fY * w[None, :, :], axis=(1, 2)) / norm2
    
    grad = dnorm4_dc / norm2 - (norm4 / norm2**2) * dnorm2_dc
    
    return norm4 / norm2, grad

def rudin_problem10() -> np.ndarray:
    """
    Find coefficients for degree-10 spherical harmonic minimizing L4/L2 ratio.
    """
    start_time = time.time()
    deadline = 600.0  # seconds, with safety margin
    
    # Precompute harmonics and weights
    Y, weights_2d = _build_harmonics()
    
    # Verify orthonormality
    # sum_m Y_m * Y_n * w = delta_mn
    # This is a sanity check
    
    # Initial guess: all ones
    best_coeffs = np.ones(21)
    best_score = _objective(best_coeffs, Y, weights_2d)
    
    # Strategy 1: Try structured initial guesses targeting flatness
    # Key insight: to minimize ||f||4/||f||2, we want f to be as "flat" as possible
    # This means we want energy spread across many m values
    
    # Try: all m=0 (pure zonal)
    c0 = np.zeros(21)
    c0[10] = 1.0
    s0 = _objective(c0, Y, weights_2d)
    if s0 < best_score:
        best_score = s0
        best_coeffs = c0.copy()
    
    # Try: equal weight on all cos(m*phi) terms (m >= 0)
    c_cos = np.zeros(21)
    for m in range(0, 11):
        c_cos[10 + m] = 1.0 / np.sqrt(11)
    s_cos = _objective(c_cos, Y, weights_2d)
    if s_cos < best_score:
        best_score = s_cos
        best_coeffs = c_cos.copy()
    
    # Try: equal weight on all sin(|m|*phi) terms (m < 0)
    c_sin = np.zeros(21)
    for m in range(1, 11):
        c_sin[10 - m] = 1.0 / np.sqrt(10)
    s_sin = _objective(c_sin, Y, weights_2d)
    if s_sin < best_score:
        best_score = s_sin
        best_coeffs = c_sin.copy()
    
    # Try: equal weight on all 21 modes
    c_all = np.ones(21) / np.sqrt(21)
    s_all = _objective(c_all, Y, weights_2d)
    if s_all < best_score:
        best_score = s_all
        best_coeffs = c_all.copy()
    
    # Try: high-m dominance (tends to be flatter)
    c_high = np.zeros(21)
    for m in [8, 9, 10]:
        c_high[10 + m] = 1.0 / np.sqrt(3)
        c_high[10 - m] = 1.0 / np.sqrt(3)
    s_high = _objective(c_high, Y, weights_2d)
    if s_high < best_score:
        best_score = s_high
        best_coeffs = c_high.copy()
    
    # Try: alternating signs to create interference patterns
    c_alt = np.zeros(21)
    for m in range(0, 11):
        c_alt[10 + m] = (1.0 if m % 2 == 0 else -1.0) / np.sqrt(11)
    s_alt = _objective(c_alt, Y, weights_2d)
    if s_alt < best_score:
        best_score = s_alt
        best_coeffs = c_alt.copy()
    
    # Try random starts with local optimization
    np.random.seed(42)
    
    # Multiple random restarts with L-BFGS-B using gradient
    n_restarts = 80
    for i in range(n_restarts):
        if time.time() - start_time > deadline - 90:
            break
        
        # Generate diverse initial points
        if i < 20:
            # Sparse: pick random subset of modes
            n_active = np.random.randint(3, 15)
            active_idx = np.random.choice(21, size=n_active, replace=False)
            x0 = np.zeros(21)
            x0[active_idx] = np.random.randn(n_active) * 2.0
        elif i < 40:
            # Dense with random signs
            x0 = np.random.choice([-1, 1], size=21) * np.abs(np.random.randn(21))
            x0 *= 2.0 / (np.max(np.abs(x0)) + 1e-10)
        else:
            # Random Gaussian
            x0 = np.random.randn(21) * 2.0
        
        try:
            res = minimize(
                lambda c: _objective_and_grad(c, Y, weights_2d),
                x0,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
            )
            if res.fun < best_score and np.isfinite(res.fun):
                best_score = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass
    
    # Strategy 2: Basin hopping / simulated annealing style local search
    if time.time() - start_time < deadline - 150:
        # Take current best and do multiple perturbation + polish cycles
        n_sa_rounds = 30
        for i in range(n_sa_rounds):
            if time.time() - start_time > deadline - 60:
                break
            
            # Perturb current best
            x0 = best_coeffs + np.random.randn(21) * 0.5
            try:
                res = minimize(
                    lambda c: _objective_and_grad(c, Y, weights_2d),
                    x0,
                    jac=True,
                    method='L-BFGS-B',
                    options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10}
                )
                if res.fun < best_score and np.isfinite(res.fun):
                    best_score = res.fun
                    best_coeffs = res.x.copy()
            except Exception:
                pass
    
    # Final polish with high-precision L-BFGS-B using gradient
    if time.time() - start_time < deadline - 30:
        try:
            res = minimize(
                lambda c: _objective_and_grad(c, Y, weights_2d),
                best_coeffs,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 10000, 'ftol': 1e-16, 'gtol': 1e-12, 'maxls': 100}
            )
            if res.fun < best_score and np.isfinite(res.fun):
                best_score = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass
        
        # One more round of small perturbations from the polished result
        for i in range(20):
            if time.time() - start_time > deadline - 15:
                break
            x0 = best_coeffs + np.random.randn(21) * 0.1
            try:
                res = minimize(
                    lambda c: _objective_and_grad(c, Y, weights_2d),
                    x0,
                    jac=True,
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-11}
                )
                if res.fun < best_score and np.isfinite(res.fun):
                    best_score = res.fun
                    best_coeffs = res.x.copy()
            except Exception:
                pass
    
    # Validate
    if len(best_coeffs) != 21 or not np.all(np.isfinite(best_coeffs)):
        return np.ones(21)
    max_abs = np.max(np.abs(best_coeffs))
    if max_abs < 1e-3 or max_abs > 1e5:
        # Scale to valid range
        if max_abs < 1e-3:
            best_coeffs *= 1.0 / max_abs if max_abs > 0 else 1.0
        elif max_abs > 1e5:
            best_coeffs *= 1e5 / max_abs
        if not np.all(np.isfinite(best_coeffs)):
            return np.ones(21)
    
    return best_coeffs.astype(np.float64)
