# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.special import lpmv, factorial
from scipy.optimize import minimize

# Precompute the quadrature and basis at module level for speed
_K = 10
_NTHETA = 70
_NPHI = 130

def _build_quadrature():
    """Build Gauss-Legendre quadrature in cos(theta) and uniform phi grid."""
    # Gauss-Legendre nodes and weights on [-1, 1]
    x, w = np.polynomial.legendre.leggauss(_NTHETA)
    # x = cos(theta), w are weights for integral over cos(theta) from -1 to 1
    # The surface measure is dOmega = dphi * (-d(cos theta)) = dphi * dx (with x=cos theta)
    # Actually integral over sphere: ∫ f dOmega = ∫_0^{2π} dφ ∫_{-1}^{1} f(cosθ, φ) d(cosθ)
    # So weights for cos(theta) are just w, and we sum over phi with weight 2π/NPHI
    
    # Create phi grid
    phi = np.linspace(0, 2 * np.pi, _NPHI, endpoint=False)
    
    # Create 2D grid
    X, PHI = np.meshgrid(x, phi, indexing='ij')  # X shape (NTHETA, NPHI), PHI shape (NTHETA, NPHI)
    
    # Quadrature weights: w_theta * dphi
    dphi = 2 * np.pi / _NPHI
    W = np.outer(w, np.ones(_NPHI)) * dphi  # shape (NTHETA, NPHI)
    
    return X, PHI, W

def _compute_basis_values(X, PHI):
    """
    Compute the 21 real orthonormal spherical harmonic values at quadrature points.
    Returns array B of shape (NTHETA, NPHI, 21) where B[:,:,i] corresponds to m = i - 10.
    
    Y_0 = N_0 P_10^0(cos theta)
    Y_m = sqrt(2) N_m P_10^m(cos theta) cos(m phi) for m > 0
    Y_m = sqrt(2) N_|m| P_10^|m|(cos theta) sin(|m| phi) for m < 0
    
    N_m = sqrt(21/(4π) * (10-m)!/(10+m)!)
    """
    n = _NTHETA
    m = _NPHI
    B = np.zeros((n, m, 2 * _K + 1))
    
    for mm in range(-_K, _K + 1):
        abs_mm = abs(mm)
        # Compute N_m
        # N_m = sqrt(21/(4π) * (10-m)!/(10+m)!)
        # Using log-gamma for stability
        log_num = np.log(21.0) - np.log(4.0 * np.pi) + np.log(factorial(10 - abs_mm, exact=False))
        log_den = np.log(factorial(10 + abs_mm, exact=False))
        N_mm = np.sqrt(np.exp(log_num - log_den))
        
        # Compute associated Legendre function P_10^|m|(cos theta) with Condon-Shortley phase
        # scipy.special.lpmv(m, n, x) includes the Condon-Shortley phase (-1)^m
        P = lpmv(abs_mm, _K, X)  # shape (NTHETA, NPHI)
        
        if mm == 0:
            B[:, :, abs_mm] = N_mm * P
        elif mm > 0:
            B[:, :, _K + mm] = np.sqrt(2.0) * N_mm * P * np.cos(mm * PHI)
        else:
            # mm < 0
            B[:, :, _K + mm] = np.sqrt(2.0) * N_mm * P * np.sin(abs_mm * PHI)
    
    return B

# Precompute at module level
_X, _PHI, _W = _build_quadrature()
_B = _compute_basis_values(_X, _PHI)  # shape (NTHETA, NPHI, 21)

# Flatten for matrix operations
# B_flat shape (21, NTHETA*NPHI)
_B_flat = _B.transpose(2, 0, 1).reshape(21, -1)  # (21, NTHETA*NPHI)
_W_flat = _W.flatten()  # (NTHETA*NPHI,)

def _objective_and_gradient(coeffs):
    """
    Compute R = ||f||_4 / ||f||_2 and its gradient.
    
    f = sum_m c_m Y_m, so at each quadrature point j:
    f_j = sum_m c_m B[m, j] = B_flat @ coeffs
    
    ||f||_2^2 = sum_j W_j f_j^2 = coeffs^T (B_flat^T diag(W) B_flat) coeffs
    ||f||_4^4 = sum_j W_j f_j^4
    
    R = ||f||_4 / ||f||_2 = (sum W_j f_j^4)^{1/4} / (sum W_j f_j^2)^{1/2}
    """
    f = _B_flat @ coeffs  # (NTHETA*NPHI,)
    
    norm2_sq = np.sum(_W_flat * f * f)
    norm4_4 = np.sum(_W_flat * f * f * f * f)
    
    if norm2_sq <= 0:
        return 1e10, np.zeros_like(coeffs)
    
    norm2 = np.sqrt(norm2_sq)
    norm4 = np.sqrt(np.sqrt(norm4_4))
    
    R = norm4 / norm2
    
    # Gradient of R with respect to coeffs
    # R = norm4 / norm2
    # dR/dcoeffs = (d(norm4)/dcoeffs * norm2 - norm4 * d(norm2)/dcoeffs) / norm2^2
    #            = (d(norm4)/dcoeffs / norm2) - (norm4 / norm2^2) * d(norm2)/dcoeffs
    #            = (d(norm4)/dcoeffs / norm2) - (norm4 / norm2) * (d(norm2)/dcoeffs / norm2)
    #            = (1/norm2) * d(norm4)/dcoeffs - (R / norm2) * d(norm2)/dcoeffs
    
    # d(norm2)/dcoeffs = d/sqrt(sum W f^2) / dcoeffs
    # norm2 = (sum W f^2)^{1/2}
    # d(norm2)/dcoeffs = (1/(2*norm2)) * 2 * W * f * B_flat^T = (W * f) @ B_flat / norm2
    d_norm2_dcoeffs = (np.sum(_W_flat * f * _B_flat, axis=1)) / norm2
    # Wait, let me redo this more carefully.
    # norm2_sq = sum_j W_j f_j^2
    # d(norm2_sq)/dcoeffs = 2 * B_flat^T (W * f)
    # d(norm2)/dcoeffs = (1/(2*norm2)) * 2 * B_flat^T (W * f) = B_flat^T (W * f) / norm2
    
    d_norm2_dcoeffs = _B_flat.T @ (_W_flat * f) / norm2
    
    # norm4 = (sum W f^4)^{1/4}
    # d(norm4)/dcoeffs = (1/4) (sum W f^4)^{-3/4} * 4 * sum W f^3 * df/dcoeffs
    #                  = (sum W f^4)^{-3/4} * B_flat^T (W * f^3)
    #                  = norm4_4^{-3/4} * B_flat^T (W * f^3)
    
    d_norm4_dcoeffs = _B_flat.T @ (_W_flat * f * f * f) / (norm4_4 ** 0.75)
    
    grad = (1.0 / norm2) * d_norm4_dcoeffs - (R / norm2) * d_norm2_dcoeffs
    
    return R, grad

def _is_valid(coeffs):
    """Check if coefficients are valid per the task contract."""
    if len(coeffs) != 21:
        return False
    if np.any(np.isnan(coeffs)) or np.any(np.isinf(coeffs)):
        return False
    max_abs = np.max(np.abs(coeffs))
    if max_abs < 1e-3 or max_abs > 1e5:
        return False
    return True

def rudin_problem10() -> np.ndarray:
    """
    Find real coefficients c_m for m = -10..10 minimizing R = ||f||_4 / ||f||_2
    for f = sum_m c_m Y_m with real orthonormal degree-10 spherical harmonics.
    
    Returns:
        coeffs: np.ndarray of shape (21,) with real coefficients c_m.
    """
    start_time = time.time()
    deadline = start_time + 590.0  # Safety margin before 600s limit
    
    # The objective is scale-invariant, so we can fix a constraint like sum(c^2) = 1
    # or ||f||_2 = 1 to avoid the trivial zero solution.
    # Let's use constraint: ||f||_2^2 = 1, i.e., coeffs^T G coeffs = 1
    # where G = B_flat^T diag(W) B_flat
    
    G = _B_flat.T @ (_W_flat[:, None] * _B_flat)  # (21, 21)
    
    # Since the problem is scale-invariant, we can minimize R subject to coeffs^T G coeffs = 1
    # But R is already scale-invariant, so we can just minimize R directly.
    # The optimizer might drift to zero or infinity, so let's add a constraint.
    
    # Actually, since R is scale-invariant, the gradient-based optimizer should work
    # if we start from a reasonable point. Let's use SLSQP with a constraint.
    
    def constraint_func(coeffs):
        return coeffs @ G @ coeffs - 1.0
    
    def constraint_grad(coeffs):
        return 2.0 * G @ coeffs
    
    constraints = [{'type': 'eq', 'fun': constraint_func, 'jac': constraint_grad}]
    
    # Try multiple starting points
    best_coeffs = np.ones(21, dtype=np.float64)
    best_R = _objective_and_gradient(best_coeffs)[0]
    
    # Generate diverse starting points
    np.random.seed(42)
    start_points = [np.ones(21, dtype=np.float64)]
    
    # Add some structured starting points
    for trial in range(20):
        sp = np.random.randn(21)
        # Normalize to have ||f||_2 = 1
        norm2_sq = sp @ G @ sp
        if norm2_sq > 0:
            sp = sp / np.sqrt(norm2_sq)
        start_points.append(sp)
    
    # Also try some specific patterns
    # Concentrate on certain m values
    for m_target in range(-10, 11):
        sp = np.zeros(21)
        sp[m_target + 10] = 1.0
        norm2_sq = sp @ G @ sp
        if norm2_sq > 0:
            sp = sp / np.sqrt(norm2_sq)
        start_points.append(sp)
    
    for i, sp in enumerate(start_points):
        if time.time() > deadline - 30:
            break
        
        try:
            result = minimize(
                _objective_and_gradient,
                sp,
                jac=True,
                method='SLSQP',
                constraints=constraints,
                options={'maxiter': 500, 'ftol': 1e-12, 'disp': False}
            )
            
            if result.success and _is_valid(result.x):
                R_val = _objective_and_gradient(result.x)[0]
                if R_val < best_R:
                    best_R = R_val
                    best_coeffs = result.x.copy()
        except Exception:
            continue
    
    # Final polish with tighter tolerance
    if time.time() < deadline - 10:
        try:
            result = minimize(
                _objective_and_gradient,
                best_coeffs,
                jac=True,
                method='SLSQP',
                constraints=constraints,
                options={'maxiter': 1000, 'ftol': 1e-15, 'disp': False}
            )
            if result.success and _is_valid(result.x):
                R_val = _objective_and_gradient(result.x)[0]
                if R_val < best_R:
                    best_R = R_val
                    best_coeffs = result.x.copy()
        except Exception:
            pass
    
    # Ensure validity
    if not _is_valid(best_coeffs):
        best_coeffs = np.ones(21, dtype=np.float64)
    
    return best_coeffs


# EVOLVE-BLOCK-END
