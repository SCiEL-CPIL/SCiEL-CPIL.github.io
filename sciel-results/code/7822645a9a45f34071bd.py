# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _gauss_legendre(n):
    """Return (x, w) for n-point Gauss-Legendre on [-1, 1]."""
    x, w = np.polynomial.legendre.leggauss(n)
    return x, w

def _spherical_harmonics_real(x, phi, l=10):
    """
    Compute real orthonormal spherical harmonics Y_m for m = -l..l.
    x = cos(theta) array of shape (N,), phi array of shape (M,).
    Returns Y of shape (N, M, 2*l+1) where Y[i,j,m+l] = Y_m(x[i], phi[j]).
    """
    N = len(x)
    M = len(phi)
    Y = np.zeros((N, M, 2*l+1))
    
    # Precompute associated Legendre polynomials with Condon-Shortley phase
    # P_l^m(x) for m = 0..l
    # Using recurrence: P_l^l(x) = (-1)^l (2l-1)!! (1-x^2)^(l/2)
    # P_{l-1}^l(x) = 0
    # (l-m) P_l^m = (2l-1) x P_{l-1}^m - (l+m-1) P_{l-2}^m
    
    # We'll compute P_l^m for each m using the recurrence in l
    # For fixed m, P_l^m for l >= m
    # P_m^m(x) = (-1)^m (2m-1)!! (1-x^2)^(m/2)
    # P_{m+1}^m(x) = (2m+1) x P_m^m(x)
    # (l-m) P_l^m = (2l-1) x P_{l-1}^m - (l+m-1) P_{l-2}^m
    
    # (2m-1)!! = product of odd numbers up to (2m-1)
    # (2m-1)!! = (2m)! / (2^m m!)
    
    def double_factorial_odd(n):
        """(2n-1)!! = 1*3*5*...*(2n-1)"""
        if n == 0:
            return 1
        result = 1
        for k in range(1, n+1):
            result *= (2*k - 1)
        return result
    
    # Compute P_l^m(x) for all m = 0..l
    # P[m, i] = P_l^m(x[i])
    P = np.zeros((l+1, N))
    
    for m in range(l+1):
        # P_m^m(x) = (-1)^m (2m-1)!! (1-x^2)^(m/2)
        df = double_factorial_odd(m)
        sign = (-1)**m
        P[m, :] = sign * df * np.power(np.maximum(1 - x**2, 0), m/2.0)
        
        if m < l:
            # P_{m+1}^m(x) = (2m+1) x P_m^m(x)
            P[m+1, :] = (2*m + 1) * x * P[m, :]
            
            # P_l^m for l > m+1
            for ll in range(m+2, l+1):
                P[ll, :] = ((2*ll - 1) * x * P[ll-1, :] - (ll + m - 1) * P[ll-2, :]) / (ll - m)
    
    # Now P[m, i] = P_l^m(x[i]) for m = 0..l
    # Normalization: N_m = sqrt(21/(4*pi) * (l-m)!/(l+m)!)
    # Y_0 = N_0 * P_l^0
    # Y_m = sqrt(2) * N_m * P_l^m * cos(m*phi) for m > 0
    # Y_{-m} = sqrt(2) * N_m * P_l^m * sin(m*phi) for m > 0
    
    norm = 21.0 / (4.0 * np.pi)
    for m in range(l+1):
        N_m = np.sqrt(norm * np.math.factorial(l - m) / np.math.factorial(l + m))
        
        if m == 0:
            Y[:, :, m] = N_m * P[0, :, np.newaxis]  # shape (N, 1) broadcast to (N, M)
        else:
            cos_m = np.cos(m * phi)  # shape (M,)
            sin_m = np.sin(m * phi)  # shape (M,)
            Pm = P[m, :, np.newaxis]  # shape (N, 1)
            Y[:, :, l + m] = np.sqrt(2.0) * N_m * Pm * cos_m[np.newaxis, :]  # (N, M)
            Y[:, :, l - m] = np.sqrt(2.0) * N_m * Pm * sin_m[np.newaxis, :]  # (N, M)
    
    return Y

def _compute_objective(c, Y, weights):
    """
    Compute R^4 = ||f||_4^4 / ||f||_2^4 where f = sum c_m Y_m.
    Y has shape (N, M, 21), weights has shape (N, M) (product of GL weights and phi weights).
    """
    # f has shape (N, M)
    f = np.einsum('i,ijm->ij', c, Y)  # (N, M)
    
    # ||f||_2^2 = sum f^2 * w
    norm2_sq = np.sum(f**2 * weights)
    if norm2_sq <= 0:
        return 1e10
    
    # ||f||_4^4 = sum f^4 * w
    norm4_4 = np.sum(f**4 * weights)
    
    R4 = norm4_4 / norm2_sq**2
    return R4

def _logsumexp_safe(x, axis=None):
    """Numerically stable log-sum-exp."""
    x_max = np.max(x, axis=axis, keepdims=True)
    # Handle case where x_max is -inf
    x_max = np.where(np.isfinite(x_max), x_max, 0)
    out = x_max + np.log(np.sum(np.exp(x - x_max), axis=axis, keepdims=True))
    if axis is not None:
        out = np.squeeze(out, axis=axis)
    return out

def _objective_smooth(c, Y, weights, alpha=1.0):
    """
    Smooth surrogate: (1/alpha) * log(sum exp(alpha * f^4 * w)) / norm2_sq^2
    This approximates max(f^4 * w) / norm2_sq^2 for large alpha, but is smoother.
    Actually, let's use: log(sum exp(alpha * f^4)) / norm2_sq^2
    But we need to be careful. Let me use a different approach.
    
    Actually, the objective is R4 = sum(f^4 * w) / (sum(f^2 * w))^2
    This is already smooth. Let's just use it directly but with SLSQP.
    """
    f = np.einsum('i,ijm->ij', c, Y)
    norm2_sq = np.sum(f**2 * weights)
    if norm2_sq <= 1e-15:
        return 1e10
    norm4_4 = np.sum(f**4 * weights)
    return norm4_4 / (norm2_sq**2)

def _objective_grad(c, Y, weights):
    """Compute objective and gradient for R4 = ||f||_4^4 / ||f||_2^4."""
    f = np.einsum('i,ijm->ij', c, Y)  # (N, M)
    f2 = f**2
    f4 = f2**2
    
    norm2_sq = np.sum(f2 * weights)
    if norm2_sq <= 1e-15:
        return 1e10, np.zeros_like(c)
    
    norm4_4 = np.sum(f4 * weights)
    
    R4 = norm4_4 / (norm2_sq**2)
    
    # Gradient of R4 w.r.t. c_m
    # dR4/dc_m = (1/norm2_sq^2) * d(norm4_4)/dc_m - 2*norm4_4/norm2_sq^3 * d(norm2_sq)/dc_m
    # d(norm4_4)/dc_m = sum 4*f^3 * Y_m * w = 4 * sum(f^3 * Y_m * w)
    # d(norm2_sq)/dc_m = sum 2*f * Y_m * w = 2 * sum(f * Y_m * w)
    
    # f^3 * w has shape (N, M)
    f3w = f**3 * weights  # (N, M)
    fw = f * weights  # (N, M)
    
    # grad_norm4_4[m] = 4 * sum over i,j of f3w[i,j] * Y[i,j,m]
    grad_n4 = 4.0 * np.einsum('ij,ijm->m', f3w, Y)
    grad_n2 = 2.0 * np.einsum('ij,ijm->m', fw, Y)
    
    grad_R4 = grad_n4 / (norm2_sq**2) - 2.0 * norm4_4 * grad_n2 / (norm2_sq**3)
    
    return R4, grad_R4

def rudin_problem10() -> np.ndarray:
    """
    Find real coefficients c_m for m = -10..10 of a degree-10 spherical harmonic
    that minimizes R = ||f||_4 / ||f||_2 over the unit sphere.
    """
    start_time = time.time()
    deadline = start_time + 600.0
    safety_margin = 30.0  # seconds
    
    l = 10
    n_theta = 70
    n_phi = 130
    
    # Set up quadrature
    x, w_theta = _gauss_legendre(n_theta)
    phi = np.linspace(0, 2*np.pi, n_phi, endpoint=False)
    w_phi = np.full(n_phi, 2*np.pi/n_phi)
    
    # Weights: w[i,j] = w_theta[i] * w_phi[j] / 4*pi  (normalized so total mass = 1)
    # Actually, the problem says norms under surface measure with total mass 4*pi.
    # dOmega = dphi * d(cos theta) = dphi * dx
    # So ||f||_p^p = integral |f|^p dOmega = sum_{i,j} |f(i,j)|^p * w_theta[i] * w_phi[j]
    # The ratio R = ||f||_4/||f||_2 is scale-invariant, so normalization doesn't matter.
    weights = np.outer(w_theta, w_phi)  # shape (n_theta, n_phi)
    
    # Compute spherical harmonics
    Y = _spherical_harmonics_real(x, phi, l=l)  # shape (n_theta, n_phi, 21)
    
    # Define constrained objective: minimize R4 subject to ||c||_2^2 = 1
    def obj_constrained(c):
        return _objective_smooth(c, Y, weights)
    
    def obj_and_grad_constrained(c):
        val, grad = _objective_grad(c, Y, weights)
        return val, grad
    
    # SLSQP constraint: ||c||_2^2 - 1 = 0
    constraint = {'type': 'eq', 'fun': lambda c: np.dot(c, c) - 1.0,
                  'jac': lambda c: 2.0 * c}
    
    best_c = None
    best_R4 = np.inf
    
    # Generate multiple starting points
    starts = []
    
    # Pure modes: e_m for each m
    for m in range(2*l + 1):
        e = np.zeros(2*l + 1)
        e[m] = 1.0
        starts.append(e)
    
    # Random starts
    rng = np.random.default_rng(42)
    n_random = 10
    for _ in range(n_random):
        c = rng.standard_normal(2*l + 1)
        c /= np.linalg.norm(c)
        starts.append(c)
    
    # Also try the all-ones start (normalized)
    c_ones = np.ones(2*l + 1)
    c_ones /= np.linalg.norm(c_ones)
    starts.append(c_ones)
    
    # Optimize from each start
    for idx, c0 in enumerate(starts):
        if time.time() > deadline - safety_margin:
            break
        
        try:
            result = minimize(
                obj_and_grad_constrained,
                c0,
                method='SLSQP',
                constraints=[constraint],
                options={'maxiter': 200, 'ftol': 1e-12, 'disp': False}
            )
            
            if result.success or result.fun < best_R4:
                c_opt = result.x
                # Ensure it's on the sphere
                c_opt = c_opt / np.linalg.norm(c_opt)
                
                R4_val = _objective_smooth(c_opt, Y, weights)
                
                if R4_val < best_R4:
                    best_R4 = R4_val
                    best_c = c_opt.copy()
                    
        except Exception as e:
            continue
    
    # If no optimization succeeded, use the best pure mode
    if best_c is None:
        # Try pure modes and pick best
        for m in range(2*l + 1):
            e = np.zeros(2*l + 1)
            e[m] = 1.0
            R4_val = _objective_smooth(e, Y, weights)
            if R4_val < best_R4:
                best_R4 = R4_val
                best_c = e.copy()
    
    # Final polish with tighter tolerance
    if best_c is not None and time.time() < deadline - safety_margin:
        try:
            result = minimize(
                obj_and_grad_constrained,
                best_c,
                method='SLSQP',
                constraints=[constraint],
                options={'maxiter': 500, 'ftol': 1e-15, 'disp': False}
            )
            if result.fun < best_R4:
                best_c = result.x / np.linalg.norm(result.x)
                best_R4 = result.fun
        except:
            pass
    
    # Validate
    if best_c is None:
        best_c = np.ones(2*l + 1)
    
    # Ensure valid range
    max_abs = np.max(np.abs(best_c))
    if max_abs < 1e-3:
        best_c = best_c / max_abs * 1e-2
    elif max_abs > 1e5:
        best_c = best_c / max_abs * 1e4
    
    # Check for NaN/inf
    if not np.all(np.isfinite(best_c)):
        best_c = np.ones(2*l + 1)
    
    return best_c

# EVOLVE-BLOCK-END
