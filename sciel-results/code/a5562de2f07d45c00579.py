# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import lpmv, gammaln
from scipy.optimize import minimize, trust_constr, LinearConstraint


def _build_quadrature():
    """Build the 70-point Gauss-Legendre (cos theta) x 130-point phi grid."""
    x_gl, w_gl = np.polynomial.legendre.leggauss(70)
    n_phi = 130
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    dphi = 2 * np.pi / n_phi
    weights = np.outer(w_gl, np.full(n_phi, dphi))
    cos_theta = x_gl
    return cos_theta, phi, weights


def _spherical_harmonics_real(cos_theta, phi, k=10):
    """
    Compute real orthonormal spherical harmonics of degree k.
    Returns array of shape (2*k+1, n_theta, n_phi) where index i corresponds to m = i - k.
    """
    n_theta = len(cos_theta)
    n_phi = len(phi)
    Y = np.zeros((2 * k + 1, n_theta, n_phi))
    
    for m_abs in range(k + 1):
        log_N = 0.5 * (np.log(2 * k + 1) - np.log(4 * np.pi) + 
                       gammaln(k - m_abs + 1) - gammaln(k + m_abs + 1))
        N_m = np.exp(log_N)
        P_m = lpmv(m_abs, k, cos_theta)
        
        if m_abs == 0:
            Y[k] = N_m * P_m[:, np.newaxis] * np.ones((1, n_phi))
        else:
            Y[k + m_abs] = np.sqrt(2) * N_m * P_m[:, np.newaxis] * np.cos(m_abs * phi)[np.newaxis, :]
            Y[k - m_abs] = np.sqrt(2) * N_m * P_m[:, np.newaxis] * np.sin(m_abs * phi)[np.newaxis, :]
    
    return Y


def _compute_ratio_and_grad(coeffs, Y, weights):
    """
    Compute R = ||f||_4 / ||f||_2 and its gradient with respect to coeffs.
    """
    f = np.tensordot(coeffs, Y, axes=([0], [0]))
    
    f2 = f * f
    f4 = f2 * f2
    f3 = f * f2
    
    w_f2 = weights * f2
    w_f4 = weights * f4
    w_f3 = weights * f3
    w_f = weights * f
    
    norm2_sq = np.sum(w_f2)
    norm4_sq = np.sum(w_f4)
    
    if norm2_sq <= 1e-15:
        return np.inf, np.zeros_like(coeffs)
    
    norm2 = np.sqrt(norm2_sq)
    norm4 = np.sqrt(np.sqrt(norm4_sq))
    
    R = norm4 / norm2
    
    if norm4_sq <= 1e-30:
        return R, np.zeros_like(coeffs)
    
    # d(norm4)/dc = (1/(4*norm4^3)) * d(norm4_sq)/dc = (1/(4*norm4^3)) * 4 * sum(w * f^3 * Y_m)
    d_norm4_sq_dc = 4.0 * np.einsum('ij,mij->m', w_f3, Y)
    d_norm2_sq_dc = 2.0 * np.einsum('ij,mij->m', w_f, Y)
    
    d_norm4_dc = d_norm4_sq_dc / (4.0 * norm4**3)
    d_norm2_dc = d_norm2_sq_dc / (2.0 * norm2)
    
    # R = norm4 / norm2
    # dR/dc = d_norm4_dc / norm2 - norm4 * d_norm2_dc / norm2^2
    grad = d_norm4_dc / norm2 - norm4 * d_norm2_dc / (norm2 * norm2)
    
    return R, grad


def _try_symmetric_patterns(Y, weights):
    """Try various symmetric coefficient patterns."""
    best_coeffs = None
    best_ratio = np.inf
    
    patterns = []
    
    # All ones normalized
    c = np.ones(21) / np.sqrt(21)
    patterns.append(c)
    
    # Only m=0
    c = np.zeros(21)
    c[10] = 1.0
    patterns.append(c)
    
    # m=0 and m=±10
    c = np.zeros(21)
    c[10] = 1.0
    c[0] = 1.0
    c[20] = 1.0
    c /= np.linalg.norm(c)
    patterns.append(c)
    
    # Even m modes: 0, 2, 4, 6, 8, 10
    c = np.zeros(21)
    for m in [0, 2, 4, 6, 8, 10]:
        c[10 + m] = 1.0
        if m > 0:
            c[10 - m] = 1.0
    c /= np.linalg.norm(c)
    patterns.append(c)
    
    # Odd m modes: 1, 3, 5, 7, 9
    c = np.zeros(21)
    for m in [1, 3, 5, 7, 9]:
        c[10 + m] = 1.0
        c[10 - m] = 1.0
    c /= np.linalg.norm(c)
    patterns.append(c)
    
    # Cosine-weighted: c_m = cos(m * pi/20) for m=0..10
    c = np.zeros(21)
    for m in range(11):
        c[10 + m] = np.cos(m * np.pi / 20.0)
        if m > 0:
            c[10 - m] = np.cos(m * np.pi / 20.0)
    c /= np.linalg.norm(c)
    patterns.append(c)
    
    # Sine-weighted
    c = np.zeros(21)
    for m in range(1, 11):
        c[10 + m] = np.sin(m * np.pi / 21.0)
        c[10 - m] = np.sin(m * np.pi / 21.0)
    c /= np.linalg.norm(c)
    patterns.append(c)
    
    # High-m concentration
    c = np.zeros(21)
    c[0] = 1.0
    c[20] = 1.0
    c /= np.linalg.norm(c)
    patterns.append(c)
    
    # Low-m concentration
    c = np.zeros(21)
    c[10] = 1.0
    c[9] = 1.0
    c[11] = 1.0
    c /= np.linalg.norm(c)
    patterns.append(c)
    
    # Alternating signs
    c = np.zeros(21)
    for m in range(21):
        c[m] = 1.0 if m % 2 == 0 else -1.0
    c /= np.linalg.norm(c)
    patterns.append(c)
    
    # Gaussian-like distribution
    c = np.zeros(21)
    for m in range(21):
        idx = m - 10
        c[m] = np.exp(-idx**2 / 20.0)
    c /= np.linalg.norm(c)
    patterns.append(c)
    
    # Weighted by m^alpha
    for alpha in [0.5, 1.0, 1.5, 2.0]:
        c = np.zeros(21)
        for m in range(11):
            c[10 + m] = m**alpha if m > 0 else 1.0
            if m > 0:
                c[10 - m] = m**alpha
        c /= np.linalg.norm(c)
        patterns.append(c)
    
    # Inverse weighted
    for alpha in [0.5, 1.0, 1.5]:
        c = np.zeros(21)
        c[10] = 1.0
        for m in range(1, 11):
            c[10 + m] = 1.0 / (m**alpha)
            c[10 - m] = 1.0 / (m**alpha)
        c /= np.linalg.norm(c)
        patterns.append(c)
    
    # Pure high m
    for m_val in [8, 9, 10]:
        c = np.zeros(21)
        c[10 + m_val] = 1.0
        c[10 - m_val] = 1.0
        c /= np.linalg.norm(c)
        patterns.append(c)
    
    # Phase-shifted m=10
    for phase in np.linspace(0, np.pi, 7):
        c = np.zeros(21)
        c[20] = np.cos(phase)
        c[0] = np.sin(phase)
        c /= np.linalg.norm(c)
        patterns.append(c)
    
    for c in patterns:
        r, _ = _compute_ratio_and_grad(c, Y, weights)
        if r < best_ratio:
            best_ratio = r
            best_coeffs = c.copy()
    
    return best_coeffs, best_ratio


def _simulated_annealing(c_init, Y, weights, n_steps=3000, t_start=0.3, t_end=0.001, rng=None):
    """Simulated annealing to find better coefficients."""
    if rng is None:
        rng = np.random.default_rng()
    
    c = c_init.copy()
    c /= np.linalg.norm(c)
    current_ratio, _ = _compute_ratio_and_grad(c, Y, weights)
    best_c = c.copy()
    best_ratio = current_ratio
    
    for step in range(n_steps):
        t = t_start * (t_end / t_start) ** (step / n_steps)
        
        delta = rng.standard_normal(21) * 0.08
        c_new = c + delta
        norm = np.linalg.norm(c_new)
        if norm < 1e-10:
            continue
        c_new /= norm
        
        new_ratio, _ = _compute_ratio_and_grad(c_new, Y, weights)
        
        if new_ratio < current_ratio or rng.random() < np.exp((current_ratio - new_ratio) / t):
            c = c_new
            current_ratio = new_ratio
            if new_ratio < best_ratio:
                best_ratio = new_ratio
                best_c = c.copy()
    
    return best_c, best_ratio


def _polish_local(c_init, Y, weights, deadline, safety_margin):
    """High-precision polish using trust-constr with tighter tolerances."""
    def objective(c):
        c_norm = np.linalg.norm(c)
        if c_norm < 1e-10:
            return 1e10
        c_normalized = c / c_norm
        R, _ = _compute_ratio_and_grad(c_normalized, Y, weights)
        return R
    
    def gradient(c):
        c_norm = np.linalg.norm(c)
        if c_norm < 1e-10:
            return np.zeros_like(c)
        c_normalized = c / c_norm
        R, grad = _compute_ratio_and_grad(c_normalized, Y, weights)
        # Project gradient onto tangent space
        dot = np.dot(grad, c_normalized)
        grad_proj = grad - c_normalized * dot
        return grad_proj / c_norm
    
    # Constraint: ||c|| = 1 (use inequality ||c||^2 <= 1 and >= 1)
    # Actually, let's just use the sphere constraint via the projection
    # trust_constr can handle this with a non-linear constraint
    
    def constraint(c):
        return np.linalg.norm(c) - 1.0
    
    def constraint_jac(c):
        c_norm = np.linalg.norm(c)
        if c_norm < 1e-10:
            return np.zeros((1, 21))
        return (c / c_norm).reshape(1, -1)
    
    try:
        result = trust_constr(
            objective,
            c_init,
            constraints=[{'type': 'eq', 'fun': constraint, 'jac': constraint_jac}],
            jac=gradient,
            method='trf',
            options={'maxiter': 5000, 'gtol': 1e-15, 'xtol': 1e-15}
        )
        
        if result.success and result.fun < objective(c_init):
            c_opt = result.x / np.linalg.norm(result.x)
            r_opt, _ = _compute_ratio_and_grad(c_opt, Y, weights)
            if r_opt < objective(c_init):
                return c_opt, r_opt
    except Exception:
        pass
    
    # Fallback to L-BFGS-B
    def objective_and_grad(c):
        c_norm = np.linalg.norm(c)
        if c_norm < 1e-10:
            return 1e10, np.zeros_like(c)
        c_normalized = c / c_norm
        R, grad = _compute_ratio_and_grad(c_normalized, Y, weights)
        dot = np.dot(grad, c_normalized)
        grad_proj = grad - c_normalized * dot
        return R, grad_proj / c_norm
    
    result = minimize(
        objective_and_grad,
        c_init,
        method='L-BFGS-B',
        jac=True,
        options={'maxiter': 10000, 'ftol': 1e-18, 'gtol': 1e-16, 'maxls': 300}
    )
    
    if result.fun < objective(c_init):
        c_opt = result.x / np.linalg.norm(result.x)
        r_opt, _ = _compute_ratio_and_grad(c_opt, Y, weights)
        if r_opt < objective(c_init):
            return c_opt, r_opt
    
    return c_init, objective(c_init)


def rudin_problem10() -> np.ndarray:
    """
    Find coefficients minimizing R = ||f||_4 / ||f||_2 for degree-10 spherical harmonic.
    Returns 21 real coefficients ordered m = -10..10.
    """
    deadline = time.time() + 600.0
    safety_margin = 20.0
    
    # Build quadrature grid
    cos_theta, phi, weights = _build_quadrature()
    
    # Precompute spherical harmonics
    Y = _spherical_harmonics_real(cos_theta, phi, k=10)
    
    rng = np.random.default_rng(42)
    
    # Try symmetric patterns first
    best_coeffs, best_ratio = _try_symmetric_patterns(Y, weights)
    
    # Add random initializations
    initial_guesses = [best_coeffs.copy()]
    for i in range(50):
        c_rand = rng.standard_normal(21)
        c_rand /= np.linalg.norm(c_rand)
        initial_guesses.append(c_rand)
    
    # Evaluate all initial guesses
    for c_init in initial_guesses:
        r, _ = _compute_ratio_and_grad(c_init, Y, weights)
        if r < best_ratio:
            best_ratio = r
            best_coeffs = c_init.copy()
    
    # Run simulated annealing from best
    if time.time() < deadline - safety_margin:
        sa_coeffs, sa_ratio = _simulated_annealing(
            best_coeffs, Y, weights,
            n_steps=3000, t_start=0.3, t_end=0.001, rng=rng
        )
        if sa_ratio < best_ratio:
            best_ratio = sa_ratio
            best_coeffs = sa_coeffs.copy()
    
    # Run local optimization from best
    if time.time() < deadline - safety_margin:
        polished, polished_ratio = _polish_local(best_coeffs, Y, weights, deadline, safety_margin)
        if polished_ratio < best_ratio:
            best_ratio = polished_ratio
            best_coeffs = polished.copy()
    
    # Try more random restarts with local optimization
    n_restarts = 40
    for i in range(n_restarts):
        if time.time() > deadline - safety_margin:
            break
            
        c_rand = rng.standard_normal(21)
        c_rand /= np.linalg.norm(c_rand)
        
        # Quick local optimization
        def objective_and_grad(c):
            c_norm = np.linalg.norm(c)
            if c_norm < 1e-10:
                return 1e10, np.zeros_like(c)
            c_normalized = c / c_norm
            R, grad = _compute_ratio_and_grad(c_normalized, Y, weights)
            dot = np.dot(grad, c_normalized)
            grad_proj = grad - c_normalized * dot
            return R, grad_proj / c_norm
        
        max_iter = 2000 if i < 10 else 1000
        result = minimize(
            objective_and_grad,
            c_rand,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': max_iter, 'ftol': 1e-15, 'gtol': 1e-13, 'maxls': 100}
        )
        
        if result.fun < best_ratio:
            c_opt = result.x / np.linalg.norm(result.x)
            r_opt, _ = _compute_ratio_and_grad(c_opt, Y, weights)
            if r_opt < best_ratio:
                best_ratio = r_opt
                best_coeffs = c_opt.copy()
    
    # Final high-precision polish
    if time.time() < deadline - safety_margin:
        polished, polished_ratio = _polish_local(best_coeffs, Y, weights, deadline, safety_margin)
        if polished_ratio < best_ratio:
            best_ratio = polished_ratio
            best_coeffs = polished.copy()
    
    # Ensure output is valid
    best_coeffs = best_coeffs.astype(np.float64)
    if np.any(np.isnan(best_coeffs)) or np.any(np.isinf(best_coeffs)):
        best_coeffs = np.ones(21, dtype=np.float64) / np.sqrt(21)
        
    max_abs = np.max(np.abs(best_coeffs))
    if max_abs < 1e-3 or max_abs > 1e5:
        target = 1.0
        best_coeffs = best_coeffs * (target / max_abs)
        
    return best_coeffs


# EVOLVE-BLOCK-END
