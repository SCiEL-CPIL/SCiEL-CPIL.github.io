import numpy as np
import time
from scipy.special import eval_legendre, gammaln
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

def _precompute():
    """Precompute quadrature nodes, weights, and spherical harmonic basis."""
    k = 10
    n_theta = 70
    x, w = np.polynomial.legendre.leggauss(n_theta)
    
    n_phi = 130
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    dphi = 2 * np.pi / n_phi
    
    # Build basis matrix B[m_idx, i, j]
    # m_idx = 0..20 corresponds to m = -10..10
    P = np.zeros((11, n_theta))
    for m in range(11):
        P[m, :] = eval_legendre(10, m, x)
        
    N = np.zeros(11)
    for m in range(11):
        log_N = 0.5 * (np.log(21.0 / (4 * np.pi)) + gammaln(11 - m) - gammaln(11 + m))
        N[m] = np.exp(log_N)
        
    B = np.zeros((21, n_theta, n_phi))
    for m_idx in range(21):
        m = m_idx - 10
        if m == 0:
            for j in range(n_phi):
                B[m_idx, :, j] = N[0] * P[0, :]
        elif m > 0:
            for j in range(n_phi):
                B[m_idx, :, j] = np.sqrt(2) * N[m] * P[m, :] * np.cos(m * phi[j])
        else:
            am = abs(m)
            for j in range(n_phi):
                B[m_idx, :, j] = np.sqrt(2) * N[am] * P[am, :] * np.sin(am * phi[j])
                
    # Quadrature weights
    W = np.outer(w, np.ones(n_phi)) * dphi  # shape (n_theta, n_phi)
    
    # Flatten B and W for matrix operations
    B_flat = B.reshape(21, n_theta * n_phi)
    W_flat = W.reshape(n_theta * n_phi)
    
    return B_flat, W_flat

def _objective_penalty(c, B_flat, W_flat, penalty_coeff):
    """
    Objective: minimize ||f||_4^4 + penalty * (||c||_2^2 - 1)^2
    """
    f = B_flat.T @ c
    f4 = f**4
    base = np.dot(W_flat, f4)
    norm_sq = np.dot(c, c)
    penalty = penalty_coeff * (norm_sq - 1.0)**2
    return base + penalty

def _grad_penalty(c, B_flat, W_flat, penalty_coeff):
    """
    Gradient of the penalized objective.
    """
    f = B_flat.T @ c
    base_grad = 4.0 * B_flat @ (W_flat * f**3)
    norm_sq = np.dot(c, c)
    p_grad = 4.0 * penalty_coeff * (norm_sq - 1.0) * c
    return base_grad + p_grad

def _objective_constraint(c, B_flat, W_flat):
    """
    Objective for SLSQP: minimize ||f||_4^4
    """
    f = B_flat.T @ c
    f4 = f**4
    return np.dot(W_flat, f4)

def _grad_constraint(c, B_flat, W_flat):
    """
    Gradient for SLSQP.
    """
    f = B_flat.T @ c
    return 4.0 * B_flat @ (W_flat * f**3)

def _constraint_func(c):
    """
    Constraint for SLSQP: ||c||_2^2 - 1 = 0
    """
    return np.dot(c, c) - 1.0

def _jac_constraint(c):
    """
    Jacobian of the constraint.
    """
    return 2.0 * c

def _compute_ratio(c, B_flat, W_flat):
    """Compute ||f||_4 / ||f||_2 for coefficient vector c."""
    f = B_flat.T @ c
    f2 = f * f
    norm2_sq = np.dot(W_flat, f2)
    if norm2_sq <= 0:
        return np.inf
    f4 = f2 * f2
    norm4_4 = np.dot(W_flat, f4)
    norm2 = np.sqrt(norm2_sq)
    norm4 = np.sqrt(np.sqrt(norm4_4))
    return norm4 / norm2

def rudin_problem10() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 600  # Safety margin
    
    B_flat, W_flat = _precompute()
    dim = 21
    
    best_coeffs = None
    best_ratio = np.inf
    candidates = []
    
    # Phase 1: Global Search with L-BFGS-B (Penalty Method)
    # We use a moderate penalty coefficient.
    penalty_coeff = 100.0
    
    np.random.seed(42)
    
    # Generate initializations
    initializations = []
    
    # 1. Random unit vectors
    for _ in range(100):
        c = np.random.randn(dim)
        c /= np.linalg.norm(c)
        initializations.append(c)
        
    # 2. Structured initializations
    # All ones
    c = np.ones(dim) / np.sqrt(dim)
    initializations.append(c)
    
    # Only m=0
    c0 = np.zeros(dim)
    c0[10] = 1.0
    initializations.append(c0)
    
    # Alternating signs
    c_alt = np.ones(dim)
    c_alt[::2] = -1.0
    initializations.append(c_alt / np.linalg.norm(c_alt))
    
    # High m modes
    for m_val in [5, 8, 10]:
        for sign in [1, -1]:
            c = np.zeros(dim)
            c[m_val + 10] = sign
            c[-m_val + 10] = sign
            initializations.append(c / np.linalg.norm(c))
            
    # Cosine/Sine patterns
    for phase in [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]:
        c_ph = np.array([np.cos((m_idx - 10) * phase) for m_idx in range(dim)])
        if np.linalg.norm(c_ph) > 1e-10:
            initializations.append(c_ph / np.linalg.norm(c_ph))
        c_ph2 = np.array([np.sin((m_idx - 10) * phase) for m_idx in range(dim)])
        if np.linalg.norm(c_ph2) > 1e-10:
            initializations.append(c_ph2 / np.linalg.norm(c_ph2))

    # Run L-BFGS-B from each initialization
    for i, c_init in enumerate(initializations):
        if time.time() > deadline - 100:  # Leave time for SLSQP polish
            break
            
        try:
            result = minimize(
                _objective_penalty,
                c_init,
                args=(B_flat, W_flat, penalty_coeff),
                jac=_grad_penalty,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-6, 'maxls': 50}
            )
            
            c_opt = result.x
            # Normalize to ensure constraint is satisfied approximately
            c_opt = c_opt / np.linalg.norm(c_opt)
            
            ratio = _compute_ratio(c_opt, B_flat, W_flat)
            
            if ratio < best_ratio:
                best_ratio = ratio
                best_coeffs = c_opt.copy()
                
            # Store top candidates
            if len(candidates) < 10 or ratio < candidates[-1][0]:
                candidates.append((ratio, c_opt.copy()))
                candidates.sort(key=lambda x: x[0])
                if len(candidates) > 10:
                    candidates.pop()
                    
        except Exception as e:
            pass

    # Phase 2: Local Polish with SLSQP (Hard Constraint)
    # SLSQP is more accurate for the final result but slower.
    # We run it on the best candidates from Phase 1.
    
    if time.time() < deadline - 20:
        for ratio_est, c_init in candidates:
            if time.time() > deadline - 5:
                break
                
            try:
                constraints = [{'type': 'eq', 'fun': _constraint_func, 'jac': _jac_constraint}]
                result = minimize(
                    _objective_constraint,
                    c_init,
                    args=(B_flat, W_flat),
                    jac=_grad_constraint,
                    method='SLSQP',
                    constraints=constraints,
                    options={'maxiter': 500, 'ftol': 1e-12, 'disp': False}
                )
                
                c_opt = result.x
                c_opt = c_opt / np.linalg.norm(c_opt)
                
                ratio = _compute_ratio(c_opt, B_flat, W_flat)
                
                if ratio < best_ratio:
                    best_ratio = ratio
                    best_coeffs = c_opt.copy()
                    
            except Exception as e:
                pass

    # If no valid result found, fallback
    if best_coeffs is None:
        best_coeffs = np.ones(dim) / np.sqrt(dim)
        
    # Ensure validity
    max_abs = np.max(np.abs(best_coeffs))
    if max_abs < 1e-3:
        best_coeffs = best_coeffs * (1e-3 / max_abs)
    elif max_abs > 1e5:
        best_coeffs = best_coeffs * (1e5 / max_abs)
        
    if not np.all(np.isfinite(best_coeffs)):
        best_coeffs = np.ones(dim) / np.sqrt(dim)
        
    return best_coeffs.astype(np.float64)
