# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import lpmv
import time
import sys

def _build_quadrature(n_theta=70, n_phi=130):
    """Build 70-point Gauss-Legendre in cos(theta) and 130 equally spaced phi."""
    x, w = np.polynomial.legendre.leggauss(n_theta)
    # x in [-1, 1], w are weights for integral over [-1,1]
    # For sphere: integral = sum w_i * (2/pi) * sum_j f(theta_i, phi_j) * (dphi)
    # Actually: dOmega = dphi * d(cos theta) = dphi * (-sin theta dtheta)
    # Standard: integral_{S^2} f dOmega = integral_{-1}^{1} integral_0^{2pi} f(theta,phi) dphi dx
    # where x = cos(theta). The Gauss-Legendre nodes x_i and weights w_i approximate
    # integral_{-1}^{1} g(x) dx = sum w_i g(x_i)
    # So integral_{S^2} f dOmega = sum_i w_i * integral_0^{2pi} f(theta_i, phi) dphi
    # With uniform phi: integral_0^{2pi} f dphi ≈ (2pi/n_phi) * sum_j f(theta_i, phi_j)
    
    theta = np.arccos(x)
    phi = np.linspace(0, 2*np.pi, n_phi, endpoint=False)
    
    # Full grid
    THETA, PHI = np.meshgrid(theta, phi, indexing='ij')  # shape (n_theta, n_phi)
    # Weights: w_i * (2*pi/n_phi)
    W = np.outer(w, np.ones(n_phi) * (2.0*np.pi/n_phi))
    
    return THETA, PHI, W, x, w

def _spherical_harmonics_matrix(theta, phi, k=10):
    """
    Build matrix Y where Y[i,j] = Y_j(theta_i, phi_i) for j = -k..k (2k+1 columns).
    Real orthonormal spherical harmonics:
      Y_0 = N_0 * P_k^0(cos theta)
      Y_m = sqrt(2) * N_m * P_k^m(cos theta) * cos(m*phi)  for m > 0
      Y_m = sqrt(2) * N_|m| * P_k^|m|(cos theta) * sin(|m|*phi)  for m < 0
    where N_m = sqrt((2k+1)/(4*pi) * (k-m)!/(k+m)!)
    """
    n_theta, n_phi = theta.shape
    cos_theta = np.cos(theta)
    m_vals = np.arange(-k, k+1)
    Y = np.zeros((n_theta, n_phi, 2*k+1))
    
    for idx, m in enumerate(m_vals):
        am = abs(m)
        # N_m
        # (k-am)!/(k+am)!
        # Use log gamma for numerical stability
        log_num = np.log(2*k+1) - np.log(4*np.pi)
        # (k-am)! / (k+am)!
        # For k=10, am<=10, these are integers, so we can compute directly
        num_fact = 1.0
        for i in range(1, k-am+1):
            num_fact *= i
        den_fact = 1.0
        for i in range(1, k+am+1):
            den_fact *= i
        N_m = np.sqrt((2*k+1)/(4*np.pi) * num_fact/den_fact)
        
        # P_k^am with Condon-Shortley phase
        P = lpmv(am, k, cos_theta)  # scipy includes Condon-Shortley phase
        
        if m == 0:
            Y[:,:,idx] = N_m * P
        elif m > 0:
            Y[:,:,idx] = np.sqrt(2) * N_m * P * np.cos(m*phi)
        else:
            Y[:,:,idx] = np.sqrt(2) * N_m * P * np.sin(am*phi)
    
    return Y

def _compute_ratio(coeffs, Y, W):
    """
    Compute R = ||f||_4 / ||f||_2 where f = sum_m c_m Y_m.
    """
    # f has shape (n_theta, n_phi)
    f = np.tensordot(coeffs, Y, axes=([0],[2]))
    
    # ||f||_2^2 = integral |f|^2 dOmega = sum W * f^2
    f2 = f**2
    norm2_sq = np.sum(W * f2)
    
    # ||f||_4^4 = integral |f|^4 dOmega = sum W * f^4
    f4 = f2**2
    norm4_4 = np.sum(W * f4)
    
    if norm2_sq <= 0:
        return np.inf
    
    norm2 = np.sqrt(norm2_sq)
    norm4 = np.sqrt(np.sqrt(norm4_4))
    
    return norm4 / norm2

def _nelder_mead_minimize(func, x0, max_iter=2000, tol=1e-10, deadline=None):
    """
    Simple Nelder-Mead minimization.
    """
    n = len(x0)
    # Initial simplex
    simplex = [x0.copy()]
    for i in range(n):
        xi = x0.copy()
        xi[i] += 0.1 if xi[i] != 0 else 0.1
        simplex.append(xi)
    
    # Evaluate initial simplex
    fvals = [func(v) for v in simplex]
    
    alpha = 1.0  # reflection
    gamma = 2.0  # expansion
    rho = 0.5    # contraction
    sigma = 0.5  # shrink
    
    best_idx = np.argmin(fvals)
    best_val = fvals[best_idx]
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline - 30:
            break
        
        # Sort simplex by function value
        order = np.argsort(fvals)
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]
        
        # Check convergence
        if abs(fvals[-1] - fvals[0]) < tol * (1 + abs(fvals[0])):
            break
        
        # Centroid of all but worst point
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        xr = centroid + alpha * (centroid - simplex[-1])
        fr = func(xr)
        
        if fr < fvals[0]:
            # Expansion
            xe = centroid + gamma * (centroid - simplex[-1])
            fe = func(xe)
            if fe < fr:
                simplex[-1] = xe
                fvals[-1] = fe
            else:
                simplex[-1] = xr
                fvals[-1] = fr
        elif fr < fvals[-2]:
            simplex[-1] = xr
            fvals[-1] = fr
        else:
            # Contraction
            if fr < fvals[-1]:
                # Outside contraction
                xc = centroid + rho * (simplex[-1] - centroid)
            else:
                # Inside contraction
                xc = centroid - rho * (simplex[-1] - centroid)
            fc = func(xc)
            
            if fc < fvals[-1]:
                simplex[-1] = xc
                fvals[-1] = fc
            else:
                # Shrink
                for i in range(1, n+1):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    fvals[i] = func(simplex[i])
        
        # Track best
        current_best = min(fvals)
        if current_best < best_val:
            best_val = current_best
            best_idx = fvals.index(current_best)
    
    return best_val, simplex[np.argmin(fvals)]

def rudin_problem10() -> np.ndarray:
    """
    Find coefficients c_m for m = -10..10 minimizing L4/L2 ratio of degree-10 spherical harmonic.
    """
    start_time = time.time()
    deadline = start_time + 580  # Safety margin
    
    k = 10
    
    # Build quadrature
    THETA, PHI, W, x, w = _build_quadrature(n_theta=70, n_phi=130)
    
    # Build spherical harmonics matrix
    Y = _spherical_harmonics_matrix(THETA, PHI, k=k)
    
    # Objective function
    def objective(coeffs):
        if not np.all(np.isfinite(coeffs)):
            return np.inf
        return _compute_ratio(coeffs, Y, W)
    
    # Initial guess: all ones
    x0 = np.ones(2*k+1)
    
    # Try multiple starting points
    best_coeffs = x0.copy()
    best_val = objective(x0)
    
    # Random restarts
    np.random.seed(42)
    n_restarts = 5
    
    for r in range(n_restarts):
        if time.time() > deadline - 60:
            break
        
        if r == 0:
            x_init = x0.copy()
        else:
            x_init = np.random.randn(2*k+1) * 0.5 + np.random.choice([-1, 1], size=2*k+1) * 0.5
        
        # Normalize initial guess
        if np.max(np.abs(x_init)) > 0:
            x_init = x_init / np.max(np.abs(x_init))
        
        # Run Nelder-Mead
        val, x_opt = _nelder_mead_minimize(objective, x_init, max_iter=500, tol=1e-12, deadline=deadline)
        
        if val < best_val:
            best_val = val
            best_coeffs = x_opt.copy()
    
    # Additional local refinement with smaller steps
    if time.time() < deadline - 60:
        # Try perturbing best solution
        for trial in range(10):
            if time.time() > deadline - 30:
                break
            
            perturbation = np.random.randn(2*k+1) * 0.01
            x_trial = best_coeffs + perturbation
            
            val_trial = objective(x_trial)
            if val_trial < best_val:
                # Refine from here
                val_ref, x_ref = _nelder_mead_minimize(objective, x_trial, max_iter=200, tol=1e-12, deadline=deadline)
                if val_ref < best_val:
                    best_val = val_ref
                    best_coeffs = x_ref.copy()
    
    # Final validation
    coeffs = best_coeffs.astype(np.float64)
    
    # Check constraints
    if len(coeffs) != 21:
        coeffs = x0.copy()
    if not np.all(np.isfinite(coeffs)):
        coeffs = x0.copy()
    max_c = np.max(np.abs(coeffs))
    if max_c < 1e-3 or max_c > 1e5:
        if max_c < 1e-3 and max_c > 0:
            coeffs = coeffs / max_c * 1.0
        elif max_c > 1e5:
            coeffs = coeffs / max_c * 1.0
        else:
            coeffs = x0.copy()
    
    return coeffs

# EVOLVE-BLOCK-END
