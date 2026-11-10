# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.special import lpmv
import math

def rudin_problem10() -> np.ndarray:
    """
    Finds the coefficients for a degree-10 spherical harmonic that minimizes
    the ratio ||f||_4 / ||f||_2.
    
    Returns:
        np.ndarray of shape (21,) with coefficients c_m for m = -10..10.
    """
    k = 10
    n_m = 2 * k + 1
    ms = np.arange(-k, k + 1)
    
    # Quadrature setup
    n_theta = 70
    n_phi = 130
    
    # Gauss-Legendre nodes and weights in x = cos(theta)
    x, w = np.polynomial.legendre.leggauss(n_theta)
    # x is in [-1, 1]. theta = arccos(x).
    # The integral over the sphere is \int_0^{2\pi} d\phi \int_{-1}^1 dx f(\theta, \phi).
    # So the weight for each (x_i, phi_j) is w_i * (2*pi / n_phi).
    dphi = 2 * np.pi / n_phi
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    
    # Create meshgrid
    # We need values for all m at all points.
    # Shape: (n_theta, n_phi, n_m)
    # Let's compute the harmonics.
    
    # P_l^m(x) with Condon-Shortley phase is given by lpmv(m, l, x) in scipy?
    # scipy.special.lpmv(m, v, x) computes P_v^m(x) including Condon-Shortley phase (-1)^m.
    # The definition in the prompt: P^m = P_10^m(cos theta) with Condon-Shortley phase.
    # So we can use lpmv directly.
    
    # Normalization constants N_m
    # N_m = sqrt( (2l+1)/(4 pi) * (l-m)!/(l+m)! )
    # For real orthonormal harmonics:
    # Y_0 = N_0 P_l^0
    # Y_m = sqrt(2) N_m P_l^m cos(m phi) for m > 0
    # Y_m = sqrt(2) N_|m| P_l^{|m|} sin(|m| phi) for m < 0
    
    N = np.zeros(n_m)
    for idx, m in enumerate(ms):
        am = abs(m)
        # Factorial ratio (l-m)! / (l+m)!
        # Using log-gamma for stability, though l=10 is small.
        log_ratio = math.lgamma(l - am + 1) - math.lgamma(l + am + 1)
        N[idx] = np.sqrt((2 * k + 1) / (4 * np.pi) * np.exp(log_ratio))
        
    # Precompute harmonic values
    # H[i, j, idx] = Y_{ms[idx]}(theta_i, phi_j)
    H = np.zeros((n_theta, n_phi, n_m))
    
    # cos(theta) = x
    # sin(theta) = sqrt(1-x^2)
    # We need P_l^m(x).
    # lpmv(m, l, x) returns P_l^m(x).
    
    for idx, m in enumerate(ms):
        am = abs(m)
        # Compute P_k^am(x)
        # Note: lpmv expects m as first arg, l as second.
        P_vals = lpmv(am, k, x) # Shape (n_theta,)
        
        if m == 0:
            # Y_0 = N_0 P_k^0
            # P_vals is P_k^0(x). Broadcast to phi.
            H[:, :, idx] = N[idx] * P_vals[:, np.newaxis]
        elif m > 0:
            # Y_m = sqrt(2) N_m P_k^m cos(m phi)
            cos_term = np.cos(m * phi) # Shape (n_phi,)
            # P_vals[:, np.newaxis] * cos_term[np.newaxis, :]
            H[:, :, idx] = np.sqrt(2) * N[idx] * P_vals[:, np.newaxis] * cos_term[np.newaxis, :]
        else:
            # m < 0, am = -m
            # Y_m = sqrt(2) N_am P_k^am sin(am phi)
            sin_term = np.sin(am * phi) # Shape (n_phi,)
            H[:, :, idx] = np.sqrt(2) * N[idx] * P_vals[:, np.newaxis] * sin_term[np.newaxis, :]
            
    # Flatten H for matrix multiplication
    # H_flat shape: (n_theta * n_phi, n_m)
    H_flat = H.reshape(-1, n_m)
    
    # Weights
    # w_i for x, dphi for phi.
    # Total weight for point (i, j) is w_i * dphi.
    weights_x = w # Shape (n_theta,)
    weights_phi = np.full(n_phi, dphi)
    # Outer product of weights
    W = np.kron(weights_x, weights_phi) # Shape (n_theta * n_phi,)
    # Wait, np.kron(w_x, w_phi) with w_x length n_theta and w_phi length n_phi
    # produces array where element (i*n_phi + j) is w_x[i] * w_phi[j].
    # This matches the reshape order (C-order) of H.reshape(-1, n_m) if H was (n_theta, n_phi, ...).
    # H[i, j, :] becomes row i*n_phi + j.
    # Yes, this is correct.
    
    # Objective function and gradient
    def objective_and_grad(c):
        # f values at all points
        f = H_flat @ c # Shape (n_pts,)
        
        # Norms
        # ||f||_2^2 = sum(W * f^2)
        # ||f||_4^4 = sum(W * f^4)
        
        f2 = f * f
        f4 = f2 * f2
        
        N2 = np.dot(W, f2)
        N4 = np.dot(W, f4)
        
        if N2 < 1e-12:
            return 1e10, np.zeros_like(c)
            
        R = N4**0.25 / np.sqrt(N2)
        
        # Gradient calculation
        # dR/dc_k = N4^(-3/4) N2^(-1/2) * sum(W * f^3 * Y_k) - 0.5 * N4^(1/4) N2^(-3/2) * sum(W * f * Y_k)
        
        # sum(W * f^3 * Y_k) for all k:
        # This is H_flat.T @ (W * f^3)
        # sum(W * f * Y_k) for all k:
        # This is H_flat.T @ (W * f)
        
        grad_term_4 = H_flat.T @ (W * (f * f2)) # W * f^3
        grad_term_2 = H_flat.T @ (W * f)
        
        coeff_4 = N4**(-0.75) * N2**(-0.5)
        coeff_2 = 0.5 * N4**0.25 * N2**(-1.5)
        
        grad = coeff_4 * grad_term_4 - coeff_2 * grad_term_2
        
        return R, grad

    # Initial guess: all ones
    c0 = np.ones(n_m)
    
    # Optimization
    # Use L-BFGS-B which handles bounds well, though we don't strictly need bounds 
    # other than avoiding zero vector.
    # We can try to fix the norm to 1 to avoid scale issues, but the ratio is scale invariant.
    # L-BFGS-B should handle it.
    
    try:
        res = minimize(
            lambda c: objective_and_grad(c)[0],
            c0,
            jac=lambda c: objective_and_grad(c)[1],
            method='L-BFGS-B',
            options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-12}
        )
        c_opt = res.x
    except Exception:
        c_opt = c0
        
    # Validation
    # Ensure no NaN or Inf
    if not np.all(np.isfinite(c_opt)):
        c_opt = c0
        
    # Check magnitude constraints
    max_c = np.max(np.abs(c_opt))
    if max_c < 1e-3 or max_c > 1e5:
        # Rescale if necessary
        if max_c < 1e-3:
            c_opt = c_opt / max_c * 1.0
        elif max_c > 1e5:
            c_opt = c_opt / max_c * 1.0
            
    # Final check for validity
    if not np.all(np.isfinite(c_opt)) or len(c_opt) != 21:
        return np.ones(21)
        
    return c_opt

# EVOLVE-BLOCK-END
