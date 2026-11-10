# EVOLVE-BLOCK-START
import numpy as np
import time
import sys
from scipy.optimize import minimize
from scipy.special import lpmv, factorial

def _compute_ratio(coeffs):
    """
    Compute the ratio ||f||_4 / ||f||_2 for a given coefficient vector.
    coeffs: array of shape (21,) corresponding to m = -10 .. 10.
    """
    k = 10
    # Quadrature points
    # 70-point Gauss-Legendre for cos(theta) in [-1, 1]
    x, w = np.polynomial.legendre.leggauss(70)
    # x is cos(theta), w are weights.
    # The integral over theta is ∫_{-1}^{1} ... dx.
    # The integral over phi is ∫_0^{2pi} ... dphi.
    # Total integral is sum_i sum_j w_i * (2pi/N_phi) * f(x_i, phi_j)
    
    N_phi = 130
    phi = np.linspace(0, 2 * np.pi, N_phi, endpoint=False)
    dphi = 2 * np.pi / N_phi
    
    # Create meshgrid
    # x shape (70,), phi shape (130,)
    # We need f(x_i, phi_j)
    # It is more efficient to loop over x or phi?
    # Let's construct the basis matrix B of shape (70*130, 21)
    # Then f = B @ coeffs
    
    # Precompute basis values
    # Y_0 = N_0 P^0
    # Y_m = sqrt(2) N_m P^m cos(m phi) for m > 0
    # Y_m = sqrt(2) N_|m| P^|m| sin(|m| phi) for m < 0
    
    # P^m = P_10^m(cos theta) with Condon-Shortley phase.
    # scipy.special.lpmv(m, n, x) includes Condon-Shortley phase.
    
    B = np.zeros((70 * 130, 21), dtype=np.float64)
    
    # Weights for integration
    # w_i is for dx (cos theta).
    # The surface measure dOmega = sin(theta) dtheta dphi = -d(cos theta) dphi.
    # So the integral is ∫_{-1}^{1} ∫_0^{2pi} f(x, phi) dphi dx.
    # Discretized: sum_i sum_j w_i * dphi * f(x_i, phi_j)
    
    # Reshape x and phi for broadcasting
    # x: (70, 1), phi: (1, 130)
    x_grid, phi_grid = np.meshgrid(x, phi, indexing='ij')
    x_flat = x_grid.ravel()
    phi_flat = phi_grid.ravel()
    
    # Calculate P_10^m for each x
    # We only need m = 0, 1, ..., 10
    P_vals = np.zeros((11, 70 * 130), dtype=np.float64)
    for m in range(11):
        # lpmv(m, 10, x_flat)
        P_vals[m] = lpmv(m, 10, x_flat)
        
    # Calculate Normalization constants N_m
    # N_m = sqrt(21/(4 pi) * (10-m)!/(10+m)!)
    N_vals = np.zeros(11, dtype=np.float64)
    for m in range(11):
        N_vals[m] = np.sqrt(21.0 / (4.0 * np.pi) * factorial(10 - m) / factorial(10 + m))
        
    # Construct B
    # Index mapping: c[i] belongs to m = i - 10
    # m = 0 -> i = 10
    # m > 0 -> i = 10 + m
    # m < 0 -> i = 10 + m = 10 - |m|
    
    # m = 0
    B[:, 10] = N_vals[0] * P_vals[0]
    
    for m in range(1, 11):
        # m > 0
        idx = 10 + m
        B[:, idx] = np.sqrt(2.0) * N_vals[m] * P_vals[m] * np.cos(m * phi_flat)
        
        # m < 0
        idx = 10 - m
        B[:, idx] = np.sqrt(2.0) * N_vals[m] * P_vals[m] * np.sin(m * phi_flat)
        
    # f = B @ coeffs
    f = B @ coeffs
    
    # Norms
    # ||f||_2^2 = sum w_i dphi f^2
    # ||f||_4^2 = sum w_i dphi f^4
    # Note: The weights w_i are for the integral over [-1, 1].
    # The quadrature rule is sum w_i f(x_i) ≈ ∫_{-1}^1 f(x) dx.
    # So we multiply by w_i and dphi.
    
    weights = np.repeat(w, N_phi) # shape (70*130,)
    weights *= dphi
    
    f2 = f * f
    f4 = f2 * f2
    
    norm2_sq = np.sum(weights * f2)
    norm4_sq = np.sum(weights * f4)
    
    # Avoid division by zero
    if norm2_sq <= 1e-15:
        return np.inf
        
    norm2 = np.sqrt(norm2_sq)
    norm4 = np.sqrt(norm4_sq)
    
    return norm4 / norm2

def rudin_problem10() -> np.ndarray:
    k = 10
    start_time = time.time()
    deadline = 580.0 # Safety margin
    
    # Initial guess: all ones
    c0 = np.ones(2 * k + 1, dtype=np.float64)
    
    best_c = c0.copy()
    best_score = _compute_ratio(best_c)
    
    # Try a few random restarts or just one solid optimization
    # Nelder-Mead is a good choice for this dimensionality and smoothness
    
    options = {
        'maxiter': 1000,
        'xatol': 1e-6,
        'fatol': 1e-6
    }
    
    # Run optimization
    try:
        res = minimize(_compute_ratio, c0, method='Nelder-Mead', options=options)
        if res.success and res.fun < best_score:
            best_c = res.x
            best_score = res.fun
    except Exception:
        pass
        
    # If time permits, try another random start
    if time.time() - start_time < deadline - 30:
        rng = np.random.default_rng(42)
        c1 = rng.normal(size=2 * k + 1)
        # Normalize to avoid issues? The ratio is scale invariant, so it doesn't matter.
        # But large values might cause numerical issues in f4. Let's keep it simple.
        try:
            res1 = minimize(_compute_ratio, c1, method='Nelder-Mead', options=options)
            if res1.success and res1.fun < best_score:
                best_c = res1.x
                best_score = res1.fun
        except Exception:
            pass
            
    # Final check for validity
    if not np.all(np.isfinite(best_c)):
        return c0
        
    max_abs = np.max(np.abs(best_c))
    if max_abs < 1e-3 or max_abs > 1e5:
        # Rescale if necessary, though Nelder-Mead usually stays in reasonable bounds
        # If it's too small, scale up. If too big, scale down.
        # The ratio is invariant to scaling, so we can just rescale.
        target_mag = 1.0
        if max_abs > 0:
            best_c = best_c * (target_mag / max_abs)
            
    return best_c

# EVOLVE-BLOCK-END
