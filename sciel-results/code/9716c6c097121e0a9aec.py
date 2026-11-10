# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import lpmv, factorial
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# Precompute quadrature nodes and weights
_N_GL = 70
_N_PHI = 130
_x_gl, _w_gl = np.polynomial.legendre.leggauss(_N_GL)
_theta_gl = np.arccos(_x_gl)
_phi_grid = 2 * np.pi * np.arange(_N_PHI) / _N_PHI
_dphi = 2 * np.pi / _N_PHI

# Precompute spherical harmonic values at all quadrature points
# Shape: (N_points, 21) for m = -10..10
_k = 10
_N_COEFFS = 2 * _k + 1
_N_POINTS = _N_GL * _N_PHI

# Flatten theta and phi grids for matrix multiplication
_theta_grid, _phi_grid_2d = np.meshgrid(_theta_gl, _phi_grid, indexing='ij')
_theta_flat = _theta_grid.flatten()
_phi_flat = _phi_grid_2d.flatten()
_cos_theta_flat = _x_gl[:, None] * np.ones((1, _N_PHI))
_cos_theta_flat = _cos_theta_flat.flatten()

# Weights
_W_flat = np.tile(_w_gl, _N_PHI) * _dphi

# Y matrix: (N_POINTS, _N_COEFFS)
_Y_matrix = np.zeros((_N_POINTS, _N_COEFFS))

for idx, m in enumerate(range(-_k, _k + 1)):
    abs_m = abs(m)
    P = lpmv(abs_m, _k, _x_gl)  # shape (70,)
    P_flat = np.tile(P, _N_PHI)
    
    n_fact = factorial(_k - abs_m, exact=True)
    d_fact = factorial(_k + abs_m, exact=True)
    N_m = np.sqrt((2 * _k + 1) / (4 * np.pi) * (n_fact / d_fact))
    
    if m == 0:
        _Y_matrix[:, idx] = N_m * P_flat
    elif m > 0:
        cos_term = np.cos(m * _phi_flat)
        _Y_matrix[:, idx] = np.sqrt(2) * N_m * P_flat * cos_term
    else:
        sin_term = np.sin(abs_m * _phi_flat)
        _Y_matrix[:, idx] = np.sqrt(2) * N_m * P_flat * sin_term

def _compute_ratio(c):
    """
    Compute R = ||f||_4 / ||f||_2 using precomputed quadrature.
    c: length 21 vector
    """
    if np.any(np.isnan(c)) or np.any(np.isinf(c)):
        return np.inf
    
    f_val = _Y_matrix @ c
    f2 = f_val ** 2
    f4 = f_val ** 4
    
    norm2_sq = np.dot(f2, _W_flat)
    norm4_4 = np.dot(f4, _W_flat)
    
    if norm2_sq <= 0:
        return np.inf
    
    norm2 = np.sqrt(norm2_sq)
    norm4 = norm4_4 ** 0.25
    
    return norm4 / norm2

def _compute_ratio_normalized(c):
    """
    Compute R = ||f||_4 / ||f||_2 with c normalized to unit L2 norm first.
    This removes scale invariance and improves gradient conditioning.
    c: length 21 vector
    """
    if np.any(np.isnan(c)) or np.any(np.isinf(c)):
        return np.inf
    
    norm_c = np.linalg.norm(c)
    if norm_c < 1e-15:
        return np.inf
    
    c_norm = c / norm_c
    f_val = _Y_matrix @ c_norm
    f2 = f_val ** 2
    f4 = f_val ** 4
    
    norm2_sq = np.dot(f2, _W_flat)
    norm4_4 = np.dot(f4, _W_flat)
    
    if norm2_sq <= 0:
        return np.inf
    
    norm2 = np.sqrt(norm2_sq)
    norm4 = norm4_4 ** 0.25
    
    return norm4 / norm2

def _compute_ratio_and_grad(c):
    """
    Compute R = ||f||_4 / ||f||_2 and its gradient with respect to c.
    Assumes c is already on the unit sphere (||c|| = 1).
    Returns (ratio, gradient).
    """
    if np.any(np.isnan(c)) or np.any(np.isinf(c)):
        return np.inf, np.zeros_like(c)
    
    f_val = _Y_matrix @ c  # shape (N_POINTS,)
    f2 = f_val ** 2
    f4 = f_val ** 4
    
    norm2_sq = np.dot(f2, _W_flat)
    norm4_4 = np.dot(f4, _W_flat)
    
    if norm2_sq <= 0 or norm4_4 <= 0:
        return np.inf, np.zeros_like(c)
    
    norm2 = np.sqrt(norm2_sq)
    norm4 = norm4_4 ** 0.25
    ratio = norm4 / norm2
    
    # Gradient of R = ||f||_4 / ||f||_2 with respect to c:
    # dR/dc = f^3 * Y / (||f||_4^3 * ||f||_2) - ||f||_4 * f * Y / ||f||_2^3
    bracket = f_val**3 / (norm4**3 * norm2) - norm4 * f_val / (norm2**3)
    grad = _Y_matrix.T @ (_W_flat * bracket)
    
    return ratio, grad

def _trust_region_step(c, grad, delta=0.5):
    """
    Trust-region step on the unit sphere.
    c: current point on unit sphere
    grad: gradient at c
    delta: trust region radius
    Returns: new point on unit sphere, and whether we should increase/decrease delta.
    """
    # Project gradient to tangent space
    grad_tangent = grad - np.dot(grad, c) * c
    grad_norm = np.linalg.norm(grad_tangent)
    
    if grad_norm < 1e-15:
        return c, 'converged'
    
    # Direction: negative projected gradient
    d = -grad_tangent / grad_norm
    
    # Try several step sizes within trust region
    best_c = c
    best_ratio = _compute_ratio_normalized(c)
    improved = False
    
    for alpha in [delta, delta * 0.5, delta * 0.25, delta * 0.1, delta * 0.01]:
        # Move along geodesic: c_new = cos(alpha)*c + sin(alpha)*d
        c_new = np.cos(alpha) * c + np.sin(alpha) * d
        c_new = c_new / np.linalg.norm(c_new)  # Re-normalize for numerical stability
        
        r_new = _compute_ratio_normalized(c_new)
        if r_new < best_ratio - 1e-15:
            best_c = c_new
            best_ratio = r_new
            improved = True
            break
    
    if improved:
        return best_c, 'increase'
    else:
        # Try smaller steps
        for alpha in [delta * 0.001, delta * 0.0001]:
            c_new = np.cos(alpha) * c + np.sin(alpha) * d
            c_new = c_new / np.linalg.norm(c_new)
            r_new = _compute_ratio_normalized(c_new)
            if r_new < best_ratio - 1e-15:
                best_c = c_new
                best_ratio = r_new
                improved = True
                break
        if improved:
            return best_c, 'decrease'
        else:
            return c, 'decrease'

def _trust_region_optimize(c_init, deadline, max_iter=200):
    """
    Trust-region optimization on the unit sphere.
    """
    c = c_init.copy()
    c = c / np.linalg.norm(c)
    
    delta = 0.5  # Initial trust region radius
    best_ratio = _compute_ratio_normalized(c)
    best_c = c.copy()
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        ratio, grad = _compute_ratio_and_grad(c)
        if not np.isfinite(ratio):
            break
        
        if ratio < best_ratio:
            best_ratio = ratio
            best_c = c.copy()
        
        # Check convergence
        grad_tangent = grad - np.dot(grad, c) * c
        if np.linalg.norm(grad_tangent) < 1e-12:
            break
        
        c_new, status = _trust_region_step(c, grad, delta)
        
        if status == 'converged':
            break
        
        r_new = _compute_ratio_normalized(c_new)
        if r_new < best_ratio:
            best_ratio = r_new
            best_c = c_new.copy()
        
        c = c_new
        
        if status == 'increase':
            delta = min(delta * 2.0, 1.0)
        else:
            delta = max(delta * 0.5, 1e-6)
    
    return best_c, best_ratio

def rudin_problem10() -> np.ndarray:
    """
    Find the real coefficients (c_m) for m = -10..10 of a real degree-10 spherical harmonic
    that MINIMIZE the norm ratio ||f||_4 / ||f||_2.
    """
    start_time = time.time()
    deadline = 630  # seconds, with safety margin before 630s kill
    
    best_coeffs = None
    best_ratio = np.inf
    
    def _update_best(c, r):
        nonlocal best_coeffs, best_ratio
        if r < best_ratio and np.all(np.isfinite(c)):
            best_ratio = r
            best_coeffs = c.copy()
    
    n_coeffs = _N_COEFFS
    rng = np.random.default_rng(42)
    
    # 1. Initial structured guesses
    candidates = []
    
    # All ones
    candidates.append(np.ones(n_coeffs))
    
    # Single modes
    for i in range(n_coeffs):
        c = np.zeros(n_coeffs)
        c[i] = 1.0
        candidates.append(c)
    
    # Pairs of modes (cos/sin symmetry often helps)
    for i in range(n_coeffs):
        for j in range(i+1, n_coeffs):
            c = np.zeros(n_coeffs)
            c[i] = 1.0
            c[j] = 1.0
            candidates.append(c)
    
    # Sparse random patterns
    for _ in range(30):
        n_nonzero = rng.integers(3, 10)
        indices = rng.choice(n_coeffs, size=n_nonzero, replace=False)
        c = np.zeros(n_coeffs)
        c[indices] = rng.normal(size=n_nonzero)
        candidates.append(c)
        
    # Symmetric patterns: even m only
    c_even = np.zeros(n_coeffs)
    c_even[::2] = 1.0
    candidates.append(c_even)
    
    # Odd m only
    c_odd = np.zeros(n_coeffs)
    c_odd[1::2] = 1.0
    candidates.append(c_odd)
    
    # Alternating signs
    c_alt = np.ones(n_coeffs)
    c_alt[1::2] = -1.0
    candidates.append(c_alt)
    
    # Higher-order harmonic-like patterns
    for freq in [1, 2, 3]:
        c_harm = np.zeros(n_coeffs)
        for i in range(n_coeffs):
            m = i - 10
            c_harm[i] = np.cos(freq * m * np.pi / 10)
        candidates.append(c_harm)
    
    # 2. Evaluate all candidates
    for c in candidates:
        r = _compute_ratio(c)
        _update_best(c, r)
        
    # 3. Local optimization from best candidates using trust-region method
    # Sort candidates by ratio
    scored_candidates = []
    for c in candidates:
        r = _compute_ratio(c)
        scored_candidates.append((r, c))
    scored_candidates.sort(key=lambda x: x[0])
    
    # Take top 20 and run trust-region optimization
    for i, (r_init, c_init) in enumerate(scored_candidates[:20]):
        if time.time() > deadline - 120:
            break
        
        c_opt, r_opt = _trust_region_optimize(c_init, deadline - 30)
        if np.isfinite(r_opt):
            _update_best(c_opt, r_opt)
    
    # 4. Random restarts with trust-region optimization
    n_restarts = 100
    for i in range(n_restarts):
        if time.time() > deadline - 60:
            break
            
        c_start = rng.normal(size=n_coeffs)
        norm_c = np.linalg.norm(c_start)
        if norm_c < 1e-15:
            continue
        c_start_norm = c_start / norm_c
        
        c_opt, r_opt = _trust_region_optimize(c_start_norm, deadline - 30)
        if np.isfinite(r_opt):
            _update_best(c_opt, r_opt)
            
    # 5. Final polish: run trust-region on the best result with more iterations
    if best_coeffs is not None:
        c_opt, r_opt = _trust_region_optimize(best_coeffs, deadline - 10, max_iter=500)
        if np.isfinite(r_opt):
            _update_best(c_opt, r_opt)
    
    # 6. Second polish with even more iterations
    if best_coeffs is not None and time.time() < deadline - 5:
        c_opt, r_opt = _trust_region_optimize(best_coeffs, deadline, max_iter=1000)
        if np.isfinite(r_opt):
            _update_best(c_opt, r_opt)

    # Final validation
    if best_coeffs is None or np.any(np.isnan(best_coeffs)) or np.any(np.isinf(best_coeffs)):
        best_coeffs = np.ones(n_coeffs)
    
    max_abs = np.max(np.abs(best_coeffs))
    if max_abs < 1e-3 or max_abs > 1e5:
        if max_abs == 0:
            best_coeffs = np.ones(n_coeffs)
        else:
            best_coeffs = best_coeffs / max_abs * 1.0
            if np.max(np.abs(best_coeffs)) < 1e-3:
                best_coeffs = np.ones(n_coeffs)
    
    return best_coeffs.astype(np.float64)

# EVOLVE-BLOCK-END
