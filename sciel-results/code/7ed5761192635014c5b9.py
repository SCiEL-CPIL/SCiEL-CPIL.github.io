# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import lpmv
from scipy.optimize import minimize, differential_evolution

# Constants
K = 10
M_VALUES = np.arange(-K, K + 1)

# Precompute normalization constants N_m for real orthonormal spherical harmonics
N = np.zeros(2 * K + 1)
for i, m in enumerate(M_VALUES):
    abs_m = abs(m)
    from math import lgamma, log, pi
    log_n_sq = log(21.0 / (4.0 * pi)) + lgamma(11 - abs_m) - lgamma(11 + abs_m)
    N[i] = np.sqrt(np.exp(log_n_sq))

# Quadrature setup
n_theta = 70
n_phi = 130
x_gl, w_gl = np.polynomial.legendre.leggauss(n_theta)
phi_grid = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)

x_grid, phi_grid_2d = np.meshgrid(x_gl, phi_grid, indexing='ij')

# Precompute the basis functions Y_m(x, phi) for all m
Y_basis = np.zeros((21, n_theta, n_phi))

for i, m in enumerate(M_VALUES):
    abs_m = abs(m)
    p_vals = lpmv(abs_m, 10, x_gl)
    
    if m == 0:
        Y_basis[i] = N[i] * p_vals[:, np.newaxis]
    elif m > 0:
        cos_term = np.cos(m * phi_grid_2d)
        Y_basis[i] = np.sqrt(2.0) * N[i] * p_vals[:, np.newaxis] * cos_term
    else:
        sin_term = np.sin(abs_m * phi_grid_2d)
        Y_basis[i] = np.sqrt(2.0) * N[i] * p_vals[:, np.newaxis] * sin_term

# Precompute weights for integration
W_flat = (w_gl[:, np.newaxis] * (2.0 * np.pi / n_phi)).flatten()
Y_flat = Y_basis.reshape(21, -1)

def evaluate_ratio(coeffs):
    """
    Computes R = ||f||_4 / ||f||_2 for given coefficients.
    """
    f_flat = coeffs @ Y_flat
    norm2_sq = np.sum(W_flat * f_flat**2)
    norm4_4 = np.sum(W_flat * f_flat**4)
    
    if norm2_sq <= 0:
        return np.inf
    
    norm2 = np.sqrt(norm2_sq)
    norm4 = np.sqrt(np.sqrt(norm4_4))
    
    return norm4 / norm2

def objective(c):
    # Ensure finite values
    if not np.all(np.isfinite(c)):
        return 1e10
    # Penalize out of bounds if we were using bounds, but DE handles bounds internally.
    # Here we just compute the ratio.
    val = evaluate_ratio(c)
    if not np.isfinite(val):
        return 1e10
    return val

def refine_local(coeffs, time_limit):
    """
    Perform local optimization using L-BFGS-B.
    """
    start_time = time.time()
    # L-BFGS-B is a good choice for smooth multi-dimensional problems
    # We use a loose bound to allow scaling, but the ratio is scale-invariant, 
    # so the optimizer might drift in scale. We can fix the norm to 1 to help convergence.
    # However, the problem asks for coefficients, and the ratio is scale invariant.
    # To avoid drift, we can normalize inside the objective or just let it be.
    # Let's stick to the raw coefficients but use a reasonable bound.
    
    def obj_normalized(c):
        # Normalize to unit L2 norm to stabilize the landscape for L-BFGS
        # This prevents the optimizer from just scaling up/down to find "small" or "large" values
        # that might cause numerical issues or slow convergence.
        norm = np.linalg.norm(c)
        if norm < 1e-10:
            return 1e10
        c_norm = c / norm
        return evaluate_ratio(c_norm)
    
    try:
        result = minimize(obj_normalized, coeffs, method='L-BFGS-B', 
                          options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12, 'maxls': 50})
        
        if result.success and result.fun < 1e9:
            best_c = result.x
            # Scale back to a reasonable magnitude, e.g., max abs coeff ~ 1
            max_c = np.max(np.abs(best_c))
            if max_c > 0:
                best_c = best_c / max_c
            return best_c
    except Exception:
        pass
        
    return coeffs

def run_search(time_budget):
    start_time = time.time()
    def time_left():
        return time_budget - (time.time() - start_time)
    
    # Define bounds for Differential Evolution
    # The ratio is scale invariant, so the magnitude doesn't matter for the score,
    # but we need to stay within [1e-3, 1e5] for validity.
    # Let's search in [-10, 10] for each coefficient.
    bounds = [(-10.0, 10.0)] * 21
    
    # Differential Evolution is a global optimizer.
    # popsize: 15 * dim = 15 * 21 = 315 individuals.
    # This is a large population, which helps with global exploration.
    # maxiter: We limit iterations to fit within time budget.
    # We'll run DE for a portion of the time, then polish.
    
    de_time_budget = time_budget * 0.7
    polish_time_budget = time_budget * 0.3
    
    best_val = np.inf
    best_coeffs = np.ones(21)
    
    # Run Differential Evolution
    # We use a callback to stop if time is running out
    def de_callback(xk, convergence):
        if time_left() < de_time_budget * 0.2: # Stop if we've used 80% of DE time
            return True
        return False

    try:
        # seed=42 for reproducibility
        result = differential_evolution(objective, bounds, maxiter=200, popsize=10, 
                                        tol=1e-8, atol=1e-8, seed=42, 
                                        callback=de_callback, polish=False, 
                                        init='sobol')
        
        if result.fun < best_val:
            best_val = result.fun
            best_coeffs = result.x.copy()
            
    except Exception as e:
        # Fallback if DE fails
        pass
        
    # Local Polish
    if time_left() > 1.0:
        # Use the best result from DE as starting point for local refinement
        refined = refine_local(best_coeffs, polish_time_budget)
        if np.all(np.isfinite(refined)):
            val_ref = evaluate_ratio(refined)
            if val_ref < best_val:
                best_val = val_ref
                best_coeffs = refined

    # Additional random restarts if time permits
    while time_left() > 5.0:
        # Generate a new random starting point
        c_init = np.random.uniform(-5, 5, 21)
        refined = refine_local(c_init, 1.0) # Short polish
        if np.all(np.isfinite(refined)):
            val_ref = evaluate_ratio(refined)
            if val_ref < best_val:
                best_val = val_ref
                best_coeffs = refined
        else:
            # If refinement failed, try a simple random walk step
            c_step = best_coeffs + np.random.randn(21) * 0.5
            val_step = evaluate_ratio(c_step)
            if val_step < best_val:
                best_val = val_step
                best_coeffs = c_step

    return best_coeffs

def rudin_problem10() -> np.ndarray:
    """
    Rudin problem for polynomials on the 2-sphere. Find the real coefficients
    (c_m) for m = -k..k of a real degree-k spherical harmonic

        f = sum_{m=-k}^{k} c_m Y_k^m

    on the round unit sphere S^2 that MINIMIZE the norm ratio

        ||f||_{L^4(S^2)} / ||f||_{L^2(S^2)}

    (norms taken with the standard surface measure dOmega of total mass 4*pi).
    Here the degree is fixed to k = 10, so the returned vector has length
    2*k + 1 = 21.

    Returns:
        coeffs: np.ndarray of shape (21,) with the real coefficients
            c_m, ordered m = -10, -9, ..., 0, ..., 9, 10.
    """
    # Time budget management
    start_time = time.time()
    max_time = 600.0 # Safety margin from 630s limit
    
    try:
        best_coeffs = run_search(max_time)
    except Exception as e:
        # Fallback to baseline if something goes wrong
        best_coeffs = np.ones(21)
        
    # Final validation and cleanup
    if best_coeffs is None or not np.all(np.isfinite(best_coeffs)):
        best_coeffs = np.ones(21)
        
    # Ensure size is 21
    if len(best_coeffs) != 21:
        best_coeffs = np.ones(21)
        
    # Check max abs coefficient constraints
    max_c = np.max(np.abs(best_coeffs))
    if max_c < 1e-3 or max_c > 1e5:
        # Normalize to be within range, e.g., scale to max 1.0
        if max_c > 0:
            best_coeffs = best_coeffs / max_c * 1.0
        else:
            best_coeffs = np.ones(21)
            
    return best_coeffs

# EVOLVE-BLOCK-END
