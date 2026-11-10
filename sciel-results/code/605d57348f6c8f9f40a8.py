import numpy as np
import time
from scipy.optimize import differential_evolution, minimize
from scipy.special import lpmv, factorial

# Constants for the problem
K = 10
N_M = 2 * K + 1  # 21 coefficients
N_THETA = 70
N_PHI = 130

# Pre-compute quadrature weights and points
def _setup_quadrature():
    # Gauss-Legendre nodes and weights on [-1, 1] for cos(theta)
    x, w = np.polynomial.legendre.leggauss(N_THETA)
    # x = cos(theta), w are weights for integration over cos(theta)
    # The surface measure is dOmega = sin(theta) dtheta dphi = -d(cos(theta)) dphi
    # So the weight for each (x, phi) point is w_x * dphi
    dphi = 2 * np.pi / N_PHI
    weights_x = w * dphi
    
    # Grid of cos(theta) and phi
    cos_thetas = x
    phis = np.linspace(0, 2 * np.pi, N_PHI, endpoint=False)
    
    # Create meshgrid
    C, P = np.meshgrid(cos_thetas, phis, indexing='ij')
    
    # Weights for each point
    W = np.tile(weights_x, (1, N_PHI))
    
    return C, P, W

# Pre-compute spherical harmonics on the grid
def _compute_harmonics(C, P):
    """
    Computes real orthonormal spherical harmonics Y_m for m=-10..10.
    Y_0 = N_0 P^0
    Y_m = sqrt(2) N_m P^m cos(m phi) for m > 0
    Y_m = sqrt(2) N_|m| P^|m| sin(|m| phi) for m < 0
    N_m = sqrt(21/(4 pi) * (10-m)!/(10+m)!)
    P^m = P_10^m(cos theta) with Condon-Shortley phase.
    scipy.special.lpmv includes the Condon-Shortley phase (-1)^m.
    """
    Y = np.zeros((N_M, C.size), dtype=np.float64)
    
    # Normalization constants N_m
    # N_m = sqrt(21/(4*pi) * (10-m)! / (10+m)!)
    # Note: (10-m)!/(10+m)! = 1 / ((10+m) * ... * (10-m+1))
    # Using log factorial for stability or just direct calculation since n is small
    for m in range(-K, K + 1):
        abs_m = abs(m)
        # N_m calculation
        # (10-abs_m)! / (10+abs_m)!
        # We can compute this iteratively or use gamma
        # log(N_m) = 0.5 * (log(21) - log(4*pi) + log(10-abs_m)! - log(10+abs_m)!)
        # Using scipy.special.gammaln
        log_fact_num = np.lgamma(10 - abs_m + 1)
        log_fact_den = np.lgamma(10 + abs_m + 1)
        log_N = 0.5 * (np.log(21.0) - np.log(4.0 * np.pi) + log_fact_num - log_fact_den)
        N_m = np.exp(log_N)
        
        # Associated Legendre polynomial P_10^abs_m(cos(theta))
        # lpmv(m, n, x) computes P_n^m(x) with Condon-Shortley phase
        P_lm = lpmv(abs_m, 10, C)
        
        if m == 0:
            Y[m + K] = N_m * P_lm
        elif m > 0:
            Y[m + K] = np.sqrt(2.0) * N_m * P_lm * np.cos(m * P)
        else: # m < 0
            Y[m + K] = np.sqrt(2.0) * N_m * P_lm * np.sin(abs_m * P)
            
    return Y

# Global variables for speed
C_grid, P_grid, W_grid = _setup_quadrature()
Y_harmonics = _compute_harmonics(C_grid, P_grid)
# Flatten Y_harmonics to shape (21, N_points)
Y_flat = Y_harmonics.reshape(N_M, -1)
# Flatten weights to shape (N_points,)
W_flat = W_grid.reshape(-1)

def _objective_and_grad(coeffs):
    """
    Computes J(c) = ||f||_4^4 / ||f||_2^4 and its gradient.
    coeffs: shape (21,)
    """
    # f at each quadrature point: shape (N_points,)
    # f = Y_flat @ coeffs
    f_vals = Y_flat @ coeffs
    
    # S_2 = sum(w * f^2)
    S_2 = np.sum(W_flat * f_vals**2)
    
    # Avoid division by zero
    if S_2 < 1e-30:
        return 0.0, np.zeros_like(coeffs)
    
    # S_4 = sum(w * f^4)
    S_4 = np.sum(W_flat * f_vals**4)
    
    J = S_4 / (S_2**2)
    
    # Gradient calculation
    # dJ/dc_m = (1/S_2^2) * dS_4/dc_m - (2*S_4/S_2^3) * dS_2/dc_m
    # dS_2/dc_m = 2 * sum(w * f * Y_m)
    # dS_4/dc_m = 4 * sum(w * f^3 * Y_m)
    
    grad_S_2 = 2.0 * (Y_flat.T @ (W_flat * f_vals))
    grad_S_4 = 4.0 * (Y_flat.T @ (W_flat * f_vals**3))
    
    grad_J = (grad_S_4 / (S_2**2)) - (2.0 * S_4 / (S_2**3)) * grad_S_2
    
    return J, grad_J

def _local_optimize(x0, deadline):
    """
    Refines x0 using L-BFGS-B with analytic gradient.
    Returns best x and best J found.
    """
    best_x = x0.copy()
    best_J, _ = _objective_and_grad(x0)
    
    try:
        # Use a wrapper that checks time
        # L-BFGS-B supports jac
        res = minimize(
            _objective_and_grad, 
            x0, 
            method='L-BFGS-B', 
            jac=True, # Our function returns (f, grad)
            options={'maxiter': 100, 'ftol': 1e-15, 'gtol': 1e-15, 'maxls': 50}
        )
        if res.success or res.status == 0:
            J_res, _ = _objective_and_grad(res.x)
            if J_res < best_J:
                best_J = J_res
                best_x = res.x
    except Exception as e:
        pass
        
    return best_x, best_J

def rudin_problem10() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 600.0 # 600s limit, keep margin
    
    # Initial best is the baseline
    best_coeffs = np.ones(21, dtype=np.float64)
    best_J, _ = _objective_and_grad(best_coeffs)
    
    # Define bounds for DE
    # Coefficients shouldn't be too large or small
    # max|c| > 1e-3 and max|c| < 1e5
    # Let's search in a reasonable range, e.g., [-10, 10]
    # Since the objective is homogeneous of degree 0, the scale doesn't matter,
    # but the shape does. DE works better with bounded variables.
    bounds = [(-10.0, 10.0) for _ in range(21)]
    
    # We will run DE in a loop, refining the best candidate each time
    # DE is stochastic, so multiple runs or a long run is good.
    # Given 600s, we can afford a decent number of iterations.
    
    # Strategy:
    # 1. Run DE for a certain number of iterations or time.
    # 2. Take the best result, polish with L-BFGS-B.
    # 3. If time remains, run DE again starting from the polished point or random.
    
    # Let's use a single DE run with a generous maxiter, but monitor time.
    # scipy DE doesn't have a direct time limit, so we estimate iterations per second.
    # Or we can run DE with a fixed number of iterations, say 100, popsize 20.
    # 100 * 20 * 21 evals ~ 42000 evals.
    # Each eval is a matrix-vector product (21x2170) and some sums. ~ 50k flops.
    # 42000 * 50k = 2e9 flops. Should be fast in NumPy (< 1s).
    # So we can run many DE iterations.
    
    # Let's run DE for up to 300 seconds, then polish.
    de_start = time.time()
    
    # We'll use a custom callback to check time? No, DE doesn't support callbacks easily for stopping.
    # We can just run a fixed number of iterations that fits in the time budget.
    # Let's try maxiter=100, popsize=30.
    
    try:
        # Use a random seed for reproducibility if needed, but default is fine
        # tol is the tolerance for the optimization
        de_result = differential_evolution(
            _objective_and_grad, 
            bounds, 
            maxiter=200, 
            popsize=30, 
            tol=1e-10, 
            seed=42,
            polish=False, # We will polish ourselves
            init='sobol' # Better initial distribution
        )
        
        if de_result.success or de_result.fun < best_J:
            # Polish the DE result
            polished_x, polished_J = _local_optimize(de_result.x, deadline)
            if polished_J < best_J:
                best_J = polished_J
                best_coeffs = polished_x
    except Exception as e:
        print(f"DE failed: {e}")
        
    # Check time
    current_time = time.time()
    if current_time < deadline - 30.0:
        # If we have time left, do a few more local optimizations from random starts
        # or restart DE with smaller budget?
        # Let's try a few random local optimizations to catch other basins.
        for _ in range(5):
            if time.time() > deadline - 30.0:
                break
            x_rand = np.random.uniform(-5, 5, 21)
            # Normalize to avoid numerical issues, though objective is scale invariant
            # But L-BFGS might struggle if scale is bad.
            # Let's just try it.
            polished_x, polished_J = _local_optimize(x_rand, deadline)
            if polished_J < best_J:
                best_J = polished_J
                best_coeffs = polished_x
                
    # Final validation
    if not np.all(np.isfinite(best_coeffs)):
        best_coeffs = np.ones(21, dtype=np.float64)
        
    # Ensure max|c| is within bounds
    max_c = np.max(np.abs(best_coeffs))
    if max_c < 1e-3:
        best_coeffs *= (1e-2 / max_c)
    elif max_c > 1e5:
        best_coeffs *= (1e4 / max_c)
        
    return best_coeffs
