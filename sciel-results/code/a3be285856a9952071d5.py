"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 50

# EVOLVE-BLOCK-START

def _compute_score(h):
    """
    Compute the objective value:
    ||h||_2^2 / (||h||_1 * ||h * h||_inf)
    
    where h * h is the autoconvolution.
    """
    h = np.asarray(h, dtype=np.float64)
    norm_l2_sq = np.sum(h ** 2)
    norm_l1 = np.sum(np.abs(h))
    
    if norm_l1 < 1e-15:
        return 0.0
    
    # Autoconvolution: h * h
    # For discrete convolution, we can use np.convolve
    # The autoconvolution (h * h) has length 2*N - 1
    conv = np.convolve(h, h, mode='full')
    
    # Supremum norm of autoconvolution
    norm_inf = np.max(np.abs(conv))
    
    if norm_inf < 1e-15:
        return 0.0
    
    return norm_l2_sq / (norm_l1 * norm_inf)


def _objective(h):
    """
    Objective to minimize: negative of the score.
    We want to maximize score, so minimize -score.
    """
    score = _compute_score(h)
    return -score


def _constraint_nonneg(h):
    """
    Constraints: h_i >= 0.
    We'll handle this via bounds in the optimizer.
    """
    pass


def construct():
    """
    Construct 50 nonnegative heights on equal intervals of [-1/4, 1/4]
    to maximize the ratio of squared L2 norm to L1 norm times supremum norm
    of the autoconvolution.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    best_h = np.ones(N)
    best_score = _compute_score(best_h)
    
    # Try multiple random initializations
    np.random.seed(42)
    
    # Different initial patterns
    initializers = []
    
    # Uniform
    initializers.append(np.ones(N))
    
    # Gaussian-like
    x = np.linspace(-1, 1, N)
    initializers.append(np.exp(-x**2 / 0.5))
    initializers.append(np.exp(-x**2 / 0.1))
    initializers.append(np.exp(-x**2 / 1.0))
    
    # Triangular
    initializers.append(1.0 - np.abs(x))
    
    # Cosine bump
    initializers.append(0.5 * (1 + np.cos(np.pi * x)))
    
    # Random smooth
    for _ in range(20):
        # Generate smooth random functions via random Fourier series
        coeffs = np.random.randn(N) * 0.5
        h_rand = 1.0 + coeffs
        h_rand = np.maximum(h_rand, 0.0)
        if np.sum(h_rand) > 0:
            initializers.append(h_rand)
    
    # Random sparse
    for _ in range(10):
        h_rand = np.zeros(N)
        mask = np.random.rand(N) < 0.3
        h_rand[mask] = np.random.rand(np.sum(mask)) + 0.1
        initializers.append(h_rand)
    
    # Random with varying scales
    for scale in [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]:
        x = np.linspace(-1, 1, N)
        initializers.append(np.exp(-x**2 / scale))
    
    bounds = [(0.0, None)] * N
    
    for h0 in initializers:
        if time.time() > deadline:
            break
        
        h0 = np.asarray(h0, dtype=np.float64)
        
        # Normalize to avoid numerical issues
        if np.sum(h0) < 1e-10:
            h0 = np.ones(N)
        
        # Try L-BFGS-B
        try:
            result = minimize(
                _objective,
                h0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            if result.success and result.fun < -best_score:
                best_h = result.x
                best_score = -result.fun
        except Exception:
            pass
        
        # Try Nelder-Mead as well for non-smoothness
        try:
            result = minimize(
                _objective,
                h0,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            # Project to nonnegative
            h_nm = np.maximum(result.x, 0.0)
            score_nm = _compute_score(h_nm)
            if score_nm > best_score:
                best_h = h_nm
                best_score = score_nm
        except Exception:
            pass
    
    # Polish with tighter tolerances
    if time.time() < deadline - 10.0:
        try:
            result = minimize(
                _objective,
                best_h,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if result.success and result.fun < -best_score:
                best_h = result.x
                best_score = -result.fun
        except Exception:
            pass
    
    # Ensure nonnegative
    best_h = np.maximum(best_h, 0.0)
    
    # Final check
    final_score = _compute_score(best_h)
    if final_score > best_score:
        best_score = final_score
    
    return list(best_h)

# EVOLVE-BLOCK-END
