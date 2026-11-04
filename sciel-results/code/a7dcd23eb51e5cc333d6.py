"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize_scalar

_FFT_SIZE = 1
while _FFT_SIZE < 2 * N - 1:
    _FFT_SIZE *= 2

def _autoconv_max_abs(x):
    """Max absolute value of the discrete autoconvolution of x."""
    fx = np.fft.rfft(x, _FFT_SIZE)
    conv = np.fft.irfft(fx * fx, _FFT_SIZE)
    return float(np.max(np.abs(conv[:2 * N - 1])))

def _integral(x):
    """Integral of the equal-width step function = sum of the heights."""
    return float(np.sum(x))

def _metric(x):
    """max |autoconv| / (integral)^2.  Lower is better."""
    s = _integral(x)
    if abs(s) < 1e-12:
        return float('inf')
    return _autoconv_max_abs(x) / (s * s)

def _chirp_sequence(n, alpha, beta, phase_shift=0.0):
    """
    Generate a real-valued chirp sequence.
    x_k = cos(alpha * k^2 + beta * k + phase_shift)
    """
    k = np.arange(n, dtype=np.float64)
    phase = alpha * k**2 + beta * k + phase_shift
    return np.cos(phase)

def _objective_chirp(params):
    """
    Objective function for chirp parameter optimization.
    params: [alpha, beta, phase_shift]
    """
    alpha, beta, phase_shift = params
    x = _chirp_sequence(N, alpha, beta, phase_shift)
    s = _integral(x)
    if abs(s) < 1e-12:
        return 1e10
    return _autoconv_max_abs(x) / (s * s)

def _refine_chirp(alpha0, beta0, phase0, deadline):
    """
    Refine chirp parameters using local search.
    """
    best_params = np.array([alpha0, beta0, phase0])
    best_val = _objective_chirp(best_params)
    
    # Local search with decreasing step size
    step_sizes = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
    rng = np.random.default_rng(42)
    
    for step_size in step_sizes:
        if time.time() > deadline:
            break
        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False
            for _ in range(100):
                if time.time() > deadline:
                    break
                # Try random perturbation
                delta = rng.normal(0, step_size, size=3)
                new_params = best_params + delta
                # Keep phase in [0, 2pi]
                new_params[2] = new_params[2] % (2 * np.pi)
                val = _objective_chirp(new_params)
                if val < best_val:
                    best_val = val
                    best_params = new_params
                    improved = True
    
    return best_params, best_val

def construct():
    deadline = time.time() + 145.0
    
    # Strategy: Use chirp sequences with optimized quadratic phase
    # Chirp sequences have good auto-correlation properties
    # We optimize alpha (quadratic coefficient), beta (linear coefficient), and phase
    
    best_val = float('inf')
    best_x = None
    
    rng = np.random.default_rng(2024)
    
    # Phase 1: Grid search over alpha, beta, phase
    # alpha controls the chirp rate
    # We search over a range of alpha values
    alpha_range = np.linspace(-2 * np.pi / N, 2 * np.pi / N, 20)
    beta_range = np.linspace(-np.pi, np.pi, 10)
    phase_range = np.linspace(0, 2 * np.pi, 5)
    
    for alpha in alpha_range:
        if time.time() > deadline - 30:
            break
        for beta in beta_range:
            if time.time() > deadline - 30:
                break
            for phase in phase_range:
                if time.time() > deadline - 30:
                    break
                x = _chirp_sequence(N, alpha, beta, phase)
                val = _metric(x)
                if val < best_val:
                    best_val = val
                    best_x = x.copy()
    
    # Phase 2: Refine the best parameters found
    if best_x is not None and time.time() < deadline - 10:
        # Find the parameters that gave the best value
        # We need to recover alpha, beta, phase from best_x
        # This is tricky, so let's just do a local search in parameter space
        # around the grid points that were close to the best
        
        # For each grid point, check if it's close to the best
        for alpha in alpha_range:
            if time.time() > deadline - 10:
                break
            for beta in beta_range:
                if time.time() > deadline - 10:
                    break
                for phase in phase_range:
                    if time.time() > deadline - 10:
                        break
                    x = _chirp_sequence(N, alpha, beta, phase)
                    val = _metric(x)
                    # If this grid point is close to the best, refine it
                    if val < best_val * 1.5:
                        refined_params, refined_val = _refine_chirp(alpha, beta, phase, deadline - 5)
                        if refined_val < best_val:
                            best_val = refined_val
                            best_x = _chirp_sequence(N, *refined_params)
    
    # Phase 3: Try some random chirp initializations
    if time.time() < deadline - 5:
        for _ in range(50):
            if time.time() > deadline - 5:
                break
            alpha = rng.uniform(-2 * np.pi / N, 2 * np.pi / N)
            beta = rng.uniform(-np.pi, np.pi)
            phase = rng.uniform(0, 2 * np.pi)
            x = _chirp_sequence(N, alpha, beta, phase)
            val = _metric(x)
            if val < best_val:
                best_val = val
                best_x = x.copy()
            # Refine this random initialization
            if time.time() < deadline - 2:
                refined_params, refined_val = _refine_chirp(alpha, beta, phase, deadline - 2)
                if refined_val < best_val:
                    best_val = refined_val
                    best_x = _chirp_sequence(N, *refined_params)
    
    if best_x is None:
        # Fallback: all ones
        best_x = np.ones(N)
    
    return best_x
# EVOLVE-BLOCK-END
