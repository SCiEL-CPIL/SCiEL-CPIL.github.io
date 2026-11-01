"""Rewritten program for third autocorrelation inequality optimization."""
import numpy as np
from scipy.optimize import minimize
import time

N = 400
DEADLINE = 130  # seconds, safety margin before 150s kill


def _autoconvolve_fft(h):
    """Compute autoconvolution of step function h using FFT."""
    # Pad to avoid circular convolution
    n = len(h)
    nfft = 2 * n
    H = np.fft.rfft(h, n=nfft)
    conv = np.fft.irfft(H * H, n=nfft)
    # The linear autoconvolution of length 2n-1 is in conv[0:2n-1]
    # But for a function on [-1/4, 1/4], the relevant part is centered
    # For step function with N intervals, autoconv at lag k (k = -N+1...N-1)
    # The full linear convolution has 2N-1 points
    return conv[:2 * n - 1]


def _compute_c3(h):
    """Compute the upper bound C3 for the given step heights."""
    # h is the array of step heights
    # Autoconvolution: (f*f)(t) for the piecewise constant function
    # The C3 upper bound is typically the max of |autoconv| or related
    # For the third autocorrelation inequality, C3 = max_t |∫f(τ)f(t-τ)dτ|
    # normalized appropriately
    
    # Using FFT for efficient computation
    n = len(h)
    nfft = 2 * n
    H = np.fft.rfft(h, n=nfft)
    conv = np.fft.irfft(H * H, n=nfft)
    
    # The linear autoconvolution (non-circular) for the step function
    # conv[k] for k=0..2n-2 gives the autoconv at discrete lags
    # The maximum absolute value gives the L∞ norm of autoconv
    c3 = np.max(np.abs(conv[:2 * n - 1]))
    return c3


def _objective_continuous(x):
    """Objective for continuous optimization: C3 of the step function."""
    h = x  # continuous heights
    return _compute_c3(h)


def _gradient(x):
    """Numerical gradient of C3 with respect to heights."""
    eps = 1e-7
    grad = np.zeros_like(x)
    base = _compute_c3(x)
    for i in range(len(x)):
        x_plus = x.copy()
        x_plus[i] += eps
        grad[i] = (_compute_c3(x_plus) - base) / eps
    return grad


def _refine_discrete(h_continuous):
    """Refine continuous solution to discrete ±1 signs."""
    # Start from the sign pattern of the continuous solution
    h = np.sign(h_continuous)
    # Replace zeros with 1
    h[h == 0] = 1.0
    
    best_h = h.copy()
    best_c3 = _compute_c3(h)
    
    # Local search: flip individual bits
    for _ in range(50):  # limited passes
        improved = False
        for i in range(N):
            h_new = h.copy()
            h_new[i] *= -1
            c3_new = _compute_c3(h_new)
            if c3_new < best_c3 - 1e-12:
                h = h_new
                best_h = h.copy()
                best_c3 = c3_new
                improved = True
        if not improved:
            break
    
    return best_h, best_c3


def _random_discrete_search(n_trials=200):
    """Try random discrete sign patterns."""
    best_h = None
    best_c3 = np.inf
    
    for _ in range(n_trials):
        h = np.random.choice([-1.0, 1.0], size=N)
        c3 = _compute_c3(h)
        if c3 < best_c3:
            best_c3 = c3
            best_h = h.copy()
    
    return best_h, best_c3


def construct():
    """Construct the optimal signed step function. Returns sequence of signed step heights."""
    start_time = time.time()
    
    # Phase 1: Continuous relaxation with L-BFGS-B
    # Initialize with random continuous values in [-1, 1]
    np.random.seed(42)
    
    best_h = np.ones(N)
    best_c3 = _compute_c3(best_h)
    
    # Try multiple initializations for continuous optimization
    n_init = 5
    for init_i in range(n_init):
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 20:
            break
        
        if init_i == 0:
            x0 = np.ones(N)
        elif init_i == 1:
            x0 = -np.ones(N)
        elif init_i == 2:
            x0 = np.random.uniform(-1, 1, size=N)
        elif init_i == 3:
            x0 = np.random.choice([-1.0, 1.0], size=N).astype(float)
        else:
            x0 = np.sin(np.linspace(0, 4 * np.pi, N))
        
        try:
            result = minimize(
                _objective_continuous,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50}
            )
            if result.success or result.fun < best_c3:
                best_c3 = result.fun
                best_h = result.x.copy()
        except Exception:
            pass
    
    elapsed = time.time() - start_time
    if elapsed > DEADLINE - 10:
        # Not enough time for refinement, return best continuous solution rounded
        h_final = np.sign(best_h)
        h_final[h_final == 0] = 1.0
        return h_final
    
    # Phase 2: Discrete refinement from continuous solution
    h_discrete, c3_discrete = _refine_discrete(best_h)
    if c3_discrete < best_c3:
        best_h = h_discrete
        best_c3 = c3_discrete
    
    elapsed = time.time() - start_time
    if elapsed > DEADLINE - 5:
        return best_h
    
    # Phase 3: Random discrete search to potentially find better solutions
    h_rand, c3_rand = _random_discrete_search(n_trials=100)
    if c3_rand < best_c3:
        best_h = h_rand
        best_c3 = c3_rand
    
    # Final local refinement
    h_final, c3_final = _refine_discrete(best_h)
    if c3_final < best_c3:
        best_h = h_final
        best_c3 = c3_final
    
    return best_h
