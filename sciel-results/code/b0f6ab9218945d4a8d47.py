"""Second autocorrelation inequality: maximize squared L2 norm / (L1 norm * sup norm of autoconvolution)."""
import numpy as np
import time
import sys

N = 50

def _compute_ratio(heights):
    """Compute the objective: (L2 norm)^2 / (L1 norm * sup norm of autoconvolution).
    
    heights: array of N nonnegative values
    Returns the ratio value.
    """
    h = np.asarray(heights, dtype=np.float64)
    if np.any(h < -1e-12):
        return 0.0
    h = np.maximum(h, 0.0)
    
    l1 = np.sum(h)
    if l1 <= 1e-15:
        return 0.0
    
    l2_sq = np.sum(h * h)
    
    # Autoconvolution: (h * h) where * is convolution
    # For a sequence of length N, the autoconvolution has length 2N-1
    autoconv = np.convolve(h, h, mode='full')
    sup_norm = np.max(autoconv)
    
    if sup_norm <= 1e-15:
        return 0.0
    
    return l2_sq / (l1 * sup_norm)


def _smooth_objective(heights, temperature=0.01):
    """Smooth version of the objective for gradient-based optimization.
    
    Uses a soft-max approximation of the supremum norm.
    """
    h = np.asarray(heights, dtype=np.float64)
    h = np.maximum(h, 0.0)
    
    l1 = np.sum(h)
    if l1 <= 1e-15:
        return 0.0
    
    l2_sq = np.sum(h * h)
    
    autoconv = np.convolve(h, h, mode='full')
    # Soft max approximation of sup norm
    log_sum_exp = temperature * np.log(np.sum(np.exp(autoconv / temperature)))
    soft_max = log_sum_exp
    
    if soft_max <= 1e-15:
        return 0.0
    
    return l2_sq / (l1 * soft_max)


def _dft_parameterize(z):
    """Parameterize N nonnegative heights as |DFT(z)|^2 where z is a complex vector of length N.
    
    This ensures non-negativity and provides a smooth manifold.
    """
    # z is complex vector of length N
    # DFT of z gives N complex values, take squared magnitudes
    Z = np.fft.fft(z)
    heights = np.abs(Z) ** 2
    return heights


def _random_initialization(seed):
    """Generate diverse initializations."""
    rng = np.random.RandomState(seed)
    kind = seed % 5
    
    if kind == 0:
        # Gaussian-like
        x = np.linspace(-1, 1, N)
        z = np.exp(-x**2 / 0.3) * (1 + 0.5j * rng.randn(N))
    elif kind == 1:
        # Lorentzian
        x = np.linspace(-1, 1, N)
        z = 1.0 / (1 + 5*x**2) * (1 + 0.5j * rng.randn(N))
    elif kind == 2:
        # Power law
        idx = np.arange(N)
        z = (1.0 / (1 + idx * 0.1)) * (1 + 0.5j * rng.randn(N))
    elif kind == 3:
        # Two-peak
        x = np.linspace(-1, 1, N)
        z = np.exp(-((x-0.4)**2)/0.1) + np.exp(-((x+0.4)**2)/0.1)
        z = z * (1 + 0.5j * rng.randn(N))
    else:
        # Random complex
        z = rng.randn(N) + 1j * rng.randn(N)
    
    return z


def _gradient_descent(z_init, max_iter=5000, lr=0.01, deadline=None):
    """Perform gradient descent on the complex phase/magnitude parameters.
    
    Minimizes the negative of the smooth objective.
    """
    z = z_init.copy()
    best_heights = _dft_parameterize(z)
    best_score = _compute_ratio(best_heights)
    
    for i in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute heights
        heights = _dft_parameterize(z)
        
        # Compute gradient numerically (complex step or finite difference)
        # Use finite difference for simplicity
        eps = 1e-5
        grad_real = np.zeros(N)
        grad_imag = np.zeros(N)
        
        # Objective to maximize: smooth_objective
        obj = _smooth_objective(heights, temperature=0.05)
        
        # For efficiency, use a smaller number of gradient evaluations
        # Central difference on real and imaginary parts
        for j in range(N):
            # Real part
            z_plus = z.copy()
            z_plus[j].real += eps
            h_plus = _dft_parameterize(z_plus)
            obj_plus = _smooth_objective(h_plus, temperature=0.05)
            
            z_minus = z.copy()
            z_minus[j].real -= eps
            h_minus = _dft_parameterize(z_minus)
            obj_minus = _smooth_objective(h_minus, temperature=0.05)
            
            grad_real[j] = (obj_plus - obj_minus) / (2 * eps)
            
            # Imaginary part
            z_plus = z.copy()
            z_plus[j].imag += eps
            h_plus = _dft_parameterize(z_plus)
            obj_plus = _smooth_objective(h_plus, temperature=0.05)
            
            z_minus = z.copy()
            z_minus[j].imag -= eps
            h_minus = _dft_parameterize(z_minus)
            obj_minus = _smooth_objective(h_minus, temperature=0.05)
            
            grad_imag[j] = (obj_plus - obj_minus) / (2 * eps)
        
        # Update: maximize objective, so go in direction of positive gradient
        z.real += lr * grad_real
        z.imag += lr * grad_imag
        
        # Check current score
        current_heights = _dft_parameterize(z)
        current_score = _compute_ratio(current_heights)
        
        if current_score > best_score:
            best_score = current_score
            best_heights = current_heights.copy()
        
        # Adaptive learning rate
        if i % 100 == 0 and i > 0:
            lr *= 0.95
    
    return best_heights, best_score


def _direct_optimization(heights_init, max_iter=3000, lr=0.005, deadline=None):
    """Direct optimization on the height space with projection to nonnegative."""
    h = np.maximum(np.asarray(heights_init, dtype=np.float64), 0.0).copy()
    best_h = h.copy()
    best_score = _compute_ratio(best_h)
    
    for i in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute gradient numerically
        eps = 1e-6
        grad = np.zeros(N)
        obj = _smooth_objective(h, temperature=0.05)
        
        for j in range(N):
            h_plus = h.copy()
            h_plus[j] += eps
            obj_plus = _smooth_objective(h_plus, temperature=0.05)
            
            h_minus = h.copy()
            h_minus[j] = max(h[j] - eps, 0.0)
            obj_minus = _smooth_objective(h_minus, temperature=0.05)
            
            grad[j] = (obj_plus - obj_minus) / (2 * eps)
        
        # Update with projection
        h = h + lr * grad
        h = np.maximum(h, 0.0)
        
        current_score = _compute_ratio(h)
        if current_score > best_score:
            best_score = current_score
            best_h = h.copy()
        
        if i % 100 == 0 and i > 0:
            lr *= 0.95
    
    return best_h, best_score


def construct():
    """Construct 50 nonnegative heights to maximize the ratio.
    
    Returns a list of N nonnegative step heights.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Start with uniform (the seed)
    best_heights = np.ones(N)
    best_score = _compute_ratio(best_heights)
    
    # Try DFT parameterization with diverse initializations
    num_trials = 12
    
    for trial in range(num_trials):
        if time.time() > deadline - 10.0:
            break
        
        seed = 42 + trial * 137
        z_init = _random_initialization(seed)
        
        # DFT parameterization approach
        heights_dft, score_dft = _gradient_descent(
            z_init, max_iter=2000, lr=0.005, deadline=deadline
        )
        
        if score_dft > best_score:
            best_score = score_dft
            best_heights = heights_dft
        
        # Also try direct optimization from a few starting points
        if time.time() > deadline - 5.0:
            break
        
        if trial % 2 == 0:
            # Start from power-law
            idx = np.arange(N)
            h_init = 1.0 / (1 + idx * 0.05)
            h_direct, score_direct = _direct_optimization(
                h_init, max_iter=2000, lr=0.003, deadline=deadline
            )
            if score_direct > best_score:
                best_score = score_direct
                best_heights = h_direct
        
        if trial % 3 == 0:
            # Start from Gaussian
            x = np.linspace(-1, 1, N)
            h_init = np.exp(-x**2 / 0.5)
            h_direct, score_direct = _direct_optimization(
                h_init, max_iter=2000, lr=0.003, deadline=deadline
            )
            if score_direct > best_score:
                best_score = score_direct
                best_heights = h_direct
    
    # Final polish: try a few more direct optimizations from the best found
    if time.time() < deadline - 2.0:
        h_polish, score_polish = _direct_optimization(
            best_heights, max_iter=3000, lr=0.001, deadline=deadline
        )
        if score_polish > best_score:
            best_score = score_polish
            best_heights = h_polish
    
    # Ensure nonnegative and return as list
    best_heights = np.maximum(best_heights, 0.0)
    
    # Scale doesn't matter for the ratio, but ensure valid
    if np.sum(best_heights) <= 1e-15:
        best_heights = np.ones(N)
    
    return [float(x) for x in best_heights]
