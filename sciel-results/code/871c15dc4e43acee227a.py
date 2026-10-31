"""
Strategy: Continuous SLSQP Optimization
The task asks for a signed step function. While the seed program explores discrete ±1
sequences (Barker, Legendre, etc.), the objective function (max_t |f*f(t)|) is a
convex function of the vector f. Therefore, we can treat the step heights as
continuous variables in [-1, 1] and use a convex optimization algorithm (SLSQP)
to minimize the peak autoconvolution. This allows the function to take non-binary
values (e.g., 0.5, -0.8) which can significantly reduce the peak compared to
forced binary choices, potentially breaking the 1.0 seed score.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 400
_START = time.time()
_DEADLINE = 135.0  # Safety margin before 150s kill

def _autoconv_bound(f):
    """
    Compute the upper bound: max_t |f*f(t)| for a step function f (length N)
    on [-1/4, 1/4], interval h=1/200.
    """
    h = 1.0 / 200.0
    nfft = 1
    while nfft < 2 * N:
        nfft *= 2
    F = np.zeros(nfft)
    F[:N] = f
    # Linear convolution via FFT
    conv = np.fft.irfft(np.fft.rfft(F) ** 2) * h
    return float(np.max(np.abs(conv)))

def construct():
    # 1. Initialize with a random continuous vector
    # Using a mix of random and deterministic seeds to ensure robustness
    # The seed program uses ±1. We start with ±1 but allow the optimizer to move them.
    # Alternatively, starting with a smooth window (like a Gaussian) might help SLSQP converge faster
    # away from the high-entropy random state. Let's try a few starting points.
    
    best_f = None
    best_val = float('inf')
    
    # Starting points
    # 1. Random binary (like seed)
    rng = np.random.RandomState(42)
    f_start_1 = rng.choice([-1.0, 1.0], size=N)
    
    # 2. Gaussian-like window (smooth start)
    x = np.linspace(-1, 1, N)
    f_start_2 = np.exp(-x**2 * 5)
    f_start_2 = (f_start_2 - f_start_2.mean()) / (f_start_2.std() + 1e-9)
    f_start_2 = np.clip(f_start_2, -1.0, 1.0)
    
    # 3. Sine wave
    f_start_3 = np.sin(2 * np.pi * 2 * np.arange(N) / N)
    
    starts = [f_start_1, f_start_2, f_start_3]
    
    for f0 in starts:
        if time.time() > _DEADLINE - 5:
            break
            
        try:
            # Define the objective for SLSQP
            # SLSQP minimizes f(x). We want to minimize _autoconv_bound.
            # Note: _autoconv_bound involves a max, which is non-smooth but convex.
            # SLSQP can handle this reasonably well, but it might get stuck in local maxima
            # of the non-smooth surface. However, since the function is convex, any local minimum
            # is global. The issue is the gradient of the max.
            # SLSQP uses numerical gradients or analytical if provided.
            # Providing analytical gradient is complex for the max operation.
            # Let's rely on SLSQP's numerical gradient first.
            
            # Constraints: f_i in [-1, 1]
            bounds = [(-1.0, 1.0) for _ in range(N)]
            
            # Optimization
            res = minimize(
                _autoconv_bound, 
                f0, 
                method='SLSQP', 
                bounds=bounds, 
                options={'maxiter': 50, 'ftol': 1e-9}
            )
            
            if res.success or res.fun < best_val:
                # SLSQP might return values slightly outside bounds due to numerical error
                f_opt = np.clip(res.x, -1.0, 1.0)
                val = _autoconv_bound(f_opt)
                if val < best_val:
                    best_val = val
                    best_f = f_opt
        except Exception as e:
            # If optimization fails, fall back to the initial guess
            val = _autoconv_bound(f0)
            if val < best_val:
                best_val = val
                best_f = f0.copy()

    # If optimization didn't find anything better (or failed), return a known good seed
    if best_f is None or best_val > 1.0:
        # Fallback to a simple binary sequence if continuous optimization failed or was worse
        # The seed program's best was ~1.0.
        # A simple alternating sequence has a known bound.
        f_alt = np.where(np.arange(N) % 2 == 0, 1.0, -1.0)
        val_alt = _autoconv_bound(f_alt)
        if val_alt < best_val:
            best_f = f_alt
            best_val = val_alt
        else:
            best_f = f0 # Use the last tried start
    
    return best_f
