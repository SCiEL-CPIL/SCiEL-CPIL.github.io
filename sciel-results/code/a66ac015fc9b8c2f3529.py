"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.fft import rfft, irfft

N = 400

# EVOLVE-BLOCK-START
def _autoconvolve_max_abs(signs):
    """Compute max absolute value of autoconvolution of the step function.
    
    The step function has N intervals of equal width on [-1/4, 1/4].
    Each interval has height signs[i] (should be +/-1 for signed step).
    The autoconvolution at offset k is sum_i signs[i] * signs[i-k] * width^2
    (for the appropriate overlap).
    
    Actually, let's think about this more carefully.
    
    The function f(x) is a step function with N equal intervals on [-1/4, 1/4].
    Each interval has width w = (1/2)/N = 1/(2N).
    The height on interval i is s_i (signed).
    
    The autoconvolution (f * f)(t) = integral f(x) f(x-t) dx.
    
    For a step function, this is a piecewise linear function.
    We want to bound C3, which likely involves the L-infinity norm of the autoconvolution
    or some related quantity.
    
    The "upper bound on C3" - let me think about what C3 is.
    
    In the context of autocorrelation inequalities, C3 might be related to the
    third moment or a specific constant in an inequality.
    
    Given the task says "improve an autoconvolution upper bound" and the score is
    the upper bound on C3 (lower is better), I'll interpret C3 as the maximum
    absolute value of the autoconvolution (f*f)(t) over t, normalized appropriately.
    
    Let's compute (f*f)(t) for the step function.
    
    For a step function with values s_0, s_1, ..., s_{N-1} on intervals of width w,
    the autoconvolution at lag k*w (for integer k) is:
    (f*f)(k*w) = w^2 * sum_{i} s_i * s_{i-k}
    
    where s_j = 0 if j < 0 or j >= N.
    
    The maximum of |f*f| over all t occurs at some t, and for step functions it's
    piecewise linear, so the extrema are at the breakpoints, which are at multiples of w.
    
    So C3 might be max_k |w^2 * sum_i s_i * s_{i-k}| = w^2 * max_k |sum_i s_i s_{i-k}|
    
    Or it could be the unnormalized sum. Let me just compute the max abs of the
    autocorrelation sum and use that as the metric.
    
    Actually, let me re-read: "score is the upper bound on C3" and "lower is better".
    The seed returns np.ones(N), which gives max autocorrelation sum = N (at lag 0).
    So the score for the seed would be related to N.
    
    Let me just define the score as max_k |sum_i s_i s_{i-k}| and minimize it.
    """
    # Compute autocorrelation via FFT
    n = len(signs)
    # Zero-pad to avoid circular convolution
    size = 2 * n
    f = np.zeros(size)
    f[:n] = signs
    F = rfft(f)
    corr = irfft(F * np.conj(F), size)
    # The autocorrelation at lag k is corr[k] for k >= 0
    # We want max absolute value over all lags
    return np.max(np.abs(corr))


def _autoconvolution_bound(signs):
    """Compute the upper bound on C3 for the given signed step function.
    
    I'll define C3 as the maximum absolute value of the discrete autocorrelation
    sum, which is a common quantity in these types of problems.
    """
    n = len(signs)
    # Compute linear autocorrelation
    size = 2 * n
    f = np.zeros(size)
    f[:n] = signs
    F = rfft(f)
    corr = irfft(F * np.conj(F), size)
    # Maximum absolute autocorrelation
    return np.max(np.abs(corr))


def _greedy_optimize(signs, time_limit=10.0):
    """Greedy optimization: flip signs that reduce the objective."""
    start_time = time.time()
    n = len(signs)
    current_signs = signs.copy()
    current_obj = _autoconvolution_bound(current_signs)
    
    # Precompute autocorrelation
    size = 2 * n
    f = np.zeros(size)
    f[:n] = current_signs
    F = rfft(f)
    corr = irfft(F * np.conj(F), size)
    
    best_obj = current_obj
    best_signs = current_signs.copy()
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        # Try flipping each sign
        for i in range(n):
            old_val = current_signs[i]
            current_signs[i] = -old_val
            
            # Recompute autocorrelation
            f[:n] = current_signs
            F = rfft(f)
            new_corr = irfft(F * np.conj(F), size)
            new_obj = np.max(np.abs(new_corr))
            
            if new_obj < current_obj:
                current_obj = new_obj
                corr = new_corr
                improved = True
                if current_obj < best_obj:
                    best_obj = current_obj
                    best_signs = current_signs.copy()
            else:
                current_signs[i] = old_val
    
    return best_signs, best_obj


def _simulated_annealing(signs, time_limit=20.0):
    """Simulated annealing to optimize the sign pattern."""
    start_time = time.time()
    n = len(signs)
    current_signs = signs.copy()
    current_obj = _autoconvolution_bound(current_signs)
    
    best_signs = current_signs.copy()
    best_obj = current_obj
    
    T = 10.0
    T_min = 0.01
    cooling = 0.99
    
    size = 2 * n
    f = np.zeros(size)
    f[:n] = current_signs
    F = rfft(f)
    corr = irfft(F * np.conj(F), size)
    
    while T > T_min and (time.time() - start_time) < time_limit:
        # Pick a random position to flip
        i = np.random.randint(n)
        old_val = current_signs[i]
        current_signs[i] = -old_val
        
        f[:n] = current_signs
        F = rfft(f)
        new_corr = irfft(F * np.conj(F), size)
        new_obj = np.max(np.abs(new_corr))
        
        delta = new_obj - current_obj
        if delta < 0 or np.random.rand() < np.exp(-delta / T):
            current_obj = new_obj
            corr = new_corr
            if current_obj < best_obj:
                best_obj = current_obj
                best_signs = current_signs.copy()
        else:
            current_signs[i] = old_val
        
        T *= cooling
    
    return best_signs, best_obj


def _projected_gradient(signs, time_limit=30.0):
    """Projected gradient descent on the continuous relaxation.
    
    We relax s_i in [-1, 1] and minimize the max autocorrelation.
    Use subgradient method.
    """
    start_time = time.time()
    n = len(signs)
    x = signs.astype(float).copy()
    best_x = x.copy()
    best_obj = _autoconvolution_bound(np.sign(x))
    
    size = 2 * n
    f = np.zeros(size)
    f[:n] = x
    
    lr = 0.1
    momentum = 0.9
    v = np.zeros(n)
    
    while (time.time() - start_time) < time_limit:
        # Compute autocorrelation
        f[:n] = x
        F = rfft(f)
        corr = irfft(F * np.conj(F), size)
        
        # Find the lag with max absolute value
        max_idx = np.argmax(np.abs(corr))
        max_val = corr[max_idx]
        
        # Subgradient of max_k |corr[k]| with respect to x
        # corr[k] = sum_i x_i x_{i-k} (with appropriate indexing)
        # d/dx_j corr[k] = x_{j-k} + x_{j+k} (approximately)
        
        # For the linear autocorrelation:
        # corr[k] = sum_{i} x[i] * x[i-k] for i-k in [0, n-1]
        # = sum_{i=k}^{n-1} x[i] * x[i-k]
        
        # d/dx_j corr[k] = x[j-k] (if j-k in [0,n-1]) + x[j+k] (if j+k in [0,n-1])
        
        subgrad = np.zeros(n)
        if max_val > 0:
            # Gradient of corr[k]
            k = max_idx
            for j in range(n):
                if j - k >= 0 and j - k < n:
                    subgrad[j] += x[j - k]
                if j + k >= 0 and j + k < n:
                    subgrad[j] += x[j + k]
        else:
            # Gradient of -corr[k]
            k = max_idx
            for j in range(n):
                if j - k >= 0 and j - k < n:
                    subgrad[j] -= x[j - k]
                if j + k >= 0 and j + k < n:
                    subgrad[j] -= x[j + k]
        
        # Update with momentum
        v = momentum * v - lr * subgrad
        x = x + v
        
        # Project to [-1, 1]
        x = np.clip(x, -1, 1)
        
        # Evaluate discrete version
        discrete_x = np.sign(x)
        obj = _autoconvolution_bound(discrete_x)
        if obj < best_obj:
            best_obj = obj
            best_x = discrete_x.copy()
        
        # Adaptive learning rate
        lr *= 0.999
    
    return best_x, best_obj


def construct():
    """Construct a signed step function with N intervals to minimize the
    autoconvolution upper bound C3."""
    start_time = time.time()
    time_limit = 120.0  # Must finish within 120s
    
    # Start with random signs
    np.random.seed(42)
    signs = np.random.choice([-1, 1], size=N)
    
    best_signs = signs.copy()
    best_obj = _autoconvolution_bound(best_signs)
    
    # Greedy optimization
    t1 = time.time()
    signs, obj = _greedy_optimize(signs, time_limit=min(10.0, time_limit - (t1 - start_time) - 5.0))
    if obj < best_obj:
        best_obj = obj
        best_signs = signs.copy()
    
    # Simulated annealing
    t2 = time.time()
    remaining = time_limit - (t2 - start_time) - 5.0
    if remaining > 5.0:
        signs, obj = _simulated_annealing(best_signs, time_limit=min(30.0, remaining))
        if obj < best_obj:
            best_obj = obj
            best_signs = signs.copy()
    
    # Projected gradient on continuous relaxation
    t3 = time.time()
    remaining = time_limit - (t3 - start_time) - 5.0
    if remaining > 5.0:
        signs, obj = _projected_gradient(best_signs, time_limit=min(30.0, remaining))
        if obj < best_obj:
            best_obj = obj
            best_signs = signs.copy()
    
    # Final greedy polish
    t4 = time.time()
    remaining = time_limit - (t4 - start_time) - 2.0
    if remaining > 1.0:
        signs, obj = _greedy_optimize(best_signs, time_limit=min(5.0, remaining))
        if obj < best_obj:
            best_obj = obj
            best_signs = signs.copy()
    
    # Ensure we're within time limit
    if time.time() - start_time > time_limit - 2.0:
        # Return immediately
        return best_signs
    
    return best_signs

# EVOLVE-BLOCK-END
