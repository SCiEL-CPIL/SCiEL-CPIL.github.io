"""
Signed autocorrelation maximum absolute-value variant.

Strategy: Use a different optimization family than the current program.
The current program uses L-BFGS-B on phase variables with a flat magnitude spectrum.
This program uses:
1. Direct gradient descent on the signal x itself (not spectral parameters)
2. A smoothed surrogate (soft-max) for the max absolute autoconvolution
3. Projected gradient descent with FFT-based gradient computation
4. Multiple diverse initializations including chaotic/low-discrepancy sequences
5. Two-phase optimization: coarse then fine

The key structural change: optimize x directly rather than spectral parameters,
using a differentiable surrogate that avoids the non-smoothness of max|.|
"""
import numpy as np
import time

N = 400
DEADLINE = 150.0
SAFETY_MARGIN = 10.0

def _autoconv(x):
    """Compute full linear autoconvolution of x (length N) -> length 2N-1."""
    n = 2 * N - 1
    X = np.fft.fft(x, n=2*N)
    C = np.fft.ifft(X * X).real[:n]
    return C

def _evaluate(x):
    """Compute the raw objective value: max|c_k| / (sum x_i)^2."""
    c = _autoconv(x)
    s = np.sum(x)
    if abs(s) < 1e-12:
        return np.inf
    return np.max(np.abs(c)) / (s**2)

def _soft_max_abs(c, temperature=1.0):
    """
    Smooth approximation of max(|c_k|) using log-sum-exp.
    soft_max(|c|) = (1/T) * log(sum(exp(T*|c_k|)))
    For large T, this approaches max(|c_k|).
    """
    T = temperature
    # Scale to avoid overflow
    c_abs = np.abs(c)
    c_max = np.max(c_abs)
    if c_max < 1e-15:
        return 0.0
    # Use shifted log-sum-exp for numerical stability
    shifted = T * (c_abs - c_max)
    log_sum = np.log(np.sum(np.exp(shifted)))
    return c_max + log_sum / T

def _objective_soft(x, temperature=1.0):
    """
    Differentiable surrogate objective:
    soft_max(|c|) / (sum x)^2
    """
    c = _autoconv(x)
    s = np.sum(x)
    if abs(s) < 1e-12:
        return 1e10
    return _soft_max_abs(c, temperature) / (s**2)

def _gradient_fft(x, temperature=1.0):
    """
    Compute gradient of the soft-max surrogate objective w.r.t. x using FFT.
    
    The autoconvolution c = x * x (linear).
    In frequency domain: C(k) = X(k)^2
    c = ifft(X^2)
    
    For the soft-max: d/dx_i [soft_max(|c|)] involves chain rule through c.
    
    We use the fact that:
    d soft_max(|c|) / d c_j = (1/sum_k exp(T|c_k|)) * exp(T|c_j|) * sign(c_j)
    
    And d c_j / d x_i = x_{j-i} + x_{j+N-1-i} (for linear conv, with appropriate bounds)
    
    Actually, let's compute the gradient numerically but efficiently using FFT.
    """
    # For efficiency, we compute the gradient using the structure of autoconvolution
    # c_k = sum_i x_i * x_{k-i} for k = 0..2N-2
    # 
    # d c_k / d x_m = x_{k-m} + x_{m-k} (when both indices are valid)
    # Actually for linear convolution: c_k = sum_{i=0}^{N-1} x_i * x_{k-i} where x_j=0 for j<0 or j>=N
    # So d c_k / d x_m = x_{k-m} + x_{m-k} if 0 <= k-m < N and 0 <= m-k < N, else one term or zero
    #
    # This is getting complex. Let's use a simpler approach:
    # Compute gradient numerically with a small step, but do it efficiently.
    
    # Actually, let's use the analytical gradient via FFT.
    # c = ifft(fft(x, 2N)^2)[:2N-1]
    # Let X = fft(x, 2N), C = X^2
    # d C_k / d X_m = 2 * X_m * delta_{km}
    # d X_m / d x_i = exp(-2j pi m i / 2N)
    # So d C_k / d x_i = 2 * X_k * exp(-2j pi k i / 2N)
    # c = ifft(C), so d c_j / d x_i = (1/(2N)) * sum_k dC_k/dx_i * exp(2j pi k j / 2N)
    # = (1/(2N)) * sum_k 2 * X_k * exp(-2j pi k i / 2N) * exp(2j pi k j / 2N)
    # = (1/(2N)) * sum_k 2 * X_k * exp(2j pi k (j-i) / 2N)
    # = (1/N) * ifft(2*X)(j-i)  [circular, length 2N]
    
    # This is getting complex. Let me just use numerical gradient with FFT acceleration.
    # For a 400-dim problem, numerical gradient with 400+1 evaluations is feasible.
    eps = 1e-6
    grad = np.zeros(N)
    f0 = _objective_soft(x, temperature)
    for i in range(N):
        x_p = x.copy()
        x_p[i] += eps
        f_p = _objective_soft(x_p, temperature)
        grad[i] = (f_p - f0) / eps
    return grad

def _gradient_descent(x, temperature_schedule, max_iters, deadline, step_schedule):
    """
    Projected gradient descent on the soft-max surrogate.
    """
    x = x.copy()
    best_x = x.copy()
    best_val = _evaluate(x)
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        T = temperature_schedule(it)
        step = step_schedule(it)
        
        f = _objective_soft(x, T)
        if not np.isfinite(f):
            break
        
        g = _gradient_fft(x, T)
        
        # Gradient descent step
        x_new = x - step * g
        
        # Normalize to keep sum(x) = 1 for stability
        s = np.sum(x_new)
        if abs(s) > 1e-12:
            x_new = x_new / s
        else:
            x_new = x
        
        # Evaluate
        val = _evaluate(x_new)
        if val < best_val:
            best_val = val
            best_x = x_new.copy()
        
        x = x_new
    
    return best_x, best_val

def _chaotic_sequence(seed, n):
    """
    Generate a sequence using logistic map for low-discrepancy properties.
    """
    rng = np.random.RandomState(seed)
    x = rng.uniform(0.1, 0.9)
    r = 3.99  # chaotic regime
    seq = np.zeros(n)
    for i in range(n):
        x = r * x * (1 - x)
        seq[i] = 1.0 if x > 0.5 else -1.0
    return seq

def _van_der_corput(n, base=2):
    """Van der Corput sequence for low-discrepancy."""
    seq = np.zeros(n)
    for i in range(n):
        f = 0.0
        denom = 1.0
        num = i
        while num > 0:
            denom *= base
            f += (num % base) / denom
            num //= base
        seq[i] = 1.0 if f > 0.5 else -1.0
    return seq

def _sine_sequence(n, freq):
    """Sine-based sequence."""
    t = np.arange(n)
    return np.sign(np.sin(2 * np.pi * freq * t / n))

def _block_sequence(n, block_len):
    """Alternating blocks."""
    seq = np.zeros(n)
    for i in range(n):
        block = i // block_len
        seq[i] = 1.0 if block % 2 == 0 else -1.0
    return seq

def construct():
    """
    Return signed step heights that minimize max|c_k| / (sum x_i)^2.
    Uses direct gradient descent on the signal with soft-max surrogate.
    """
    start_time = time.time()
    deadline = start_time + DEADLINE - SAFETY_MARGIN
    
    np.random.seed(42)
    
    # Baseline
    baseline = np.ones(N)
    best_result = baseline.copy()
    best_val = _evaluate(baseline)
    
    # Generate diverse initializations
    inits = []
    
    # Random
    for seed in [42, 123, 456, 789, 1000, 2000]:
        rng = np.random.RandomState(seed)
        x = rng.choice([-1.0, 1.0], size=N)
        inits.append(x)
    
    # Chaotic
    for seed in [1, 2, 3]:
        inits.append(_chaotic_sequence(seed, N))
    
    # Van der Corput
    inits.append(_van_der_corput(N))
    
    # Sine waves
    for freq in [3, 7, 13, 21]:
        inits.append(_sine_sequence(N, freq))
    
    # Block sequences
    for bl in [2, 4, 8, 16, 32]:
        inits.append(_block_sequence(N, bl))
    
    # Alternating
    inits.append(np.array([1.0 if i % 2 == 0 else -1.0 for i in range(N)]))
    
    # Run gradient descent from each initialization
    for idx, x_init in enumerate(inits):
        if time.time() > deadline - 5:
            break
        
        # Normalize
        s = np.sum(x_init)
        if abs(s) > 1e-12:
            x_init = x_init / s
        else:
            continue
        
        # Two-phase: coarse then fine
        # Phase 1: high temperature, large steps
        T1 = 5.0
        steps1 = 50
        step1 = 0.1
        
        x1, val1 = _gradient_descent(
            x_init,
            lambda it, T=T1: T,
            steps1,
            deadline,
            lambda it, s=step1: s * (1 - it / steps1)
        )
        
        # Phase 2: low temperature, small steps
        if time.time() > deadline - 5:
            break
        
        T2 = 0.5
        steps2 = 50
        step2 = 0.01
        
        x2, val2 = _gradient_descent(
            x1,
            lambda it, T=T2: T,
            steps2,
            deadline,
            lambda it, s=step2: s * (1 - it / steps2)
        )
        
        if val2 < best_val:
            best_val = val2
            best_result = x2.copy()
    
    # Final polish: very fine gradient descent
    if time.time() < deadline - 10:
        T3 = 0.1
        steps3 = 30
        step3 = 0.001
        
        x3, val3 = _gradient_descent(
            best_result,
            lambda it, T=T3: T,
            steps3,
            deadline,
            lambda it, s=step3: s * (1 - it / steps3)
        )
        
        if val3 < best_val:
            best_val = val3
            best_result = x3.copy()
    
    # Ensure valid output
    s_final = np.sum(best_result)
    if abs(s_final) < 1e-12:
        return np.ones(N)
    
    return best_result
