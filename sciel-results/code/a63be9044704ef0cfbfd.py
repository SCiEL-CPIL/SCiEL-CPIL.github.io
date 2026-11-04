import numpy as np
import time
import scipy.optimize

N = 400
SAFETY = 12.0

def _autoconv_lin(x):
    """Compute linear autoconvolution using FFT."""
    n = len(x)
    size = 1
    while size < 2 * n - 1:
        size *= 2
    fx = np.fft.rfft(x, size)
    return np.fft.irfft(fx * fx, size)[:2 * n - 1]

def _metric_std(x):
    """Standard metric: max |c_k| / S^2 including k=0."""
    s = np.sum(x)
    if abs(s) < 1e-12:
        return float('inf')
    c = _autoconv_lin(x)
    return float(np.max(np.abs(c))) / (s * s)

def _smooth_metric(x, temp=1.0):
    """
    Smooth surrogate for the metric using a soft-max over |c_k|.
    This is differentiable and suitable for L-BFGS-B.
    """
    s = np.sum(x)
    if abs(s) < 1e-12:
        return 1e10
    c = _autoconv_lin(x)
    abs_c = np.abs(c)
    max_c = np.max(abs_c)
    if max_c < 1e-15:
        return 0.0
    # Soft-max: max_c + temp * log(sum(exp((abs_c - max_c)/temp)))
    exps = np.exp((abs_c - max_c) / temp)
    soft_max = max_c + temp * np.log(np.sum(exps))
    return soft_max / (s * s)

def _chirp_sequence(alpha, beta, gamma):
    """
    Generate a real-valued sequence from a quadratic chirp phase.
    x[i] = cos(alpha * i^2 + beta * i + gamma)
    """
    i = np.arange(N)
    phase = alpha * i * i + beta * i + gamma
    return np.cos(phase)

def _multi_sine_sequence(params):
    """
    Generate a sequence as a sum of sines: x[i] = sum_k a_k * sin(2*pi*f_k*i + phi_k)
    params: [a_1, f_1, phi_1, a_2, f_2, phi_2, ...]
    """
    n_sines = len(params) // 3
    i = np.arange(N)
    x = np.zeros(N)
    for k in range(n_sines):
        a = params[3*k]
        f = params[3*k+1]
        phi = params[3*k+2]
        x += a * np.sin(2 * np.pi * f * i / N + phi)
    return x

def _optimize_chirp_params(deadline):
    """
    Optimize a 3-parameter chirp: x[i] = cos(a*i^2 + b*i + c)
    Uses L-BFGS-B on the smooth surrogate.
    """
    best_x = np.ones(N)
    best_val = _metric_std(best_x)
    
    def _obj(p):
        a, b, c = p
        x = _chirp_sequence(a, b, c)
        return _smooth_metric(x, temp=0.5)
    
    rng = np.random.default_rng(123)
    for trial in range(30):
        if time.time() > deadline - 10:
            break
        
        a0 = rng.uniform(-0.2, 0.2)
        b0 = rng.uniform(-0.5, 0.5)
        c0 = rng.uniform(-np.pi, np.pi)
        
        try:
            res = scipy.optimize.minimize(
                _obj, [a0, b0, c0],
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            a_opt, b_opt, c_opt = res.x
            x_opt = _chirp_sequence(a_opt, b_opt, c_opt)
            val = _metric_std(x_opt)
            if val < best_val:
                best_val = val
                best_x = x_opt.copy()
        except Exception:
            pass
    
    return best_x

def _optimize_multi_sine(deadline, n_sines=4):
    """
    Optimize a sum of sines using differential evolution for global search,
    then L-BFGS-B for local refinement.
    """
    n_params = 3 * n_sines
    best_x = np.ones(N)
    best_val = _metric_std(best_x)
    
    def _obj(params):
        x = _multi_sine_sequence(params)
        return _smooth_metric(x, temp=0.5)
    
    # Differential evolution for global search
    rng = np.random.default_rng(456)
    bounds = []
    for k in range(n_sines):
        bounds.append((0.1, 2.0))  # amplitude
        bounds.append((0.1, N/2))  # frequency
        bounds.append((-np.pi, np.pi))  # phase
    
    try:
        # Use DE for a few iterations
        de_result = scipy.optimize.differential_evolution(
            _obj, bounds,
            maxiter=50,
            popsize=10,
            tol=1e-6,
            seed=42,
            polish=True
        )
        x_opt = _multi_sine_sequence(de_result.x)
        val = _metric_std(x_opt)
        if val < best_val:
            best_val = val
            best_x = x_opt.copy()
    except Exception:
        pass
    
    # Local refinement with L-BFGS-B
    if time.time() < deadline - 5:
        try:
            res = scipy.optimize.minimize(
                _obj, de_result.x,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            x_opt = _multi_sine_sequence(res.x)
            val = _metric_std(x_opt)
            if val < best_val:
                best_val = val
                best_x = x_opt.copy()
        except Exception:
            pass
    
    return best_x

def _spectral_flattening_iterative(x, max_iters=50, deadline=None):
    """
    Iterative spectral flattening: repeatedly modify spectrum to reduce peak.
    """
    x = x.copy().astype(float)
    s = np.sum(x)
    if abs(s) < 1e-12:
        return x
    
    n = len(x)
    size = 1
    while size < 2 * n - 1:
        size *= 2
    
    best_x = x.copy()
    best_val = _metric_std(x)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        fx = np.fft.rfft(x, size)
        mag = np.abs(fx)
        phase = np.angle(fx)
        
        # Identify peak frequencies and suppress them
        peak_threshold = 0.85 * np.max(mag)
        peak_mask = mag > peak_threshold
        
        # Reduce peak magnitudes, increase trough magnitudes
        target = np.mean(mag)
        new_mag = mag.copy()
        new_mag[peak_mask] = (1 - 0.4) * mag[peak_mask] + 0.4 * target
        trough_mask = mag < 0.6 * target
        new_mag[trough_mask] = (1 - 0.3) * mag[trough_mask] + 0.3 * target
        
        # Ensure non-negative
        new_mag = np.maximum(new_mag, 0.01 * np.max(mag))
        
        fx_new = new_mag * np.exp(1j * phase)
        x_new = np.fft.irfft(fx_new, size)[:n]
        
        # Preserve sum
        s_new = np.sum(x_new)
        if abs(s_new) > 1e-12:
            x_new = x_new * (s / s_new)
        
        val = _metric_std(x_new)
        if val < best_val:
            best_val = val
            best_x = x_new.copy()
        
        x = x_new
    
    return best_x

def _gradient_descent_polish(x, max_iters=200, deadline=None):
    """
    Gradient descent on the smooth surrogate for max|c_k|/S^2.
    Uses analytic gradient of the soft-max surrogate.
    """
    x = x.copy().astype(float)
    s = np.sum(x)
    if abs(s) < 1e-12:
        return x
    
    n = len(x)
    size = 1
    while size < 2 * n - 1:
        size *= 2
    
    beta = 50.0  # temperature for soft-max
    lr = 0.005
    best_x = x.copy()
    best_val = _metric_std(x)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute autoconvolution
        fx = np.fft.rfft(x, size)
        c = np.fft.irfft(fx * fx, size)[:2 * n - 1]
        abs_c = np.abs(c)
        
        s = np.sum(x)
        if abs(s) < 1e-12:
            break
        
        # Soft-max of |c_k| / s^2
        norm_c = abs_c / (s * s)
        m = np.max(norm_c)
        if m < 1e-15:
            break
        
        # Soft-max value and weights
        exps = np.exp(beta * (norm_c - m))
        weights = exps / np.sum(exps)
        
        # Gradient w.r.t. x using FFT-based convolution
        # grad_x[i] = sum_k weights_k * (x[i-k] + x[i+k]) / s^2
        # This can be computed efficiently using FFT
        w = np.zeros(2 * n - 1)
        w[:] = weights
        # Symmetrize: w[k] contributes to both x[i-k] and x[i+k]
        # So the gradient is the convolution of x with w (symmetric)
        w_sym = np.zeros(size)
        w_sym[:2*n-1] = w
        # Make it symmetric for convolution
        w_sym_fft = np.fft.rfft(w_sym, size)
        grad_x = np.fft.irfft(fx * w_sym_fft, size)[:n]
        
        # Account for the 1/s^2 factor
        grad_x = grad_x / (s * s)
        
        # Normalize and step
        grad_norm = np.linalg.norm(grad_x)
        if grad_norm < 1e-15:
            break
        
        # Adaptive learning rate
        x_new = x - lr * grad_x
        
        # Preserve sum
        s_new = np.sum(x_new)
        if abs(s_new) > 1e-12:
            x_new = x_new * (s / s_new)
        
        val = _metric_std(x_new)
        if val < best_val:
            best_val = val
            best_x = x_new.copy()
            lr *= 1.02  # increase step slightly on improvement
        else:
            lr *= 0.9  # decrease on failure
        
        x = x_new
    
    return best_x

def construct():
    deadline = time.time() + 150.0 - SAFETY  # Safety margin from 150s limit
    
    # Start with constant sequence
    best_x = np.ones(N)
    best_val = _metric_std(best_x)
    
    # Phase 1: Chirp optimization (fast, structural)
    if time.time() < deadline - 60:
        try:
            x_chirp = _optimize_chirp_params(deadline - 50)
            val = _metric_std(x_chirp)
            if val < best_val:
                best_val = val
                best_x = x_chirp.copy()
        except Exception:
            pass
    
    # Phase 2: Multi-sine optimization with differential evolution
    # This reduces dimensionality to ~12-16 params, enabling global search
    if time.time() < deadline - 30:
        try:
            x_ms = _optimize_multi_sine(deadline - 20, n_sines=4)
            val = _metric_std(x_ms)
            if val < best_val:
                best_val = val
                best_x = x_ms.copy()
        except Exception:
            pass
    
    # Phase 3: Try more sines if time allows
    if time.time() < deadline - 20:
        try:
            x_ms5 = _optimize_multi_sine(deadline - 15, n_sines=5)
            val = _metric_std(x_ms5)
            if val < best_val:
                best_val = val
                best_x = x_ms5.copy()
        except Exception:
            pass
    
    # Phase 4: Spectral flattening refinement
    if time.time() < deadline - 15:
        try:
            x_flat = _spectral_flattening_iterative(best_x, max_iters=40, deadline=deadline - 10)
            val = _metric_std(x_flat)
            if val < best_val:
                best_val = val
                best_x = x_flat.copy()
        except Exception:
            pass
    
    # Phase 5: Gradient descent polish
    if time.time() < deadline - 10:
        try:
            x_grad = _gradient_descent_polish(best_x, max_iters=150, deadline=deadline - 5)
            val = _metric_std(x_grad)
            if val < best_val:
                best_val = val
                best_x = x_grad.copy()
        except Exception:
            pass
    
    # Final: Quick spectral flattening on best
    if time.time() < deadline - 5:
        try:
            x_final = _spectral_flattening_iterative(best_x, max_iters=20, deadline=deadline - 2)
            val = _metric_std(x_final)
            if val < best_val:
                best_val = val
                best_x = x_final.copy()
        except Exception:
            pass
    
    return best_x
