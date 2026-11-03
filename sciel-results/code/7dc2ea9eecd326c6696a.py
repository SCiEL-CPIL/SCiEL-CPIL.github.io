import numpy as np
import time

N = 400
START_TIME = time.time()
TIME_LIMIT = 145
SAFETY_MARGIN = 5

def _check_deadline():
    return (time.time() - START_TIME) > (TIME_LIMIT - SAFETY_MARGIN)

def _autoconv_fft(x):
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    X = np.fft.rfft(x, n=L_fft)
    C = np.fft.irfft(X * X, n=L_fft)[:L]
    return C

def _objective(x):
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10
    ac = _autoconv_fft(x)
    return np.max(np.abs(ac)) / (s * s)

def _spectral_phase_optimize(x0, n_iters=500, lr=0.01):
    """
    Optimize phase of FFT(x) to minimize max|autocorrelation|.
    Keep magnitude fixed (or adapt it).
    """
    L_fft = 1
    while L_fft < 2 * N:
        L_fft *= 2
    
    X = np.fft.rfft(x0, n=L_fft)
    X_mag = np.abs(X).copy()
    X_phase = np.angle(X).copy()
    
    s_target = np.sum(x0)
    best_x = x0.copy()
    best_val = _objective(x0)
    
    for it in range(n_iters):
        if _check_deadline():
            break
        
        # Reconstruct x from current phase and magnitude
        X_cur = X_mag * np.exp(1j * X_phase)
        X_full = np.zeros(L_fft, dtype=complex)
        X_full[:L_fft // 2 + 1] = X_cur
        for k in range(1, L_fft // 2):
            X_full[L_fft - k] = np.conj(X_full[k])
        x = np.fft.irfft(X_full, n=L_fft)[:N]
        
        s = np.sum(x)
        if abs(s) < 1e-10:
            break
        
        ac = _autoconv_fft(x)
        abs_ac = np.abs(ac)
        max_val = np.max(abs_ac)
        
        if max_val < 1e-12:
            break
        
        # Find peak lag
        peak_lag = np.argmax(abs_ac)
        peak_sign = np.sign(ac[peak_lag])
        if peak_sign == 0:
            peak_sign = 1.0
        
        # Compute gradient of ac[peak_lag] w.r.t. x
        grad_x = np.zeros(N)
        for j in range(N):
            g = 0.0
            idx1 = peak_lag - j
            if 0 <= idx1 < N:
                g += x[idx1]
            idx2 = j + peak_lag
            if 0 <= idx2 < N:
                g += x[idx2]
            grad_x[j] = peak_sign * g
        
        # Map gradient to frequency domain
        grad_X = np.fft.rfft(grad_x, n=L_fft)[:L_fft // 2 + 1]
        
        # Phase gradient: d(phase)/d(x) projected
        # grad_phase = Im(grad_X * conj(X)) / |X|
        X_nonzero = X_mag > 1e-10
        grad_phase = np.zeros_like(X_phase)
        grad_phase[X_nonzero] = np.imag(grad_X[X_nonzero] * np.conj(X_cur[X_nonzero])) / X_mag[X_nonzero]
        
        # Update phase
        X_phase -= lr * grad_phase
        
        # Reconstruct and check
        X_new = X_mag * np.exp(1j * X_phase)
        X_full_new = np.zeros(L_fft, dtype=complex)
        X_full_new[:L_fft // 2 + 1] = X_new
        for k in range(1, L_fft // 2):
            X_full_new[L_fft - k] = np.conj(X_full_new[k])
        x_new = np.fft.irfft(X_full_new, n=L_fft)[:N]
        
        s_new = np.sum(x_new)
        if abs(s_new) > 1e-10:
            x_new = x_new * (s_target / s_new)
        
        val_new = _objective(x_new)
        if val_new < best_val:
            best_val = val_new
            best_x = x_new.copy()
            # Adaptive learning rate: decrease if improving
            lr = max(0.001, lr * 0.999)
        else:
            # Revert phase and decrease learning rate
            X_phase += lr * grad_phase
            lr = max(0.001, lr * 0.9)
    
    return best_x, best_val

def _amplitude_shaping(x0, n_iters=200):
    """
    Iteratively reshape the magnitude spectrum to flatten autocorrelation.
    Key idea: if ac[k] is too large, reduce |X| at frequencies that contribute to that lag.
    """
    L_fft = 1
    while L_fft < 2 * N:
        L_fft *= 2
    
    X = np.fft.rfft(x0, n=L_fft)
    X_mag = np.abs(X).copy()
    X_phase = np.angle(X).copy()
    
    s_target = np.sum(x0)
    best_x = x0.copy()
    best_val = _objective(x0)
    
    # Target: make autocorrelation as flat as possible (except at lag 0)
    # Strategy: iteratively suppress the peak lag by reducing |X| at contributing frequencies
    
    for it in range(n_iters):
        if _check_deadline():
            break
        
        # Reconstruct
        X_cur = X_mag * np.exp(1j * X_phase)
        X_full = np.zeros(L_fft, dtype=complex)
        X_full[:L_fft // 2 + 1] = X_cur
        for k in range(1, L_fft // 2):
            X_full[L_fft - k] = np.conj(X_full[k])
        x = np.fft.irfft(X_full, n=L_fft)[:N]
        
        s = np.sum(x)
        if abs(s) < 1e-10:
            break
        
        ac = _autoconv_fft(x)
        abs_ac = np.abs(ac)
        
        # Find top few peaks (excluding lag 0 which is always large)
        abs_ac_nonzero = abs_ac.copy()
        abs_ac_nonzero[0] = 0
        top_k = min(5, len(abs_ac_nonzero))
        peak_indices = np.argsort(abs_ac_nonzero)[-top_k:]
        
        max_val = np.max(abs_ac_nonzero)
        if max_val < 1e-12:
            break
        
        # For each peak, reduce |X| at frequencies that constructively interfere
        for peak_lag in peak_indices:
            if abs_ac[peak_lag] < 1e-12:
                continue
            
            peak_sign = np.sign(ac[peak_lag])
            
            # The contribution to ac[peak_lag] from frequency k is:
            # |X[k]|² * exp(2j*phase[k]) * exp(-2j*pi*k*peak_lag/L)
            # To suppress, reduce |X[k]| where the real part is positive (constructive)
            
            freqs = np.arange(L_fft // 2 + 1)
            phase_contrib = 2 * np.pi * freqs * peak_lag / L_fft
            
            # Real part of X[k]² * exp(-i*phase_contrib)
            X2 = X_cur * X_cur
            contrib_real = np.real(X2 * np.exp(-1j * phase_contrib))
            
            # Reduce |X| where contrib_real > 0
            reduction = np.where(contrib_real > 0, 0.01 * contrib_real / (np.max(np.abs(contrib_real)) + 1e-10), 0)
            X_mag = np.maximum(0.01, X_mag - reduction)
        
        # Renormalize to maintain sum
        X_new = X_mag * np.exp(1j * X_phase)
        X_full_new = np.zeros(L_fft, dtype=complex)
        X_full_new[:L_fft // 2 + 1] = X_new
        for k in range(1, L_fft // 2):
            X_full_new[L_fft - k] = np.conj(X_full_new[k])
        x_new = np.fft.irfft(X_full_new, n=L_fft)[:N]
        
        s_new = np.sum(x_new)
        if abs(s_new) > 1e-10:
            scale = s_target / s_new
            X_mag = X_mag * scale
        
        val_new = _objective(x_new)
        if val_new < best_val:
            best_val = val_new
            best_x = x_new.copy()
    
    return best_x, best_val

def _gradient_descent_smooth(x0, n_iters=300, beta=20.0):
    """
    Gradient descent with smooth max approximation (log-sum-exp).
    Uses FFT-based gradient computation for speed.
    """
    s_target = np.sum(x0)
    best_x = x0.copy()
    best_val = _objective(x0)
    
    x = x0.copy()
    lr = 0.001
    
    for it in range(n_iters):
        if _check_deadline():
            break
        
        s = np.sum(x)
        if abs(s) < 1e-10:
            break
        
        ac = _autoconv_fft(x)
        abs_ac = np.abs(ac)
        max_abs = np.max(abs_ac)
        
        if max_abs < 1e-15:
            break
        
        # Log-sum-exp approximation of max
        shifted = beta * (abs_ac - max_abs)
        exp_shifted = np.exp(shifted)
        sum_exp = np.sum(exp_shifted)
        log_sum_exp = max_abs + (1.0 / beta) * np.log(sum_exp)
        
        val = log_sum_exp / (s * s)
        
        # Weights for each lag
        weights = exp_shifted / sum_exp
        
        # Compute gradient using FFT
        # d(log_sum_exp)/dx[j] = beta * sum_k weights[k] * sign(ac[k]) * d(ac[k])/dx[j]
        # d(ac[k])/dx[j] = x[k-j] + x[j+k] (with bounds)
        
        # This is expensive to compute directly. Use the fact that:
        # sum_k weights[k] * sign(ac[k]) * x[k-j] = (weights_sign * x)[k-j]
        # where weights_sign[k] = weights[k] * sign(ac[k])
        
        weights_sign = weights * np.sign(ac)
        # Replace zeros with small positive
        weights_sign[np.abs(weights_sign) < 1e-15] = 1e-15
        
        # Compute sum_k weights_sign[k] * x[k-j] for all j
        # This is a convolution: (weights_sign * x)[j] = sum_k weights_sign[k] * x[k-j]
        # Use FFT for speed
        L_fft = 1
        while L_fft < 2 * N:
            L_fft *= 2
        
        ws_fft = np.fft.rfft(weights_sign, n=L_fft)
        x_fft = np.fft.rfft(x, n=L_fft)
        conv1 = np.fft.irfft(ws_fft * x_fft, n=L_fft)[:2*N-1]
        
        # sum_k weights_sign[k] * x[j+k] = sum_k weights_sign[k] * x[j+k]
        # This is a cross-correlation
        # Let w2[k] = weights_sign[-k] (flip), then sum_k w2[k] * x[j+k] = (w2 * x)[j]
        w2 = np.zeros_like(weights_sign)
        for k in range(len(weights_sign)):
            idx = -k
            if 0 <= idx < len(weights_sign):
                w2[idx] = weights_sign[k]
        
        w2_fft = np.fft.rfft(w2, n=L_fft)
        conv2 = np.fft.irfft(w2_fft * x_fft, n=L_fft)[:2*N-1]
        
        # d(ac[k])/dx[j] = x[k-j] + x[j+k]
        # sum_k weights_sign[k] * (x[k-j] + x[j+k]) = conv1[j] + conv2[j]
        grad_inner = conv1[:N] + conv2[:N]
        
        d_logsumexp_dx = beta * grad_inner
        
        # Full gradient
        grad = (1.0 / (s * s)) * d_logsumexp_dx - (2.0 * log_sum_exp / (s ** 3)) * np.ones(N)
        
        # Gradient descent step
        x = x - lr * grad
        
        # Renormalize
        s_new = np.sum(x)
        if abs(s_new) > 1e-10:
            x = x * (s_target / s_new)
        
        val_new = _objective(x)
        if val_new < best_val:
            best_val = val_new
            best_x = x.copy()
            lr = min(0.01, lr * 1.05)
        else:
            lr = max(1e-5, lr * 0.9)
    
    return best_x, best_val

def _peak_suppression(x0, n_iters=100):
    """
    Directly suppress the peak autocorrelation lag by adjusting x.
    """
    s_target = np.sum(x0)
    best_x = x0.copy()
    best_val = _objective(x0)
    
    x = x0.copy()
    
    for it in range(n_iters):
        if _check_deadline():
            break
        
        s = np.sum(x)
        if abs(s) < 1e-10:
            break
        
        ac = _autoconv_fft(x)
        abs_ac = np.abs(ac)
        
        # Find the peak (excluding lag 0)
        abs_ac_nonzero = abs_ac.copy()
        abs_ac_nonzero[0] = 0
        peak_lag = np.argmax(abs_ac_nonzero)
        max_val = abs_ac_nonzero[peak_lag]
        
        if max_val < 1e-12:
            break
        
        peak_sign = np.sign(ac[peak_lag])
        if peak_sign == 0:
            peak_sign = 1.0
        
        # Compute gradient at this peak
        grad = np.zeros(N)
        for j in range(N):
            g = 0.0
            idx1 = peak_lag - j
            if 0 <= idx1 < N:
                g += x[idx1]
            idx2 = j + peak_lag
            if 0 <= idx2 < N:
                g += x[idx2]
            grad[j] = peak_sign * g
        
        # Step in negative gradient direction
        grad_norm = np.max(np.abs(grad))
        if grad_norm < 1e-15:
            break
        
        step_size = 0.005 * max_val / grad_norm
        x = x - step_size * grad
        
        # Renormalize
        s_new = np.sum(x)
        if abs(s_new) > 1e-10:
            x = x * (s_target / s_new)
        
        val_new = _objective(x)
        if val_new < best_val:
            best_val = val_new
            best_x = x.copy()
    
    return best_x, best_val

def _random_spectral_init(seed):
    """Generate random spectral initialization with good properties."""
    L_fft = 1
    while L_fft < 2 * N:
        L_fft *= 2
    
    rng = np.random.default_rng(seed)
    
    # Create a power spectrum that's somewhat flat but with random variation
    freqs = np.arange(L_fft // 2 + 1)
    
    # Mix of components
    base = np.ones(L_fft // 2 + 1)
    
    # Add some low-frequency content
    low_freq = np.exp(-freqs / 50.0)
    # Add some high-frequency content
    high_freq = np.exp(-(L_fft - freqs) / 50.0)
    
    # Random modulation
    modulation = 1.0 + 0.3 * np.sin(rng.uniform(0, 2*np.pi) * freqs / 10.0)
    
    P = base * (0.5 + 0.3 * low_freq + 0.2 * high_freq) * modulation
    P = np.maximum(0.1, P)
    
    # Random phase
    phase = rng.uniform(0, 2 * np.pi, size=L_fft // 2 + 1)
    phase[0] = 0  # Ensure x is real
    
    X_mag = np.sqrt(P)
    X = X_mag * np.exp(1j * phase)
    
    X_full = np.zeros(L_fft, dtype=complex)
    X_full[:L_fft // 2 + 1] = X
    for k in range(1, L_fft // 2):
        X_full[L_fft - k] = np.conj(X_full[k])
    
    x = np.fft.irfft(X_full, n=L_fft)[:N]
    
    s = np.sum(x)
    if abs(s) > 1e-10:
        x = x * (N / s)
    
    return x

def _structured_init():
    """Create structured initializations likely to have good autocorrelation properties."""
    inits = []
    
    # All ones
    inits.append(np.ones(N))
    
    # Cosine windows
    t = np.linspace(0, 1, N)
    inits.append(1.0 + 0.5 * np.cos(np.pi * t))
    inits.append(1.0 + 0.3 * np.sin(2 * np.pi * t))
    inits.append(1.0 + 0.2 * np.cos(2 * np.pi * 3 * t))
    inits.append(1.0 + 0.1 * np.sin(4 * np.pi * t))
    
    # Exponential decay
    inits.append(np.exp(-2.0 * t))
    inits.append(np.exp(-1.0 * t) + 0.5)
    
    # Alternating patterns
    alt = np.ones(N)
    alt[1::2] = 0.8
    inits.append(alt)
    
    alt2 = np.ones(N)
    alt2[::3] = 0.7
    inits.append(alt2)
    
    return inits

def construct():
    # Generate diverse initializations
    initializations = _structured_init()
    
    # Add random spectral initializations
    for seed in range(30):
        if _check_deadline():
            break
        initializations.append(_random_spectral_init(seed))
    
    # Find best initialization
    best_x = initializations[0].copy()
    best_val = _objective(best_x)
    
    for x0 in initializations[1:]:
        if _check_deadline():
            break
        val = _objective(x0)
        if val < best_val:
            best_val = val
            best_x = x0.copy()
    
    if _check_deadline():
        return best_x
    
    # Phase 1: Spectral phase optimization
    x1, val1 = _spectral_phase_optimize(best_x, n_iters=300, lr=0.01)
    if val1 < best_val:
        best_val = val1
        best_x = x1.copy()
    
    if _check_deadline():
        return best_x
    
    # Phase 2: Amplitude shaping
    x2, val2 = _amplitude_shaping(best_x, n_iters=150)
    if val2 < best_val:
        best_val = val2
        best_x = x2.copy()
    
    if _check_deadline():
        return best_x
    
    # Phase 3: Gradient descent with smooth objective
    x3, val3 = _gradient_descent_smooth(best_x, n_iters=200, beta=20.0)
    if val3 < best_val:
        best_val = val3
        best_x = x3.copy()
    
    if _check_deadline():
        return best_x
    
    # Phase 4: Peak suppression
    x4, val4 = _peak_suppression(best_x, n_iters=100)
    if val4 < best_val:
        best_val = val4
        best_x = x4.copy()
    
    if _check_deadline():
        return best_x
    
    # Phase 5: Fine-tune with smaller beta
    x5, val5 = _gradient_descent_smooth(best_x, n_iters=150, beta=50.0)
    if val5 < best_val:
        best_val = val5
        best_x = x5.copy()
    
    if _check_deadline():
        return best_x
    
    # Phase 6: Another round of peak suppression
    x6, val6 = _peak_suppression(best_x, n_iters=50)
    if val6 < best_val:
        best_val = val6
        best_x = x6.copy()
    
    if _check_deadline():
        return best_x
    
    # Phase 7: Local perturbation search
    rng = np.random.default_rng(999)
    for _ in range(100):
        if _check_deadline():
            break
        
        trial = best_x.copy()
        mask = rng.random(N) < 0.05
        trial[mask] += rng.normal(0, 0.02, size=np.sum(mask))
        
        s = np.sum(trial)
        s_orig = np.sum(best_x)
        if abs(s) > 1e-10:
            trial = trial * (s_orig / s)
        
        val = _objective(trial)
        if val < best_val:
            best_val = val
            best_x = trial.copy()
    
    return best_x
