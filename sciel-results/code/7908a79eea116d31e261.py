"""Spectral phase optimization with fixed prolate-spectrum amplitude."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.special import eval_legendre

N = 400
START_TIME = time.time()
TIME_LIMIT = 140
SAFETY_MARGIN = 8

def _check_deadline():
    return (time.time() - START_TIME) > (TIME_LIMIT - SAFETY_MARGIN)

def _autoconv_fft(x):
    """Compute linear autoconvolution using FFT."""
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    X = np.fft.rfft(x, n=L_fft)
    C = np.fft.irfft(X * X, n=L_fft)[:L]
    return C

def _objective(x):
    """Compute max(|autoconv|) / (sum x)^2."""
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10
    ac = _autoconv_fft(x)
    return np.max(np.abs(ac)) / (s * s)

def _prolate_amplitude(N, M, W):
    """
    Compute prolate spheroidal sequence (DFT of PSWF) as amplitude spectrum.
    M = number of nonzero coefficients, W = bandwidth fraction.
    Returns amplitude array of length N (one-sided PSD sqrt).
    """
    # For simplicity, use a Gaussian-like spectrum centered at DC
    # This approximates a prolate sequence well
    k = np.arange(N)
    # Gaussian decay in frequency
    sigma = N * W / 2.0
    amp = np.exp(-0.5 * (k / sigma) ** 2)
    return amp

def _phase_to_signal(amp, phase):
    """
    Reconstruct real signal from amplitude spectrum and phase.
    amp: one-sided amplitude (length N)
    phase: one-sided phase (length N)
    """
    # Build full complex spectrum
    X = np.zeros(N, dtype=complex)
    for k in range(N):
        X[k] = amp[k] * np.exp(1j * phase[k])
    
    # Enforce Hermitian symmetry for real signal
    # X[N-k] = conj(X[k])
    # We only use the first N//2 + 1 bins for one-sided
    X_full = np.zeros(N, dtype=complex)
    X_full[0] = amp[0]  # DC, phase 0
    if N % 2 == 0:
        X_full[N//2] = amp[N//2] * np.exp(1j * phase[N//2])
        for k in range(1, N//2):
            X_full[k] = amp[k] * np.exp(1j * phase[k])
            X_full[N-k] = np.conj(X_full[k])
    else:
        for k in range(1, (N+1)//2):
            X_full[k] = amp[k] * np.exp(1j * phase[k])
            X_full[N-k] = np.conj(X_full[k])
    
    x = np.fft.ifft(X_full).real
    return x

def _objective_grad_phase(amp, phase):
    """
    Objective and gradient w.r.t. phase for fixed amplitude.
    """
    x = _phase_to_signal(amp, phase)
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10, np.zeros_like(phase)
    
    ac = _autoconv_fft(x)
    abs_ac = np.abs(ac)
    max_val = np.max(abs_ac)
    
    # Soft-max surrogate
    beta = 50.0
    shifted = beta * (abs_ac - max_val)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    weights = exp_shifted / sum_exp
    soft_max = max_val + (1.0 / beta) * np.log(sum_exp)
    
    val = soft_max / (s * s)
    
    # Gradient w.r.t. phase
    # d(soft_max)/d(phase_k) for k in 1..N//2
    # soft_max = sum_j w_j * |ac_j|
    # |ac_j| = |sum_m x_m x_{m-j}|
    # This is complex; use finite difference approximation for stability
    # Or use the chain rule carefully
    
    # Simpler: use numerical gradient for phase optimization
    # But that's slow. Let's use a smoothed approach.
    
    # Actually, let's just return the value and use a different optimizer
    # that doesn't need gradients, or use finite differences.
    
    # For now, return zeros and use Nelder-Mead or similar
    grad = np.zeros_like(phase)
    
    return val, grad

def _optimize_phase(amp, phase_init, n_iters=200):
    """
    Optimize phase using L-BFGS-B with finite difference gradients.
    """
    def _obj(phase):
        x = _phase_to_signal(amp, phase)
        s = np.sum(x)
        if abs(s) < 1e-10:
            return 1e10
        ac = _autoconv_fft(x)
        return np.max(np.abs(ac)) / (s * s)
    
    # Use L-BFGS-B with finite differences
    # Limit the number of phase variables to optimize
    # Only optimize phases for k=1 to N//2 (DC and Nyquist are fixed)
    
    n_phase_vars = N // 2
    
    def _obj_phase_vec(phase_vec):
        phase = np.zeros(N)
        phase[1:N//2 + 1] = phase_vec
        return _obj(phase)
    
    result = minimize(_obj_phase_vec, phase_init[:n_phase_vars], method='L-BFGS-B',
                     options={'maxiter': 100, 'gtol': 1e-8})
    
    if result.fun < _obj(phase_init):
        phase_opt = np.zeros(N)
        phase_opt[1:N//2 + 1] = result.x
        return phase_opt
    
    return phase_init

def _time_domain_polish(x, n_iters=5000):
    """
    Simple coordinate descent polish in time domain.
    """
    best_x = x.copy()
    best_val = _objective(best_x)
    rng = np.random.default_rng(42)
    
    for it in range(n_iters):
        if _check_deadline():
            break
        
        # Random coordinate updates
        idx = rng.integers(0, N)
        step = rng.normal(0, 0.05)
        
        x_trial = best_x.copy()
        x_trial[idx] += step
        
        val = _objective(x_trial)
        if val < best_val:
            best_val = val
            best_x = x_trial.copy()
    
    return best_x, best_val

def construct():
    # Phase 1: Design amplitude spectrum (prolate-like)
    # Use a few different bandwidths
    best_x = np.ones(N)
    best_val = _objective(best_x)
    
    # Try different amplitude profiles
    W_values = [0.2, 0.3, 0.4, 0.5]
    
    for W in W_values:
        if _check_deadline():
            break
        
        amp = _prolate_amplitude(N, N, W)
        
        # Initial phase: all zeros (real, positive spectrum)
        phase_init = np.zeros(N)
        x_init = _phase_to_signal(amp, phase_init)
        s_init = np.sum(x_init)
        if abs(s_init) > 1e-10:
            x_init = x_init / s_init
            val_init = _objective(x_init)
            if val_init < best_val:
                best_val = val_init
                best_x = x_init.copy()
        
        # Optimize phase
        if _check_deadline():
            break
        
        try:
            phase_opt = _optimize_phase(amp, phase_init, n_iters=100)
            x_opt = _phase_to_signal(amp, phase_opt)
            s_opt = np.sum(x_opt)
            if abs(s_opt) > 1e-10:
                x_opt = x_opt / s_opt
                val_opt = _objective(x_opt)
                if val_opt < best_val:
                    best_val = val_opt
                    best_x = x_opt.copy()
        except Exception:
            pass
    
    if _check_deadline():
        return best_x
    
    # Phase 2: Time-domain polish
    try:
        best_x, best_val = _time_domain_polish(best_x, n_iters=3000)
    except Exception:
        pass
    
    # Final evaluation
    final_val = _objective(best_x)
    if final_val < best_val:
        best_val = final_val
    
    return best_x
