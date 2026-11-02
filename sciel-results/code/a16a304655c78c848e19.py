"""Model-and-optimize: spectral PSD optimization + L-BFGS + coordinate descent."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

START_TIME = time.time()
TIME_LIMIT = 135  # safety margin before 150s kill

def _check_deadline():
    return (time.time() - START_TIME) > (TIME_LIMIT - 5)

def _autoconv_fft(x):
    """Compute linear autoconvolution using FFT."""
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    X = np.fft.rfft(x, n=L_fft)
    C = np.fft.irfft(X * X, n=L_fft)[:L]
    return C

def _soft_max_and_grad(x, beta):
    """
    Compute soft-max surrogate for max|ac| and its gradient.
    Returns (soft_max_value, gradient_array)
    """
    ac = _autoconv_fft(x)
    abs_ac = np.abs(ac)
    
    m = np.max(abs_ac)
    exps = np.exp(beta * (abs_ac - m))
    log_sum = np.log(np.sum(exps))
    soft_val = m + log_sum / beta
    
    weights = exps / np.sum(exps)
    signs = np.sign(ac)
    signs[ac == 0] = 1.0
    grad_ac = weights * signs
    
    g = grad_ac  # length 2N-1
    cumsum_g = np.concatenate([[0], np.cumsum(g)])
    idx_end = np.minimum(np.arange(N) + N - 1, 2*N-2) + 1
    idx_start = np.arange(N)
    window_sums = cumsum_g[idx_end] - cumsum_g[idx_start]
    term2 = x * window_sums
    
    L_corr = 1
    while L_corr < 3 * N:
        L_corr *= 2
    X_f = np.fft.rfft(x, n=L_corr)
    G_f = np.fft.rfft(g, n=L_corr)
    corr = np.fft.irfft(X_f * np.conj(G_f), n=L_corr)
    term1 = corr[:N]
    
    grad_x = term1 + term2
    return soft_val, grad_x

def _objective_only(x, beta):
    """Compute objective value only (fast, no gradient)."""
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10
    ac = _autoconv_fft(x)
    abs_ac = np.abs(ac)
    m = np.max(abs_ac)
    if beta < 1e6:
        exps = np.exp(beta * (abs_ac - m))
        log_sum = np.log(np.sum(exps))
        soft_val = m + log_sum / beta
    else:
        soft_val = m
    return soft_val / (s * s)

def _objective_and_grad(x, beta):
    """Compute objective and gradient."""
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10, np.zeros(N)
    soft_val, grad_soft = _soft_max_and_grad(x, beta)
    # Objective = soft_val / s^2
    val = soft_val / (s * s)
    # Gradient: d/dx (soft/s^2) = grad_soft/s^2 - 2*soft/s^3 * ones
    grad = grad_soft / (s * s) - 2.0 * soft_val * np.ones(N) / (s * s * s)
    return val, grad

def _spectral_init(rng, n_init=10):
    """Generate initial points by optimizing PSD in frequency domain."""
    inits = []
    # PSD: P[k] = |X[k]|^2 where X = fft(x)
    # We want to minimize max|ifft(P)| / (sum(x))^2
    # Use iterative water-filling / dual ascent on PSD
    
    for i in range(n_init):
        # Random PSD initialization
        L_fft = 1
        while L_fft < 2*N-1:
            L_fft *= 2
        # Initialize PSD as flat + noise
        P = np.ones(N) * 0.5 + rng.uniform(0, 0.3, size=N)
        
        # Iteratively refine: minimize peak of ifft(P) subject to sum constraint
        for _ in range(50):
            if _check_deadline():
                break
            # Compute time-domain signal from PSD
            X_mag = np.sqrt(P)
            # Random phases
            phases = rng.uniform(0, 2*np.pi, size=N)
            X = X_mag * np.exp(1j * phases)
            # Zero-pad and inverse FFT to get x
            X_padded = np.zeros(L_fft, dtype=complex)
            X_padded[:N] = X
            x_candidate = np.fft.irfft(X_padded, n=L_fft)[:N]
            
            # Evaluate
            val = _objective_only(x_candidate, 50.0)
            if val < 1.0:  # reasonable candidate
                inits.append(x_candidate)
        
        # Also try structured PSDs
        if i < 3:
            # Low-pass PSD
            freqs = np.fft.rfftfreq(N)
            P_lp = np.exp(-5 * freqs * N)
            X_mag = np.sqrt(P_lp)
            phases = rng.uniform(0, 2*np.pi, size=len(X_mag))
            X = X_mag * np.exp(1j * phases)
            X_padded = np.zeros(L_fft, dtype=complex)
            X_padded[:len(X)] = X
            x_candidate = np.fft.irfft(X_padded, n=L_fft)[:N]
            inits.append(x_candidate)
            
            # Band-pass PSD
            P_bp = np.exp(-10 * (freqs * N - 0.3)**2)
            X_mag = np.sqrt(P_bp)
            phases = rng.uniform(0, 2*np.pi, size=len(X_mag))
            X = X_mag * np.exp(1j * phases)
            X_padded = np.zeros(L_fft, dtype=complex)
            X_padded[:len(X)] = X
            x_candidate = np.fft.irfft(X_padded, n=L_fft)[:N]
            inits.append(x_candidate)
    
    return inits

def _coordinate_descent(x, beta, max_rounds=500):
    """Coordinate descent refinement."""
    x = x.copy()
    best_val = _objective_only(x, beta)
    step_sizes = [0.1, 0.05, 0.01, 0.005, 0.001]
    
    for round_idx in range(max_rounds):
        if _check_deadline():
            break
        improved = False
        for j in range(N):
            if _check_deadline():
                break
            for ss in step_sizes:
                for sign in [1, -1]:
                    x_trial = x.copy()
                    x_trial[j] += sign * ss
                    val = _objective_only(x_trial, beta)
                    if val < best_val:
                        best_val = val
                        x = x_trial
                        improved = True
                        break
                if improved:
                    break
        if not improved:
            break
    
    return x, best_val

def construct():
    best_x = np.ones(N)
    best_val = _objective_only(best_x, 30.0)
    
    if _check_deadline():
        return best_x
    
    rng = np.random.default_rng(12345)
    
    # Phase 1: Spectral initialization
    spectral_inits = _spectral_init(rng, n_init=15)
    
    # Phase 2: Structured initializations
    t = np.linspace(0, 1, N)
    structured_inits = [
        np.ones(N),
        1.0 + 0.3 * np.sin(2 * np.pi * t),
        1.0 + 0.5 * np.cos(2 * np.pi * t),
        1.0 + 0.3 * np.sin(4 * np.pi * t),
        1.0 + 0.5 * np.cos(4 * np.pi * t),
        np.exp(1.0 * t),
        np.exp(-1.0 * t),
        1.0 + 0.5 * (t - 0.5),
        np.maximum(0.3, 1.0 - 2.0 * np.abs(t - 0.5)),
    ]
    
    # Add random initializations
    for _ in range(15):
        scale = rng.uniform(0.5, 1.5)
        structured_inits.append(scale * rng.normal(1.0, 0.2, size=N))
    
    all_inits = spectral_inits + structured_inits
    
    # Phase 3: L-BFGS optimization from each init
    betas = [30.0, 50.0, 100.0]
    
    for x0 in all_inits:
        if _check_deadline():
            break
        
        x_start = x0 + rng.normal(0, 0.01, size=N)
        
        for beta in betas:
            if _check_deadline():
                break
            
            try:
                def obj_and_grad_func(x, _beta=beta):
                    return _objective_and_grad(x, _beta)
                
                result = minimize(obj_and_grad_func, x_start, method='L-BFGS-B',
                                  jac=True, options={'maxiter': 150, 'gtol': 1e-8, 'ftol': 1e-10})
                
                if result.fun < best_val:
                    best_val = result.fun
                    best_x = result.x.copy()
                    
            except Exception:
                pass
    
    if _check_deadline():
        return best_x
    
    # Phase 4: Polish with higher beta
    for beta in [200.0, 500.0]:
        if _check_deadline():
            break
        try:
            def obj_and_grad_func(x, _beta=beta):
                return _objective_and_grad(x, _beta)
            
            result = minimize(obj_and_grad_func, best_x, method='L-BFGS-B',
                              jac=True, options={'maxiter': 100, 'gtol': 1e-10, 'ftol': 1e-12})
            
            if result.fun < best_val:
                best_val = result.fun
                best_x = result.x.copy()
                
        except Exception:
            pass
    
    if _check_deadline():
        return best_x
    
    # Phase 5: Coordinate descent refinement
    best_x, best_val = _coordinate_descent(best_x, beta=100.0, max_rounds=300)
    
    return best_x
# EVOLVE-BLOCK-END
