"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import differential_evolution

N = 400

# EVOLVE-BLOCK-START
def _autoconv_full(x):
    """Full linear autoconvolution of x (length 2N-1)."""
    n = len(x)
    if n == 0:
        return np.array([])
    size = 2 * n - 1
    fft_size = 1
    while fft_size < size:
        fft_size <<= 1
    F = np.fft.rfft(x, n=fft_size)
    conv = np.fft.irfft(F * F, n=fft_size)
    return conv[:size]


def _score_offdiag(x):
    """
    Compute the metric: max(|autoconv|) excluding DC term / (sum(x))^2
    We want to minimize this.
    """
    integral = np.sum(x)
    if abs(integral) < 1e-12:
        return 1e12
    conv = _autoconv_full(x)
    n = len(x)
    # Exclude the DC term at index n-1
    if n > 1:
        mask = np.ones(len(conv), dtype=bool)
        mask[n-1] = False
        max_abs = np.max(np.abs(conv[mask]))
    else:
        max_abs = np.max(np.abs(conv))
    return max_abs / (integral * integral)


def _smooth_max_abs_conv(x, temperature=0.1):
    """
    Smooth surrogate for max(|autoconv|) using log-sum-exp.
    This makes the objective differentiable.
    """
    conv = _autoconv_full(x)
    abs_conv = np.abs(conv)
    # Log-sum-exp for numerical stability
    m = np.max(abs_conv)
    if m < 1e-12:
        return 0.0, conv
    log_sum_exp = m + np.log(np.sum(np.exp((abs_conv - m) / temperature)) * temperature)
    return log_sum_exp, conv


def _param_to_signal(params):
    """
    Convert spectral coefficients to a time-domain signal.
    """
    M = len(params) // 2
    c0 = params[0]
    x = np.full(N, c0)
    n = np.arange(N)
    for k in range(1, M + 1):
        c_k = params[2 * k - 1]
        d_k = params[2 * k]
        x += c_k * np.cos(2 * np.pi * k * n / N) + d_k * np.sin(2 * np.pi * k * n / N)
    return x


def _objective(params):
    """Objective function for differential evolution."""
    x = _param_to_signal(params)
    return _score_offdiag(x)


def _gradient_descent_polish(x0, max_iter=2000, lr_init=0.01, deadline=None):
    """
    Gradient descent polish on the time-domain signal using the smooth surrogate.
    We optimize the smooth max surrogate to avoid the nonsmooth max.
    Uses off-diagonal metric.
    """
    x = x0.copy()
    lr = lr_init
    best_x = x.copy()
    best_score = _score_offdiag(x)
    
    # Precompute FFT size
    size = 2 * N - 1
    fft_size = 1
    while fft_size < size:
        fft_size <<= 1
    
    # Mask for off-diagonal (exclude DC term at index N-1)
    mask = np.ones(size, dtype=bool)
    mask[N-1] = False
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute current conv and smooth max (off-diagonal only)
        F = np.fft.rfft(x, n=fft_size)
        conv = np.fft.irfft(F * F, n=fft_size)[:size]
        abs_conv = np.abs(conv)
        abs_conv_off = abs_conv[mask]
        
        # Compute smooth max (log-sum-exp) on off-diagonal
        m = np.max(abs_conv_off)
        if m < 1e-12:
            break
        # Temperature annealing: start with higher temp, decrease
        temp = 0.5 * (1.0 + 0.5 * np.cos(np.pi * it / max_iter))  # 0.5 to 1.0
        log_sum_exp = m + temp * np.log(np.sum(np.exp((abs_conv_off - m) / temp)))
        
        # Compute integral
        integral = np.sum(x)
        if abs(integral) < 1e-12:
            break
        
        # Objective: log_sum_exp / integral^2
        obj = log_sum_exp / (integral * integral)
        
        # Compute gradient numerically (central differences) for a few coordinates
        # To save time, only compute gradient for a random subset each iteration
        n_coords = min(30, N)
        coords = np.random.choice(N, n_coords, replace=False)
        
        grad = np.zeros(N)
        eps = 1e-4
        
        for idx in coords:
            x_plus = x.copy()
            x_plus[idx] += eps
            x_minus = x.copy()
            x_minus[idx] -= eps
            
            # Compute objective for perturbed signals
            def _obj_val(sig):
                Fp = np.fft.rfft(sig, n=fft_size)
                convp = np.fft.irfft(Fp * Fp, n=fft_size)[:size]
                abs_cp = np.abs(convp)
                abs_cp_off = abs_cp[mask]
                mp = np.max(abs_cp_off)
                if mp < 1e-12:
                    return 0.0
                temp_p = 0.5 * (1.0 + 0.5 * np.cos(np.pi * it / max_iter))
                lse_p = mp + temp_p * np.log(np.sum(np.exp((abs_cp_off - mp) / temp_p)))
                int_p = np.sum(sig)
                if abs(int_p) < 1e-12:
                    return 1e12
                return lse_p / (int_p * int_p)
            
            f_plus = _obj_val(x_plus)
            f_minus = _obj_val(x_minus)
            grad[idx] = (f_plus - f_minus) / (2 * eps)
        
        # Update
        x = x - lr * grad
        
        # Normalize to avoid drift (scale invariance)
        s = np.sum(x)
        if abs(s) > 1e-12:
            x = x * (1.0 / s)  # Fix integral to 1
        
        # Check improvement
        score = _score_offdiag(x)
        if score < best_score:
            best_score = score
            best_x = x.copy()
        
        # Learning rate decay
        lr *= 0.999
    
    return best_x, best_score


def construct():
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    # Initial best: all ones (off-diagonal score is 0 for constant, but we need non-trivial)
    # Start with a non-constant signal
    rng = np.random.default_rng(42)
    
    # Try several initial patterns
    candidates = []
    
    # Hanning window
    n = np.arange(N)
    hanning = 0.5 * (1 - np.cos(2 * np.pi * n / (N - 1)))
    candidates.append(hanning)
    
    # Sine wave
    candidates.append(np.sin(2 * np.pi * n / N))
    
    # Cosine wave
    candidates.append(np.cos(2 * np.pi * n / N))
    
    # Random
    candidates.append(rng.normal(size=N))
    
    # Triangular
    candidates.append(np.abs(n - N/2) / (N/2))
    
    # Evaluate all candidates
    best_x = None
    best_score = 1e12
    
    for cand in candidates:
        score = _score_offdiag(cand)
        if score < best_score:
            best_score = score
            best_x = cand.copy()
    
    # Differential evolution with spectral parameterization
    M = 20
    n_params = 2 * M + 1
    bounds = [(-10, 10)] * n_params
    
    # Run differential evolution
    try:
        result = differential_evolution(
            _objective,
            bounds=bounds,
            maxiter=60,
            popsize=15,
            tol=1e-8,
            seed=42,
            polish=False,  # We do our own polish
            updating='immediate'
        )
        
        if result.fun < best_score:
            x_opt = _param_to_signal(result.x)
            score_opt = _score_offdiag(x_opt)
            if score_opt < best_score:
                best_x = x_opt
                best_score = score_opt
    except Exception:
        pass
    
    # Gradient descent polish from the best spectral solution
    if time.time() < deadline - 30:
        x_polished, score_polished = _gradient_descent_polish(
            best_x, max_iter=2000, lr_init=0.01, deadline=deadline - 20
        )
        if score_polished < best_score:
            best_x = x_polished
            best_score = score_polished
    
    # Try a few more random restarts if time permits
    for restart in range(3):
        if time.time() > deadline - 40:
            break
        
        init = rng.uniform(-5, 5, size=n_params)
        
        try:
            result = differential_evolution(
                _objective,
                bounds=bounds,
                maxiter=30,
                popsize=12,
                tol=1e-8,
                seed=restart + 100,
                x0=init,
                polish=False,
                updating='immediate'
            )
            
            if result.fun < best_score:
                x_opt = _param_to_signal(result.x)
                score_opt = _score_offdiag(x_opt)
                if score_opt < best_score:
                    best_x = x_opt
                    best_score = score_opt
                    
                    # Polish this new best
                    if time.time() < deadline - 30:
                        x_polished, score_polished = _gradient_descent_polish(
                            best_x, max_iter=1000, lr_init=0.01, deadline=deadline - 20
                        )
                        if score_polished < best_score:
                            best_x = x_polished
                            best_score = score_polished
        except Exception:
            pass
    
    return best_x
# EVOLVE-BLOCK-END
