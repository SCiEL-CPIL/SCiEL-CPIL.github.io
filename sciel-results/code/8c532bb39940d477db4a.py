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


def _score_from_signal(x):
    """
    Compute the metric: max(|autoconv|) / (sum(x))^2
    We want to minimize this.
    """
    integral = np.sum(x)
    if abs(integral) < 1e-12:
        return 1e12
    conv = _autoconv_full(x)
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
    return _score_from_signal(x)


def _gradient_descent_polish(x0, max_iter=2000, lr_init=0.01, deadline=None):
    """
    Gradient descent polish on the time-domain signal using the smooth surrogate.
    We optimize the smooth max surrogate to avoid the nonsmooth max.
    """
    x = x0.copy()
    lr = lr_init
    best_x = x.copy()
    best_score = _score_from_signal(x)
    
    # Precompute FFT size
    size = 2 * N - 1
    fft_size = 1
    while fft_size < size:
        fft_size <<= 1
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute current conv and smooth max
        F = np.fft.rfft(x, n=fft_size)
        conv = np.fft.irfft(F * F, n=fft_size)[:size]
        abs_conv = np.abs(conv)
        
        # Compute smooth max (log-sum-exp)
        m = np.max(abs_conv)
        if m < 1e-12:
            break
        # Temperature annealing: start with higher temp, decrease
        temp = 0.5 * (1.0 + 0.5 * np.cos(np.pi * it / max_iter))  # 0.5 to 1.0
        log_sum_exp = m + temp * np.log(np.sum(np.exp((abs_conv - m) / temp)))
        
        # Compute integral
        integral = np.sum(x)
        if abs(integral) < 1e-12:
            break
        
        # Objective: log_sum_exp / integral^2
        obj = log_sum_exp / (integral * integral)
        
        # Compute gradient numerically (central differences) for a few coordinates
        # To save time, only compute gradient for a random subset each iteration
        n_coords = min(20, N)
        coords = np.random.choice(N, n_coords, replace=False)
        
        grad = np.zeros(N)
        eps = 1e-5
        
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
                mp = np.max(abs_cp)
                if mp < 1e-12:
                    return 0.0
                temp_p = 0.5 * (1.0 + 0.5 * np.cos(np.pi * it / max_iter))
                lse_p = mp + temp_p * np.log(np.sum(np.exp((abs_cp - mp) / temp_p)))
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
        score = _score_from_signal(x)
        if score < best_score:
            best_score = score
            best_x = x.copy()
        
        # Learning rate decay
        lr *= 0.999
    
    return best_x, best_score


def construct():
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    # Initial best: all ones
    best_x = np.ones(N)
    best_score = _score_from_signal(best_x)
    
    # Differential evolution with spectral parameterization
    M = 15
    n_params = 2 * M + 1
    bounds = [(-10, 10)] * n_params
    
    # Run differential evolution
    try:
        result = differential_evolution(
            _objective,
            bounds=bounds,
            maxiter=80,
            popsize=12,
            tol=1e-8,
            seed=42,
            polish=False,  # We do our own polish
            updating='immediate'
        )
        
        if result.fun < best_score:
            x_opt = _param_to_signal(result.x)
            score_opt = _score_from_signal(x_opt)
            if score_opt < best_score:
                best_x = x_opt
                best_score = score_opt
    except Exception:
        pass
    
    # Gradient descent polish from the best spectral solution
    if time.time() < deadline - 30:
        x_polished, score_polished = _gradient_descent_polish(
            best_x, max_iter=1500, lr_init=0.005, deadline=deadline - 20
        )
        if score_polished < best_score:
            best_x = x_polished
            best_score = score_polished
    
    # Try a few more random restarts if time permits
    rng = np.random.default_rng(42)
    for restart in range(2):
        if time.time() > deadline - 40:
            break
        
        init = rng.uniform(-5, 5, size=n_params)
        
        try:
            result = differential_evolution(
                _objective,
                bounds=bounds,
                maxiter=40,
                popsize=10,
                tol=1e-8,
                seed=restart + 100,
                x0=init,
                polish=False,
                updating='immediate'
            )
            
            if result.fun < best_score:
                x_opt = _param_to_signal(result.x)
                score_opt = _score_from_signal(x_opt)
                if score_opt < best_score:
                    best_x = x_opt
                    best_score = score_opt
                    
                    # Polish this new best
                    if time.time() < deadline - 30:
                        x_polished, score_polished = _gradient_descent_polish(
                            best_x, max_iter=1000, lr_init=0.005, deadline=deadline - 20
                        )
                        if score_polished < best_score:
                            best_x = x_polished
                            best_score = score_polished
        except Exception:
            pass
    
    return best_x
# EVOLVE-BLOCK-END
