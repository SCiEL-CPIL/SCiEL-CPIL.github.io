"""Optimal control formulation: smooth waveform via low-pass filtered impulse train,
then discretize and optimize the continuous parameters for C3' minimization."""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 400
START_TIME = time.time()
TIME_LIMIT = 145

def _check_deadline():
    return (time.time() - START_TIME) > (TIME_LIMIT - 3)

def _autoconv_fft(x):
    """Compute linear autoconvolution using FFT."""
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    X = np.fft.rfft(x, n=L_fft)
    C = np.fft.irfft(X * X, n=L_fft)[:L]
    return C

def _objective_only(x):
    """Compute objective value only."""
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10
    ac = _autoconv_fft(x)
    abs_ac = np.abs(ac)
    m = np.max(abs_ac)
    return m / (s * s)

def _build_smooth_waveform(coeffs, n_taps=64, freq_cutoff=0.15):
    """
    Build a smooth waveform from filter coefficients.
    The waveform is a sum of shifted and scaled Gaussian kernels.
    coeffs: array of length K, representing amplitudes of K Gaussian bumps.
    """
    K = len(coeffs)
    # Place K Gaussian bumps evenly across [0, N)
    centers = np.linspace(0.5, N - 0.5, K)
    # Gaussian width inversely related to frequency cutoff
    sigma = N * freq_cutoff / (2 * np.pi)
    
    x = np.zeros(N)
    for k in range(K):
        # Gaussian kernel centered at centers[k]
        diff = np.arange(N) - centers[k]
        x += coeffs[k] * np.exp(-diff**2 / (2 * sigma**2))
    
    return x

def _build_waveform_v2(params, n_bumps=32):
    """
    Alternative: parametrize as sum of sinusoids with optimized amplitudes/phases.
    params: [a1, phi1, a2, phi2, ..., an, phin] for n_bumps sinusoids
    """
    n = n_bumps
    x = np.ones(N)
    t = np.arange(N)
    for i in range(n):
        freq = (i + 1) / N  # Frequencies: 1/N, 2/N, ..., n/N
        amp = params[2*i]
        phase = params[2*i+1]
        x += amp * np.cos(2 * np.pi * freq * t + phase)
    return x

def _build_waveform_v3(params, n_harmonics=20):
    """
    Parametrize as DC + sum of harmonics with cos/sin coefficients.
    params: [dc, a1, b1, a2, b2, ..., an, bn]
    """
    n = n_harmonics
    x = np.full(N, params[0])
    t = np.arange(N)
    idx = 1
    for k in range(1, n + 1):
        a = params[idx]
        b = params[idx + 1]
        idx += 2
        x += a * np.cos(2 * np.pi * k * t / N) + b * np.sin(2 * np.pi * k * t / N)
    return x

def _soft_max_obj_and_grad(x, beta):
    """
    Soft-max surrogate for max|autoconv| / (sum x)^2 with gradient.
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
    
    # Compute grad_x via FFT
    # ac[m] = sum_i x[i] * x[i+m-(N-1)] (for the linear autoconv)
    # d(ac[m])/dx[j] = x[j+m-(N-1)] + x[j-m+(N-1)] (with zero padding)
    
    # Center grad_ac
    h = np.zeros(2 * N - 1)
    h[N-1 : 2*N-1] = grad_ac[0 : N]
    h[0 : N-1] = grad_ac[2*N-2 : N-1 : -1]
    
    L_corr = 1
    while L_corr < 4 * N:
        L_corr *= 2
    
    X_f = np.fft.rfft(x, n=L_corr)
    H_f = np.fft.rfft(h, n=L_corr)
    
    corr1 = np.fft.irfft(np.conj(H_f) * X_f, n=L_corr)[:N]
    
    h_rev = h[::-1]
    H_rev_f = np.fft.rfft(h_rev, n=L_corr)
    corr2 = np.fft.irfft(np.conj(H_rev_f) * X_f, n=L_corr)[:N]
    
    grad_x = corr1 + corr2
    
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10, np.zeros(N)
    
    val = soft_val / (s * s)
    grad = grad_x / (s * s) - 2.0 * soft_val * np.ones(N) / (s * s * s)
    
    return val, grad

def _optimize_continuous(params, build_fn, beta=50.0, maxiter=200):
    """
    Optimize continuous parameters using L-BFGS-B.
    """
    def obj_and_grad(p):
        x = build_fn(p)
        return _soft_max_obj_and_grad(x, beta)
    
    try:
        result = minimize(obj_and_grad, params, method='L-BFGS-B',
                          jac=True, options={'maxiter': maxiter, 'gtol': 1e-8, 'ftol': 1e-10})
        return result.x, result.fun
    except Exception:
        return params, _objective_only(build_fn(params))

def construct():
    best_x = np.ones(N)
    best_val = _objective_only(best_x)
    
    if _check_deadline():
        return best_x
    
    rng = np.random.default_rng(42)
    
    # Strategy 1: Optimize sinusoidal parameterization (v3)
    # This is a smooth, well-conditioned parameter space
    for n_harmonics in [10, 20, 30]:
        if _check_deadline():
            break
        
        n_params = 1 + 2 * n_harmonics
        for restart in range(3):
            if _check_deadline():
                break
            
            if restart == 0:
                params0 = np.zeros(n_params)
                params0[0] = 1.0  # DC = 1
            else:
                params0 = rng.normal(0, 0.1, size=n_params)
                params0[0] = 1.0
            
            params_opt, val = _optimize_continuous(
                params0, 
                lambda p, nh=n_harmonics: _build_waveform_v3(p, nh),
                beta=50.0,
                maxiter=150
            )
            
            x_opt = _build_waveform_v3(params_opt, n_harmonics)
            s = np.sum(x_opt)
            if s < 0:
                x_opt = -x_opt
            val = _objective_only(x_opt)
            
            if val < best_val:
                best_val = val
                best_x = x_opt.copy()
    
    if _check_deadline():
        return best_x
    
    # Strategy 2: Direct optimization in x-space from the best continuous solution
    # Use the smooth solution as starting point for direct L-BFGS
    for beta in [30.0, 60.0, 100.0]:
        if _check_deadline():
            break
        
        def obj_and_grad(x, _beta=beta):
            return _soft_max_obj_and_grad(x, _beta)
        
        try:
            result = minimize(obj_and_grad, best_x, method='L-BFGS-B',
                              jac=True, options={'maxiter': 200, 'gtol': 1e-9, 'ftol': 1e-11})
            if result.fun < best_val:
                best_val = result.fun
                best_x = result.x.copy()
        except Exception:
            pass
    
    if _check_deadline():
        return best_x
    
    # Strategy 3: Multi-start with diverse smooth initializations
    t = np.linspace(0, 1, N)
    
    diverse_inits = []
    for freq in [1, 2, 3, 4, 5, 7, 10]:
        for amp in [0.2, 0.4, 0.6]:
            diverse_inits.append(1.0 + amp * np.sin(2 * np.pi * freq * t))
            diverse_inits.append(1.0 + amp * np.cos(2 * np.pi * freq * t))
    
    # Add Gaussian bump initializations
    for center in [0.25, 0.5, 0.75]:
        for width in [0.1, 0.2, 0.3]:
            x_init = np.exp(-((t - center)**2) / (2 * width**2))
            diverse_inits.append(1.0 + 0.5 * x_init)
    
    for x0 in diverse_inits:
        if _check_deadline():
            break
        
        x_start = x0 + rng.normal(0, 0.01, size=N)
        
        for beta in [40.0, 80.0]:
            if _check_deadline():
                break
            
            def obj_and_grad(x, _beta=beta):
                return _soft_max_obj_and_grad(x, _beta)
            
            try:
                result = minimize(obj_and_grad, x_start, method='L-BFGS-B',
                                  jac=True, options={'maxiter': 100, 'gtol': 1e-8, 'ftol': 1e-10})
                if result.fun < best_val:
                    best_val = result.fun
                    best_x = result.x.copy()
            except Exception:
                pass
    
    if _check_deadline():
        return best_x
    
    # Strategy 4: Coordinate descent / local search refinement
    step_size = 0.003
    no_improve = 0
    max_no_improve = 3000
    
    while no_improve < max_no_improve and not _check_deadline():
        n_perturb = rng.integers(1, 20)
        indices = rng.choice(N, size=n_perturb, replace=False)
        perturbation = np.zeros(N)
        perturbation[indices] = rng.normal(0, step_size, size=n_perturb)
        x_trial = best_x + perturbation
        val = _objective_only(x_trial)
        if val < best_val:
            best_val = val
            best_x = x_trial.copy()
            step_size = max(0.0003, step_size * 0.95)
            no_improve = 0
        else:
            no_improve += 1
            step_size = min(0.01, step_size * 1.01)
    
    # Strategy 5: Try sign-flip local search on the optimized solution
    # The task allows signed step functions
    if not _check_deadline():
        # Try flipping subsets of signs
        for _ in range(200):
            if _check_deadline():
                break
            # Random sign flips on a subset
            flip_mask = rng.random(N) < 0.1
            x_trial = best_x * np.where(flip_mask, -1, 1)
            val = _objective_only(x_trial)
            if val < best_val:
                best_val = val
                best_x = x_trial.copy()
    
    # Final high-beta polish
    if not _check_deadline():
        for beta in [200.0, 400.0]:
            if _check_deadline():
                break
            def obj_and_grad(x, _beta=beta):
                return _soft_max_obj_and_grad(x, _beta)
            try:
                result = minimize(obj_and_grad, best_x, method='L-BFGS-B',
                                  jac=True, options={'maxiter': 300, 'gtol': 1e-11, 'ftol': 1e-13})
                if result.fun < best_val:
                    best_val = result.fun
                    best_x = result.x.copy()
            except Exception:
                pass
    
    return best_x
