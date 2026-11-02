import numpy as np
import time
from scipy.signal import fftconvolve
from scipy.optimize import minimize, Bounds

N = 400
DEADLINE = 145.0  # seconds, with safety margin

def _autoconv_fft(x):
    """Compute autoconvolution via FFT (full mode)."""
    n = len(x)
    size = 2 * n - 1
    # Use next power of 2 for speed
    fft_size = 1
    while fft_size < size:
        fft_size *= 2
    X = np.fft.rfft(x, n=fft_size)
    conv = np.fft.irfft(X * X, n=fft_size)[:size]
    return conv

def _compute_c3prime(x):
    """Compute C3' = max(|x*x|) / (sum(x))^2"""
    s = np.sum(x)
    if abs(s) < 1e-15:
        return np.inf
    conv = _autoconv_fft(x)
    return np.max(np.abs(conv)) / (s * s)

def _soft_max_and_grad(x, beta):
    """
    Compute soft-max surrogate for max|ac| and its gradient.
    
    ac[k] = sum_i x[i]*x[k-i]
    d/dx[n] ac[k] = x[k-n] + x[k-n+N] (with appropriate boundaries)
    Actually for linear convolution: ac[k] = sum_i x[i] x[i+k] for k>=0
    d(ac[k])/dx[j] = x[j+k] (if j+k < N) + x[j-k] (if j-k >= 0)
    
    We use FFT-based cross-correlation for the gradient.
    """
    ac = _autoconv_fft(x)
    abs_ac = np.abs(ac)
    
    m = np.max(abs_ac)
    if m < 1e-15:
        return 0.0, np.zeros_like(x)
    
    exps = np.exp(beta * (abs_ac - m))
    sum_exps = np.sum(exps)
    soft_val = m + np.log(sum_exps) / beta
    
    weights = exps / sum_exps
    signs = np.sign(ac)
    signs[ac == 0] = 1.0
    grad_ac = weights * signs  # length 2N-1
    
    # Gradient: d(soft_val)/dx[j] = sum_k grad_ac[k] * d(ac[k])/dx[j]
    # d(ac[k])/dx[j] = x[j+k] if j+k < N else 0, plus x[j-k] if j-k >= 0 else 0
    # This is equivalent to: corr[j] = sum_k grad_ac[k] * x[abs(j-k)] with proper indexing
    # More precisely: for k >= 0: d(ac[k])/dx[j] = x[j+k] if j+k<N
    #                 for k < 0: d(ac[k])/dx[j] = x[j-k] if j-k>=0
    # 
    # Let's use the fact that autoconv is symmetric: ac[k] = ac[-k] for the full convolution
    # Actually ac is defined for lags -N+1 to N-1 in full mode, but we're using 0 to 2N-2
    # In our setup, ac[k] for k=0..2N-2 corresponds to lags -N+1..N-1 shifted
    # 
    # Simpler: use FFT cross-correlation
    # grad_x[j] = sum_{k=0}^{2N-2} grad_ac[k] * (x[j+k] if 0<=j+k<N else 0) * indicator(k corresponds to forward lag)
    # 
    # Let's just do it directly but vectorized:
    # For each j, grad_x[j] = sum_k grad_ac[k] * d(ac[k])/dx[j]
    # 
    # d(ac[k])/dx[j]:
    # ac[k] = sum_i x[i] x[i+k] where i ranges so that 0<=i<N and 0<=i+k<N
    # = sum_{i=max(0,-k)}^{min(N-1, N-1-k)} x[i] x[i+k]
    # 
    # d(ac[k])/dx[j] = x[j+k] if 0<=j+k<N AND 0<=j<N (j is always valid)
    #               + x[j-k] if 0<=j-k<N AND 0<=j<N
    # But wait, ac[k] = sum_i x[i] x[i+k], so:
    # d/dx[j] = x[j+k] if j is the first index and j+k is second: requires 0<=j+k<N
    #          + x[j-k] if j is the second index and j-k is first: requires 0<=j-k<N
    # 
    # So grad_x[j] = sum_k grad_ac[k] * (x[j+k]*I(0<=j+k<N) + x[j-k]*I(0<=j-k<N))
    # 
    # This can be computed as:
    # term1[j] = sum_k grad_ac[k] * x[j+k] * I(0<=j+k<N)  -> this is cross-correlation of grad_ac and x
    # term2[j] = sum_k grad_ac[k] * x[j-k] * I(0<=j-k<N)  -> this is also a correlation
    #
    # Let's use FFT:
    L_corr = 1
    while L_corr < 3 * N:
        L_corr *= 2
    
    g_padded = np.zeros(L_corr)
    g_padded[:2*N-1] = grad_ac
    
    X_f = np.fft.rfft(x, n=L_corr)
    G_f = np.fft.rfft(g_padded, n=L_corr)
    
    # term1: (g * x)[j] where g is at position k, x at j+k
    # This is convolution-like. Let's think of it as:
    # h[j] = sum_k g[k] x[j+k]
    # This is the cross-correlation of g and x, but with g indexed from 0
    # h[j] = sum_k g[k] x[j+k] = (g reversed convolved with x) ... 
    # 
    # Actually, let's just use the direct approach with FFT:
    # h1[j] = sum_{k=0}^{2N-2} g[k] * x[j+k] for 0<=j+k<N
    # This equals the cross-correlation. Let's compute:
    # corr1 = ifft(fft(g, L) * conj(fft(x, L)))
    # But the indexing is tricky. Let me just do a simpler vectorized computation.
    
    # Alternative: since N=400, we can do a semi-direct computation
    # grad_x[j] = sum_k grad_ac[k] * (x[j+k] if j+k<N else 0) + sum_k grad_ac[k] * (x[j-k] if j-k>=0 else 0)
    
    # Let's build it using numpy broadcasting for the first term
    # term1[j] = sum_k grad_ac[k] * x[j+k] * (j+k < N)
    # For fixed j, k ranges from 0 to min(2N-2, N-1-j)
    
    # This is getting complex. Let's use a simpler but still efficient approach:
    # Compute the gradient using the structure of the problem.
    
    # Actually, let's just use finite differences for a few key components or a simpler gradient
    # For now, let's use a subgradient approach: pick the argmax and compute its gradient
    
    # Find the dominant lag(s)
    max_idx = np.argmax(abs_ac)
    k_max = max_idx
    
    # d(ac[k_max])/dx[j] = x[j+k_max] if 0<=j+k_max<N + x[j-k_max] if 0<=j-k_max>=0
    grad_x = np.zeros_like(x)
    
    if 0 <= k_max < N:
        # Forward lag
        valid = np.arange(N) + k_max < N
        grad_x[valid] += grad_ac[k_max] * x[valid + k_max]
        # Backward lag (k_max > 0)
        if k_max > 0:
            valid2 = np.arange(N) - k_max >= 0
            grad_x[valid2] += grad_ac[k_max] * x[valid2 - k_max]
    else:
        # Negative lag: k_max >= N, actual lag = k_max - (2N-1) or k_max - N + 1?
        # In full mode, ac[k] for k=0..2N-2, lag = k - (N-1)
        lag = k_max - (N - 1)
        if lag > 0:
            # ac corresponds to lag -lag
            valid = np.arange(N) + lag < N
            grad_x[valid] += grad_ac[k_max] * x[valid + lag]
            valid2 = np.arange(N) - lag >= 0
            grad_x[valid2] += grad_ac[k_max] * x[valid2 - lag]
        elif lag < 0:
            # ac corresponds to lag -lag (positive)
            lag_abs = -lag
            valid = np.arange(N) + lag_abs < N
            grad_x[valid] += grad_ac[k_max] * x[valid + lag_abs]
            valid2 = np.arange(N) - lag_abs >= 0
            grad_x[valid2] += grad_ac[k_max] * x[valid2 - lag_abs]
    
    return soft_val, grad_x

def _objective_and_grad(x, beta):
    """Compute objective and gradient."""
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10, np.zeros_like(x)
    soft_val, grad_soft = _soft_max_and_grad(x, beta)
    val = soft_val / (s * s)
    grad = grad_soft / (s * s) - 2.0 * soft_val * np.ones_like(x) / (s * s * s)
    return val, grad

def _spectral_init(rng, mode=0):
    """
    Generate initial x via spectral design.
    """
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    n_freq = L_fft // 2 + 1
    
    if mode == 0:
        freqs = np.fft.rfftfreq(L_fft)
        mag = np.ones(n_freq)
        taper = np.minimum(1.0, np.abs(freqs) * L_fft / 5)
        mag *= taper
    elif mode == 1:
        freqs = np.fft.rfftfreq(L_fft)
        mag = 0.5 + 0.5 * np.cos(2 * np.pi * freqs * L_fft * 0.5)
    elif mode == 2:
        mag = 1.0 + 0.3 * rng.normal(0, 1, size=n_freq)
        mag = np.maximum(0.1, mag)
    elif mode == 3:
        freqs = np.fft.rfftfreq(L_fft)
        mag = np.exp(-20 * (freqs * L_fft / 2) ** 2)
    elif mode == 4:
        freqs = np.fft.rfftfreq(L_fft)
        cutoff = 0.4
        transition = 0.05
        mag = np.where(np.abs(freqs) < cutoff - transition, 1.0,
                       np.where(np.abs(freqs) < cutoff + transition,
                                0.5 * (1 + np.cos(np.pi * (np.abs(freqs) - cutoff) / transition)),
                                0.1))
    else:
        mag = np.ones(n_freq)
    
    phase = rng.uniform(0, 2 * np.pi, size=n_freq)
    phase[0] = 0
    
    spectrum = mag * np.exp(1j * phase)
    spectrum[0] = mag[0]
    
    x_full = np.fft.irfft(spectrum, n=L_fft)
    x = x_full[:N]
    
    s = np.sum(x)
    if s < 0:
        x = -x
    
    return x

def _spectral_phase_optimize(x0, rng, max_iters=200, lr=0.01, beta=50.0):
    """
    Optimize the phase of the spectrum while keeping magnitude fixed.
    """
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    n_freq = L_fft // 2 + 1
    
    X = np.fft.rfft(x0, n=L_fft)
    mag = np.abs(X).copy()
    phase = np.angle(X).copy()
    phase[0] = 0
    
    best_x = x0.copy()
    best_val = _compute_c3prime(x0)
    
    for it in range(max_iters):
        if time.time() - getattr(_spectral_phase_optimize, 'start_time', time.time()) > DEADLINE - 10:
            break
        
        val, grad = _objective_and_grad(best_x, beta)
        
        G = np.fft.rfft(grad, n=L_fft)
        
        X_curr = mag * np.exp(1j * phase)
        phase_grad = np.imag(X_curr * np.conj(G)) / (mag + 1e-10)
        phase_grad[0] = 0
        
        phase -= lr * phase_grad
        
        spectrum = mag * np.exp(1j * phase)
        spectrum[0] = mag[0]
        x_trial = np.fft.irfft(spectrum, n=L_fft)[:N]
        
        s = np.sum(x_trial)
        if s < 0:
            x_trial = -x_trial
            phase = -phase
            phase[0] = 0
        
        val_trial = _compute_c3prime(x_trial)
        
        if val_trial < best_val:
            best_val = val_trial
            best_x = x_trial.copy()
            if it % 20 == 0 and it > 0:
                lr = max(0.001, lr * 0.95)
        else:
            phase += lr * phase_grad
            lr = max(0.001, lr * 0.8)
    
    return best_x, best_val

def _flip_delta(x, idx):
    """
    Compute the new max|autoconv| and new sum if we flip x[idx].
    """
    x_new = x.copy()
    x_new[idx] = -x_new[idx]
    s_new = np.sum(x_new)
    if abs(s_new) < 1e-15:
        return np.inf, s_new
    conv_new = _autoconv_fft(x_new)
    return np.max(np.abs(conv_new)), s_new

def _peak_targeted_polish(x0, rng, max_iters=500):
    """
    Identify dominant autoconvolution lags and apply targeted perturbations.
    """
    best_x = x0.copy()
    best_val = _compute_c3prime(x0)
    ac = _autoconv_fft(best_x)
    abs_ac = np.abs(ac)
    
    for it in range(max_iters):
        if time.time() - getattr(_peak_targeted_polish, 'start_time', time.time()) > DEADLINE - 10:
            break
        
        if it % 10 == 0:
            ac = _autoconv_fft(best_x)
            abs_ac = np.abs(ac)
        
        top_k = 5
        sorted_idx = np.argsort(abs_ac)[::-1][:top_k]
        
        improved = False
        for k_idx in sorted_idx:
            if time.time() - getattr(_peak_targeted_polish, 'start_time', time.time()) > DEADLINE - 10:
                break
            
            lag = k_idx - (N - 1)
            
            # Try flipping a few random positions to reduce this peak
            for _ in range(3):
                idx = rng.integers(0, N)
                new_max_abs, s_new = _flip_delta(best_x, idx)
                if abs(s_new) < 1e-15:
                    continue
                new_score = new_max_abs / (s_new * s_new)
                if new_score < best_val:
                    best_x[idx] = -best_x[idx]
                    best_val = new_score
                    improved = True
                    break
        
        if not improved and it > 100:
            break
    
    return best_x, best_val

def _continuous_relaxation(x_init, start_time, max_iter=100):
    """
    Continuous relaxation: optimize amplitudes freely using L-BFGS, then threshold.
    """
    n = len(x_init)
    size = 2 * n - 1
    beta = 30.0
    
    def objective(a):
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 5:
            return 1e10
        s = np.sum(a)
        if abs(s) < 1e-10:
            return 1e10
        A = np.fft.rfft(a, n=size)
        conv = np.fft.irfft(A * A, n=size)
        max_abs = np.max(np.abs(conv))
        return max_abs / (s * s)
    
    def objective_grad(a):
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 5:
            return 1e10, np.zeros_like(a)
        s = np.sum(a)
        if abs(s) < 1e-10:
            return 1e10, np.zeros_like(a)
        soft_val, grad_soft = _soft_max_and_grad(a, beta)
        val = soft_val / (s * s)
        grad = grad_soft / (s * s) - 2.0 * soft_val * np.ones_like(a) / (s * s * s)
        return val, grad
    
    a0 = x_init.astype(float)
    
    try:
        result = minimize(objective_grad, a0, method='L-BFGS-B',
                          jac=True,
                          options={'maxiter': max_iter, 'gtol': 1e-4, 'ftol': 1e-6,
                                   'maxls': 50})
        
        if result.fun < objective(a0):
            a_opt = result.x
            x_thresholded = np.sign(a_opt)
            x_thresholded[x_thresholded == 0] = 1
            return x_thresholded
    except Exception:
        pass
    
    # Fallback: Nelder-Mead
    try:
        result = minimize(objective, a0, method='Nelder-Mead',
                          options={'maxiter': 50, 'xatol': 0.01, 'fatol': 1e-6,
                                   'adaptive': True})
        if result.fun < objective(a0):
            a_opt = result.x
            x_thresholded = np.sign(a_opt)
            x_thresholded[x_thresholded == 0] = 1
            return x_thresholded
    except Exception:
        pass
    
    return x_init.copy()

def construct():
    """
    Constructs a signed step function (±1 values) minimizing C3'.
    Uses spectral phase design + continuous relaxation + discrete polish.
    """
    start_time = time.time()
    _spectral_phase_optimize.start_time = start_time
    _peak_targeted_polish.start_time = start_time
    
    rng = np.random.default_rng(42)
    
    best_x = np.ones(N)
    best_score = _compute_c3prime(best_x)
    
    def _time_ok(margin=5.0):
        return (time.time() - start_time) < (DEADLINE - margin)
    
    # Phase 1: Spectral phase optimization with multiple initializations
    for mode in range(5):
        if not _time_ok(30.0):
            break
        
        x_spectral = _spectral_init(rng, mode=mode)
        x_opt, score_opt = _spectral_phase_optimize(x_spectral, rng, max_iters=100, lr=0.01, beta=50.0)
        
        if score_opt < best_score:
            best_score = score_opt
            best_x = x_opt.copy()
        
        # Threshold and use as starting point
        x_thresh = np.sign(x_opt)
        x_thresh[x_thresh == 0] = 1
        score_thresh = _compute_c3prime(x_thresh)
        if score_thresh < best_score:
            best_score = score_thresh
            best_x = x_thresh.copy()
    
    # Phase 2: Multi-start discrete search from diverse initializations
    n_starts = 4
    for start in range(n_starts):
        if not _time_ok(20.0):
            break
        
        if start == 0:
            x = best_x.copy()
        elif start == 1:
            x = np.ones(N)
            x[rng.choice(N, size=N//4, replace=False)] = -1
        elif start == 2:
            x = np.ones(N)
            x[rng.choice(N, size=N//2, replace=False)] = -1
        else:
            x = rng.choice([-1, 1], size=N)
        
        s = np.sum(x)
        if abs(s) < 1e-15:
            continue
        conv = _autoconv_fft(x)
        max_abs = np.max(np.abs(conv))
        score = max_abs / (s * s)
        
        if score < best_score:
            best_score = score
            best_x = x.copy()
        
        # Simulated annealing local search
        T = 0.3
        T_min = 0.01
        cooling = 0.9995
        no_improve = 0
        max_no_improve = 1000
        
        for it in range(20000):
            if not _time_ok(5.0):
                break
            if no_improve > max_no_improve:
                break
            
            idx = rng.integers(0, N)
            new_max_abs, s_new = _flip_delta(x, idx)
            
            if abs(s_new) < 1e-15:
                no_improve += 1
                continue
            
            new_score = new_max_abs / (s_new * s_new)
            delta = new_score - score
            
            if delta < 0:
                x[idx] = -x[idx]
                score = new_score
                no_improve = 0
                if score < best_score:
                    best_score = score
                    best_x = x.copy()
            else:
                if np.log(rng.random()) > -delta / T:
                    x[idx] = -x[idx]
                    score = new_score
                no_improve += 1
            
            T = max(T * cooling, T_min)
        
        # Peak-targeted polish
        if _time_ok(10.0):
            x_polished, score_polished = _peak_targeted_polish(x, rng, max_iters=200)
            if score_polished < best_score:
                best_score = score_polished
                best_x = x_polished.copy()
        
        # Greedy polish
        improved = True
        while improved and _time_ok(5.0):
            improved = False
            for idx in range(N):
                if not _time_ok(5.0):
                    break
                new_max_abs, s_new = _flip_delta(x, idx)
                if abs(s_new) < 1e-15:
                    continue
                new_score = new_max_abs / (s_new * s_new)
                if new_score < score:
                    x[idx] = -x[idx]
                    score = new_score
                    improved = True
                    if score < best_score:
                        best_score = score
                        best_x = x.copy()
    
    # Phase 3: Continuous relaxation polish on best solution
    if _time_ok(25.0):
        x_relaxed = _continuous_relaxation(best_x, start_time, max_iter=80)
        score_relaxed = _compute_c3prime(x_relaxed)
        if score_relaxed < best_score:
            best_score = score_relaxed
            best_x = x_relaxed.copy()
        
        # Polish the relaxed result
        x = best_x.copy()
        s = np.sum(x)
        if abs(s) > 1e-15:
            conv = _autoconv_fft(x)
            max_abs = np.max(np.abs(conv))
            score = max_abs / (s * s)
            
            # Peak-targeted polish
            if _time_ok(10.0):
                x_polished, score_polished = _peak_targeted_polish(x, rng, max_iters=200)
                if score_polished < best_score:
                    best_score = score_polished
                    best_x = x_polished.copy()
            
            # Greedy polish
            improved = True
            while improved and _time_ok(5.0):
                improved = False
                for idx in range(N):
                    if not _time_ok(5.0):
                        break
                    new_max_abs, s_new = _flip_delta(x, idx)
                    if abs(s_new) < 1e-15:
                        continue
                    new_score = new_max_abs / (s_new * s_new)
                    if new_score < score:
                        x[idx] = -x[idx]
                        score = new_score
                        improved = True
                        if score < best_score:
                            best_score = score
                            best_x = x.copy()
    
    # Final safety: ensure we return a valid ±1 sequence
    best_x = np.sign(best_x)
    best_x[best_x == 0] = 1
    
    return best_x
