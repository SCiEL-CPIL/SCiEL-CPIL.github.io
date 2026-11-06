"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_t0 = time.time()
_DEADLINE = 145.0  # Safety margin before 150s kill

def _value(h):
    """Compute the exact objective value."""
    h = np.asarray(h, dtype=float)
    if len(h) != N or not np.all(np.isfinite(h)):
        return np.inf
    s = float(np.sum(h))
    if s == 0.0:
        return np.inf
    conv = np.convolve(h, h)
    m = float(np.max(conv))
    return abs(2.0 * N * m / (s * s))

def _fft_full_autocorr(h):
    """Compute full autocorrelation via FFT, return the array of length 2N-1."""
    n = len(h)
    size = 2 * n
    f = np.fft.rfft(h, size)
    acf = np.fft.irfft(f * np.conj(f), size)
    # Full autocorrelation: acf[0:N] is one side, acf[N-1:N-1-(-1):1]... 
    # Actually irfft gives circular. For full: 
    # full_conv = np.concatenate([acf[:n], acf[1:n-1][::-1]])
    # But simpler: use the fact that full autocorr of h is:
    # c[k] = sum_i h[i]*h[i+k] for k=0..N-1 (positive lags)
    # and c[-k] = c[k]
    # The full convolve np.convolve(h,h) has length 2N-1
    # acf from FFT of length 2N gives circular correlation
    # For zero-padded to 2N, the circular correlation equals full correlation
    # acf[0] = sum h^2, acf[1] = sum h[i]h[i+1], ..., acf[N-1] = sum h[i]h[i+N-1]
    # The full convolve: index 0..N-1 is c[N-1]..c[0], index N..2N-2 is c[1]..c[N-1]
    # So max of full convolve = max(acf[0:N]) since acf is symmetric
    return float(np.max(acf[:n]))

def _fast_value(h):
    """Fast value using FFT."""
    s = float(np.sum(h))
    if s == 0.0 or not np.all(np.isfinite(h)):
        return np.inf
    m = _fft_full_autocorr(h)
    return abs(2.0 * N * m / (s * s))

def construct():
    global _t0
    _t0 = time.time()
    
    def deadline_ok():
        return (time.time() - _t0) < _DEADLINE
    
    best_h = np.ones(N)
    best_v = _value(best_h)  # 2.0
    
    def update_best(h):
        nonlocal best_h, best_v
        h = np.asarray(h, dtype=float)
        if not np.all(np.isfinite(h)):
            return
        s = np.sum(h)
        if s == 0:
            return
        v = _fast_value(h)
        if v < best_v - 1e-12:
            best_v = v
            best_h = h.copy()
    
    # Strategy: Optimize in a reduced parameter space
    # Use symmetric h: h[i] = h[N-1-i]
    # This gives N/2 = 200 free parameters
    # Also try: h as sum of a few cosine/sine modes (very low dim)
    
    half_N = N // 2
    
    # Method 1: Low-dimensional Fourier parameterization
    # h[i] = a0 + sum_k (a_k cos(2*pi*k*i/N) + b_k sin(2*pi*k*i/N))
    # For symmetric h, only cosine terms
    # Try up to K harmonics
    
    i = np.arange(N)
    
    def _obj_fourier(params, K):
        """params: [a0, a1, b1, a2, b2, ..., aK, bK]"""
        h = np.full(N, params[0])
        for k in range(1, K + 1):
            a = params[2*k - 1]
            b = params[2*k]
            h += a * np.cos(2 * np.pi * k * i / N) + b * np.sin(2 * np.pi * k * i / N)
        s = np.sum(h)
        if abs(s) < 1e-10:
            return 1e10
        m = _fft_full_autocorr(h)
        return abs(2.0 * N * m / (s * s))
    
    # Try different numbers of harmonics
    for K in [1, 2, 3, 4, 5]:
        if not deadline_ok():
            break
        num_params = 1 + 2 * K
        # Initial: a0=1, rest small random
        rng = np.random.default_rng(42 + K)
        for trial in range(3):
            if not deadline_ok():
                break
            init = np.zeros(num_params)
            init[0] = 1.0
            if trial > 0:
                init[1:] = rng.standard_normal(num_params - 1) * 0.1 / (1 + trial)
            try:
                res = minimize(
                    _obj_fourier,
                    init,
                    args=(K,),
                    method='L-BFGS-B',
                    bounds=[(-10, 10)] * num_params,
                    options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                if res.x is not None:
                    h = np.full(N, res.x[0])
                    for k in range(1, K + 1):
                        a = res.x[2*k - 1]
                        b = res.x[2*k]
                        h += a * np.cos(2 * np.pi * k * i / N) + b * np.sin(2 * np.pi * k * i / N)
                    update_best(h)
            except Exception:
                pass
    
    # Method 2: Symmetric half-vector optimization
    # h[i] = h[N-1-i], optimize first half
    def _obj_sym(x):
        h = np.zeros(N)
        h[:half_N] = x
        h[half_N:] = x[::-1]
        s = np.sum(h)
        if abs(s) < 1e-10:
            return 1e10
        m = _fft_full_autocorr(h)
        return abs(2.0 * N * m / (s * s))
    
    # Initial guesses for symmetric half
    guesses = []
    t = np.linspace(0, 1, half_N)
    guesses.append(1 - t)                    # Triangular
    guesses.append(1 - t**2)                 # Parabolic
    guesses.append(np.cos(np.pi * t / 2))    # Cosine taper
    guesses.append(1 + 0.5 * np.cos(2 * np.pi * t))  # Bump
    rng = np.random.default_rng(99)
    guesses.append(rng.random(half_N) * 0.5 + 0.25)  # Random positive
    
    for gi, g in enumerate(guesses):
        if not deadline_ok():
            break
        # Normalize to have sum ~ N
        s = np.sum(g)
        if s > 0:
            g = g * N / s
        
        try:
            res = minimize(
                _obj_sym,
                g,
                method='L-BFGS-B',
                bounds=[(-5, 5)] * half_N,
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            if res.x is not None:
                h = np.zeros(N)
                h[:half_N] = res.x
                h[half_N:] = res.x[::-1]
                update_best(h)
        except Exception:
            pass
    
    # Method 3: Full-dimensional coordinate descent with FFT
    if deadline_ok():
        h = best_h.copy()
        improved = True
        while improved and deadline_ok():
            improved = False
            for i in range(N):
                if not deadline_ok():
                    break
                old = h[i]
                for delta in [0.01, -0.01, 0.05, -0.05, 0.1, -0.1]:
                    h[i] = old + delta
                    v = _fast_value(h)
                    if v < best_v - 1e-12:
                        best_v = v
                        best_h = h.copy()
                        improved = True
                        break
                h[i] = old
    
    # Final polish with Nelder-Mead on best symmetric half
    if deadline_ok():
        best_half = best_h[:half_N]
        try:
            res_nm = minimize(
                _obj_sym,
                best_half,
                method='Nelder-Mead',
                options={'xatol': 1e-8, 'fatol': 1e-10, 'maxiter': 500, 'maxfev': 2000}
            )
            if res_nm.x is not None:
                h = np.zeros(N)
                h[:half_N] = res_nm.x
                h[half_N:] = res_nm.x[::-1]
                update_best(h)
        except Exception:
            pass
    
    # Final validation with exact convolve
    s = np.sum(best_h)
    if s == 0 or not np.all(np.isfinite(best_h)):
        return np.ones(N)
    
    v_final = _value(best_h)
    if v_final < 2.0 - 1e-12:
        return best_h
    else:
        return np.ones(N)
# EVOLVE-BLOCK-END
