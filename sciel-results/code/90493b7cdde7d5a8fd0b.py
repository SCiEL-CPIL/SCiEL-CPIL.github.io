import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.fftpack import dct, idct

N = 400

def _autoconv(h):
    n = len(h)
    f = np.fft.rfft(h, n=2 * n)
    c = np.fft.irfft(f * f, n=2 * n)
    return c

def _hard(h):
    s = h.sum()
    if abs(s) < 1e-9:
        return 1e12
    conv = _autoconv(h)
    return abs(2.0 * N * np.max(conv) / (s * s))

def _soft_max_conv(h, beta):
    n = len(h)
    conv = _autoconv(h)
    m = np.max(conv)
    shifted = beta * (conv - m)
    shifted = np.clip(shifted, -500, 500)
    ez = np.exp(shifted)
    z = ez.sum()
    sm = m + np.log(z) / beta
    w = ez / z
    
    # Gradient: d(sm)/d(h_i) = sum_k w_k * d(conv_k)/d(h_i)
    # conv_k = sum_i h_i * h_{k-i+N-1} (for full conv indexed 0..2N-2)
    # Using FFT correlation for efficiency
    W = np.fft.rfft(w, n=2 * n)
    H = np.fft.rfft(h, n=2 * n)
    
    # conv_wh[k] = sum_i w[i] * h[k-i] for circular conv of length 2n
    # We need the linear correlation part
    conv_wh = np.fft.irfft(W * H, n=2 * n)
    term2 = conv_wh[0:n]
    
    # term1: sum_k w_k * h_{k - (N-1) - i} for valid indices
    # This is the correlation of w with h
    B = np.correlate(h, w, mode='full')
    term1 = B[n-1:2*n-1]
    
    grad = term1 + term2
    return sm, grad

def _obj_and_grad(h, beta):
    s = h.sum()
    if abs(s) < 1e-9:
        return 1e12, np.zeros_like(h)
    sm, g_sm = _soft_max_conv(h, beta)
    val = 2.0 * N * sm / (s * s)
    g = 2.0 * N * (g_sm * s - 2.0 * sm) / (s * s * s)
    return val, g

def _param_to_h(p):
    """Map parameter vector to h using a structured basis.
    p[0] is the baseline (mean), p[1:] are amplitudes of basis functions.
    Basis functions are designed to spread autocorrelation energy."""
    n = len(p)
    m = n - 1  # number of basis functions
    h = np.zeros(N)
    h[:] = p[0]  # baseline
    
    # Use DCT-like basis functions for the perturbation
    # These are orthogonal and spread energy across lags
    for i in range(m):
        if abs(p[i+1]) > 1e-15:
            # Use cosine basis with different frequencies
            k = i + 1
            h += p[i+1] * np.cos(np.pi * k * np.arange(N) / N)
    
    return h

def _spectral_init(rng, m_bas):
    """Generate a diverse initial point using spectral methods."""
    # Start with random low-frequency DCT coefficients
    coeffs = np.zeros(N)
    # Random amplitudes for low frequencies
    freqs = rng.integers(1, m_bas + 1, size=m_bas)
    freqs = np.unique(freqs)
    for k in freqs:
        coeffs[k] = rng.standard_normal() * 0.1
    
    h = idct(coeffs, type=2, norm='ortho')
    # Add a baseline
    h += rng.standard_normal() * 0.5
    
    return h

def construct():
    t0 = time.time()
    DEADLINE = 140.0
    
    def time_left():
        return DEADLINE - (time.time() - t0)

    rng = np.random.default_rng(42)
    
    best_h = np.ones(N)
    best_val = _hard(best_h)
    
    # Number of basis functions for parameterization
    m_bas = 80
    
    # Phase 1: Multiple random restarts with local optimization
    n_restarts = 8
    beta_start = 20.0
    
    for restart in range(n_restarts):
        if time_left() < 20:
            break
        
        # Generate diverse initial point
        if restart == 0:
            h_curr = np.ones(N)
        elif restart == 1:
            h_curr = _spectral_init(rng, m_bas)
        else:
            # Mix of random and structured
            h_curr = _spectral_init(rng, m_bas)
            h_curr += 0.5 * rng.standard_normal(N)
        
        # Normalize to have reasonable scale
        s = h_curr.sum()
        if abs(s) > 1e-9:
            h_curr = h_curr / s
        
        # Local optimization with increasing beta
        for beta in [beta_start, 50.0, 100.0, 200.0]:
            if time_left() < 15:
                break
            
            def obj(x):
                val, _ = _obj_and_grad(x, beta)
                return val
            
            def gr(x):
                _, g = _obj_and_grad(x, beta)
                return g
                
            try:
                res = minimize(obj, h_curr, jac=gr, method='L-BFGS-B',
                               options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12})
                h_new = res.x
                v = _hard(h_new)
                if v < best_val:
                    best_val = v
                    best_h = h_new.copy()
                    h_curr = h_new.copy()
                else:
                    # Keep h_curr for next beta
                    h_curr = res.x
            except Exception:
                pass
    
    # Phase 2: Fine-tuning with coordinate descent on hard objective
    h_curr = best_h.copy()
    best_local = _hard(h_curr)
    
    for _ in range(100):
        if time_left() < 5:
            break
        conv = _autoconv(h_curr)
        kmax = int(np.argmax(conv))
        lag = kmax - (N - 1)
        sg = np.zeros(N)
        for i in range(N):
            a = lag - i
            b = i - lag
            if 0 <= a < N:
                sg[i] += h_curr[a]
            if 0 <= b < N:
                sg[i] += h_curr[b]
        
        ss = h_curr.sum()
        if abs(ss) < 1e-9:
            break
        sg_centered = sg - sg.mean()
        
        for alpha in [1e-2, 5e-3, 1e-3, 5e-4, 1e-4]:
            cand = h_curr - alpha * sg_centered
            vv = _hard(cand)
            if vv < best_local:
                best_local = vv
                h_curr = cand
                if vv < best_val:
                    best_val = vv
                    best_h = cand.copy()
                break
        else:
            # No improvement, try smaller steps or break
            if alpha < 5e-5:
                break

    if not np.all(np.isfinite(best_h)):
        best_h = np.ones(N)
    if abs(best_h.sum()) < 1e-10:
        best_h = np.ones(N)

    return list(best_h)
