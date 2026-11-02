import numpy as np
from scipy.optimize import minimize
import time

N = 600
NFFT = 2 * N  # FFT size for linear autoconvolution

def _autoconv(h):
    """Compute linear autoconvolution of h via FFT."""
    f = np.zeros(NFFT, dtype=np.complex128)
    f[:N] = h
    F = np.fft.fft(f)
    A = np.fft.ifft(F * np.conj(F)).real
    return A[:2*N - 1]

def _score(h):
    """Normalized max autoconvolution; constant gives ~1.0."""
    A = _autoconv(h)
    peak = np.max(A)
    total = np.sum(A)
    if total <= 0:
        return 1.0
    return peak / total * N

def _score_and_grad_freq(params):
    """
    Parameterize h via frequency-domain shaping.
    params = [log_amp, center, width] for a Gaussian in frequency,
    plus optional polynomial coefficients for fine-tuning.
    
    We use: H(k) = exp(-((k - center)^2) / (2*width^2)) * amp
    where H is the DFT of h. Then h = ifft(|H|^2) (autoconv in freq = |H|^2).
    
    Actually, let's think differently: we want to shape h such that its 
    autoconvolution is flat. The autoconvolution's DFT is |H(k)|^2.
    So if we can make |H(k)|^2 as flat as possible (subject to h >= 0),
    the autoconvolution will be close to a delta function (or constant).
    
    Better approach: parameterize h directly as a sum of a few smooth basis
    functions (cosine series) and optimize the coefficients.
    """
    # We'll use a different approach: parameterize h as a smooth function
    # using a small number of Fourier coefficients, then optimize those.
    pass

def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # safety margin before 150s kill
    
    best_h = np.ones(N)
    best_score = _score(best_h)
    
    # Strategy: Use frequency-domain parameterization.
    # The autoconvolution a = h*h has DFT |H(k)|^2.
    # We want a to be as flat as possible (minimize max/mean ratio).
    # 
    # Key insight: If we choose H(k) to be a low-pass filter shape,
    # then |H(k)|^2 will be concentrated at low frequencies, making
    # the autoconvolution a broad smooth bump. But we want to MINIMIZE
    # the peak relative to the mean.
    #
    # Alternative: parameterize h as a smooth curve using few parameters,
    # and use gradient-based optimization with an analytic gradient.
    #
    # Let's use a cosine-series parameterization:
    # h[i] = c0 + c1*cos(2*pi*i/N) + c2*cos(4*pi*i/N) + ... + cM*cos(2*M*pi*i/N)
    # with constraint h[i] >= 0. We optimize c = [c0, c1, ..., cM].
    #
    # For the gradient, we can compute it numerically or analytically.
    
    M = 15  # number of cosine terms
    
    def h_from_cos(c):
        """Reconstruct h from cosine coefficients."""
        i = np.arange(N)
        h = np.zeros(N)
        for m in range(M + 1):
            if m == 0:
                h += c[0]
            else:
                h += c[m] * np.cos(2 * np.pi * m * i / N)
        return np.maximum(h, 0.0)  # enforce nonnegativity
    
    def score_cos(c):
        h = h_from_cos(c)
        if np.all(h == 0):
            return 1e10
        return _score(h)
    
    # Initial guesses for cosine coefficients
    # Start with constant (c0=1, rest=0)
    inits = []
    
    # Constant
    c_const = np.zeros(M + 1)
    c_const[0] = 1.0
    inits.append(c_const)
    
    # Bell shape: h[i] = 1 - (i - N/2)^2 / (N/2)^2
    # Approximate with cosine series
    i = np.arange(N)
    bell = 1.0 - ((i - N/2) / (N/2))**2
    bell = np.maximum(bell, 0)
    # Project onto cosine basis
    c_bell = np.zeros(M + 1)
    c_bell[0] = np.mean(bell)
    for m in range(1, M + 1):
        c_bell[m] = 2.0 / N * np.sum(bell * np.cos(2 * np.pi * m * i / N))
    inits.append(c_bell)
    
    # Cosine bump
    cosbump = 0.5 * (1 + np.cos(2 * np.pi * i / N))
    c_cos = np.zeros(M + 1)
    c_cos[0] = 0.5
    c_cos[1] = 0.5
    inits.append(c_cos)
    
    # Gaussian-like
    sigma = N / 4
    gauss = np.exp(-((i - N/2)**2) / (2 * sigma**2))
    c_gauss = np.zeros(M + 1)
    c_gauss[0] = np.mean(gauss)
    for m in range(1, M + 1):
        c_gauss[m] = 2.0 / N * np.sum(gauss * np.cos(2 * np.pi * m * i / N))
    inits.append(c_gauss)
    
    # Random smooth initializations
    np.random.seed(42)
    for _ in range(8):
        c_rand = np.zeros(M + 1)
        c_rand[0] = 1.0
        c_rand[1:] = np.random.randn(M) * 0.2
        inits.append(c_rand)
    
    # Optimize each initialization
    for init in inits:
        if time.time() > deadline:
            break
        try:
            res = minimize(score_cos, init, method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8})
            h_res = h_from_cos(res.x)
            s_res = _score(h_res)
            if s_res < best_score:
                best_score = s_res
                best_h = h_res.copy()
        except Exception:
            continue
    
    # Local refinement using coordinate descent on individual heights
    h = best_h.copy()
    s = best_score
    max_rounds = 5
    for rnd in range(max_rounds):
        if time.time() > deadline:
            break
        improved = False
        # Shuffle order for better exploration
        idx = np.random.permutation(N)
        for i in idx:
            if time.time() > deadline:
                break
            h_orig = h[i]
            for factor in [0.5, 0.7, 0.85, 1.0, 1.15, 1.3, 1.5]:
                h[i] = h_orig * factor
                s_new = _score(h)
                if s_new < s:
                    s = s_new
                    improved = True
                else:
                    h[i] = h_orig
                    break
        if not improved:
            break
    
    if s < best_score:
        best_score = s
        best_h = h.copy()
    
    # Final smoothing pass: average with a small kernel to reduce noise
    kernel = np.ones(5) / 5.0
    h_smooth = np.convolve(best_h, kernel, mode='same')
    h_smooth = np.maximum(h_smooth, 0)
    s_smooth = _score(h_smooth)
    if s_smooth < best_score:
        best_score = s_smooth
        best_h = h_smooth.copy()
    
    return list(best_h)
