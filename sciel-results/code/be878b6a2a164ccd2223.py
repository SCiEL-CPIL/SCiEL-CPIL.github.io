import numpy as np
import time
from scipy import optimize

N = 600
_CONV_LEN = 2 * N - 1
_FFT_SIZE = 1

def _fft_convolve(h):
    n = _FFT_SIZE
    f = np.fft.rfft(h, n=n)
    g = np.fft.irfft(f * f, n=n)[:_CONV_LEN]
    return g

def compute_score(h):
    s = np.sum(h)
    if s == 0:
        return 0.0
    conv = _fft_convolve(h)
    max_conv = np.max(conv)
    return max_conv / (s * s)

def smooth_score(h, beta=100.0):
    s = np.sum(h)
    if s == 0:
        return 0.0
    conv = _fft_convolve(h)
    max_c = np.max(conv)
    log_sum_exp = max_c + (1.0/beta) * np.log(np.sum(np.exp(beta * (conv - max_c))))
    return log_sum_exp / (s * s)

def _fourier_to_heights(coeffs):
    """
    Convert Fourier coefficients to time-domain step heights.
    coeffs: [c0, c1, c2, ..., cM]
    h(x) = sum_{k=0}^M c_k cos(2*pi*k*x/L)
    x is in [-0.25, 0.25], so L=0.5.
    We use discrete x for the 600 intervals.
    """
    M = len(coeffs) - 1
    w = 0.5 / N
    x = np.linspace(-0.25 + w/2, 0.25 - w/2, N)
    # Normalize x to [0, 1] for standard cos frequencies? 
    # Let's use x in [-1, 1] scaled by 2.
    # Frequency k in "cycles per interval of 0.5"
    # cos(2*pi*k*x/0.5) = cos(4*pi*k*x)
    h = np.zeros(N)
    for k in range(M + 1):
        h += coeffs[k] * np.cos(4 * np.pi * k * x)
    
    # Clip to non-negative
    h = np.maximum(h, 0.0)
    
    return h

def construct():
    global _FFT_SIZE
    start_time = time.time()
    deadline = start_time + 110.0

    # Set FFT size
    _FFT_SIZE = 1
    while _FFT_SIZE < _CONV_LEN:
        _FFT_SIZE *= 2

    best_h = None
    best_score = float('inf')

    # Initial candidates in time domain
    w = 0.5 / N
    x = np.linspace(-0.25 + w/2, 0.25 - w/2, N)
    
    candidates = []
    candidates.append(np.ones(N) / N)
    
    sigma_g = 0.1
    h_gauss = np.exp(-x**2 / (2 * sigma_g**2))
    h_gauss /= np.sum(h_gauss)
    candidates.append(h_gauss)
    
    h_u = x**2
    h_u /= np.sum(h_u)
    candidates.append(h_u)
    
    h_sin = np.sin(np.pi * x / 0.25)**2
    h_sin /= np.sum(h_sin)
    candidates.append(h_sin)

    # Evaluate initial candidates
    for h in candidates:
        s = compute_score(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()

    # Structural Change: Global Optimization on Fourier Coefficients
    # We will optimize the first M+1 cosine coefficients.
    # M=4 or 5 is a good balance.
    M = 4
    n_params = M + 1
    
    # Bounds for coefficients
    # c0 should be positive and large enough to keep h positive.
    # c_k for k>0 can be negative.
    bounds = [(0.1, 2.0)] + [(-1.0, 1.0)] * M
    
    def fourier_objective(params):
        if time.time() > deadline - 10:
            return 1.0
        try:
            h = _fourier_to_heights(params)
            s = np.sum(h)
            if s < 1e-10:
                return 1.0
            h_norm = h / s
            # Use smooth score for optimization to help DE navigate
            score = smooth_score(h_norm, beta=50.0)
            return score
        except Exception:
            return 1.0

    # Use Differential Evolution for global search
    try:
        # maxiter and popsize control the search depth
        # popsize=15 is default, maybe increase if time permits
        res = optimize.differential_evolution(
            fourier_objective, bounds,
            maxiter=50,
            popsize=10,
            tol=1e-6,
            seed=42,
            polish=False # Polish later if needed
        )
        
        h_opt = _fourier_to_heights(res.x)
        s = np.sum(h_opt)
        if s > 1e-10:
            h_opt = h_opt / s
            score_opt = compute_score(h_opt)
            if score_opt < best_score:
                best_score = score_opt
                best_h = h_opt.copy()
                
                # Polish the best solution found by DE using L-BFGS-B on the 4 coeffs
                if time.time() < deadline - 20:
                    def polish_objective(params):
                        h = _fourier_to_heights(params)
                        s = np.sum(h)
                        if s < 1e-10: return 1.0
                        h_norm = h / s
                        return smooth_score(h_norm, beta=100.0)
                    
                    res_polish = optimize.minimize(
                        polish_objective, res.x,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={'maxiter': 100}
                    )
                    h_polished = _fourier_to_heights(res_polish.x)
                    s_p = np.sum(h_polished)
                    if s_p > 1e-10:
                        h_polished = h_polished / s_p
                        score_polished = compute_score(h_polished)
                        if score_polished < best_score:
                            best_score = score_polished
                            best_h = h_polished.copy()
    except Exception:
        pass

    if best_h is None:
        best_h = np.ones(N) / N

    return best_h.tolist()
