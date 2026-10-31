import numpy as np
from scipy.signal import windows
from scipy.optimize import minimize
import time

N = 400
MAX_FREQ = 8  # Number of sine/cosine pairs to use

def _get_max_sidelobe(f):
    """Compute the maximum absolute value of the autoconvolution sidelobes."""
    f_fft = np.fft.fft(f, n=2*N)
    ac = np.fft.ifft(f_fft * f_fft).real
    center = N - 1
    sidelobes = np.delete(ac, center)
    return np.max(np.abs(sidelobes))

def _soft_max_sidelobe(f, lam=15.0):
    """Compute a smooth approximation of the max sidelobe."""
    f_fft = np.fft.fft(f, n=2*N)
    ac = np.fft.ifft(f_fft * f_fft).real
    center = N - 1
    sidelobes = np.delete(ac, center)
    abs_sidelobes = np.abs(sidelobes)
    max_val = np.max(abs_sidelobes)
    if max_val < 1e-10:
        return 0.0
    shifted = abs_sidelobes - max_val
    return max_val + (1.0/lam) * np.log(np.sum(np.exp(lam * shifted)))

def _bandlimited_f(coeffs):
    """Construct a band-limited function from sine/cosine coefficients.
    coeffs: [a1, b1, a2, b2, ..., a_k, b_k] where each pair is amplitude and frequency.
    f(x) = sum_i a_i * cos(2*pi*f_i*x) + b_i * sin(2*pi*f_i*x)
    """
    x = np.linspace(-0.25, 0.25, N)
    f = np.zeros(N)
    for i in range(MAX_FREQ):
        a = coeffs[2*i]
        b = coeffs[2*i+1]
        freq = (i + 1)  # frequencies 1, 2, 3, ...
        f += a * np.cos(2 * np.pi * freq * x) + b * np.sin(2 * np.pi * freq * x)
    return f

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 150s kill
    
    # Objective: minimize soft max sidelobe of band-limited function
    def objective(coeffs):
        f = _bandlimited_f(coeffs)
        norm = np.linalg.norm(f)
        if norm < 1e-10:
            return 1e10
        f = f / norm
        return _soft_max_sidelobe(f, lam=15.0)
    
    best_val = float('inf')
    best_f = None
    
    # Try multiple random initializations
    np.random.seed(42)
    n_restarts = 8
    
    for r in range(n_restarts):
        if time.time() > deadline - 30:
            break
        
        # Random initialization with varying sparsity
        if r == 0:
            # Start with mostly low frequencies
            init = np.zeros(2 * MAX_FREQ)
            init[0] = 1.0  # DC-like component (freq 1 cosine)
        elif r == 1:
            # Random sparse
            init = np.zeros(2 * MAX_FREQ)
            nz = np.random.choice(MAX_FREQ, size=min(3, MAX_FREQ), replace=False)
            for idx in nz:
                init[2*idx] = np.random.randn() * 0.5
                init[2*idx+1] = np.random.randn() * 0.5
        else:
            # Random dense with decay
            init = np.zeros(2 * MAX_FREQ)
            for i in range(MAX_FREQ):
                decay = 1.0 / (i + 1)
                init[2*i] = np.random.randn() * decay
                init[2*i+1] = np.random.randn() * decay
        
        try:
            res = minimize(objective, init, method='Nelder-Mead', 
                          options={'maxiter': 2000, 'xatol': 1e-5, 'fatol': 1e-7,
                                  'adaptive': True})
            if res.fun < best_val:
                best_val = res.fun
                best_f = _bandlimited_f(res.x)
                norm = np.linalg.norm(best_f)
                if norm > 1e-10:
                    best_f = best_f / norm
                else:
                    best_f = None
        except Exception:
            pass
    
    # Fallback to Kaiser if band-limited approach failed
    if best_f is None:
        best_kaiser_val = float('inf')
        best_kaiser_f = None
        for beta in np.arange(2, 20, 1):
            if time.time() > deadline - 5:
                break
            f = windows.kaiser(N, beta)
            norm = np.linalg.norm(f)
            if norm == 0:
                continue
            f = f / norm
            val = _get_max_sidelobe(f)
            if val < best_kaiser_val:
                best_kaiser_val = val
                best_kaiser_f = f.copy()
        
        if best_kaiser_f is not None:
            return best_kaiser_f
        else:
            f = windows.kaiser(N, 10)
            return f / np.linalg.norm(f)
    
    return best_f
