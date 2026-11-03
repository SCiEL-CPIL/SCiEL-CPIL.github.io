"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 600
START_TIME = time.time()
TIME_LIMIT = 120.0  # wall-clock limit

def _time_left():
    return TIME_LIMIT - (time.time() - START_TIME)

def _autoconv_peak(f):
    """Compute normalized maximum of autoconvolution."""
    # f is array of N step heights on [-1/4, 1/4]
    # Each interval has width 1/300
    dw = 1.0 / 300.0
    # Autoconvolution: (f*f)(x) = sum_i f[i] * f[j] * dw where |x_i - x_j| = x
    # Use FFT for efficiency
    # Pad to avoid circular convolution
    n_fft = 2 * N
    f_pad = np.zeros(n_fft)
    f_pad[:N] = f
    conv = np.real(np.fft.ifft(np.fft.fft(f_pad) ** 2)) * dw
    # The autoconvolution is defined on [-1/2, 1/2] with 2N-1 points effectively
    # conv has n_fft points; the relevant part is the linear convolution
    # For step function, the autoconvolution at grid points:
    # Actually, let's compute it more carefully.
    # f is piecewise constant: f(x) = f[i] for x in [-1/4 + i*dw, -1/4 + (i+1)*dw)
    # (f*f)(x) = ∫ f(t) f(x-t) dt
    # This is a trapezoidal/piecewise linear function
    # Using FFT: linear convolution of f with f
    f_lin = np.convolve(f, f) * dw
    # f_lin has length 2N-1, representing values at x = -1/4 + i*dw for i=0..2N-2
    # Actually the autoconvolution of a function on [-1/4,1/4] is on [-1/2, 1/2]
    # The peak is what we want to minimize
    # Normalization: divide by (sum(f)*dw)^2 = (integral of f)^2
    integral = np.sum(f) * dw
    if integral <= 0:
        return 1e10
    peak = np.max(f_lin)
    normalized = peak / (integral ** 2)
    return normalized

def _autoconv_peak_fast(f):
    """Faster computation using FFT."""
    dw = 1.0 / 300.0
    n = len(f)
    n_fft = 1
    while n_fft < 2 * n:
        n_fft *= 2
    f_fft = np.fft.fft(f, n_fft)
    conv = np.real(np.fft.ifft(f_fft * f_fft))[:2*n-1] * dw
    integral = np.sum(f) * dw
    if integral <= 0:
        return 1e10
    peak = np.max(conv)
    return peak / (integral ** 2)

# EVOLVE-BLOCK-START
def construct():
    global START_TIME
    START_TIME = time.time()
    
    best_f = np.ones(N)
    best_score = _autoconv_peak_fast(best_f)
    
    # Strategy 1: Fourier parametrization with L-BFGS-B
    # f(x) = a0 + sum_{k=1}^{K} [a_k cos(2*pi*k*x/L) + b_k sin(2*pi*k*x/L)]
    # where L = 1/2 is the domain length
    # Ensure nonnegativity by using f = exp(g) or f = max(0, g)
    # Better: f = a0 + |sum| or use softplus
    
    # Use f = base + amplitude * (Fourier terms), with constraint f >= 0
    # Or: parametrize as f = exp(g) where g is Fourier series
    
    K = 20  # number of Fourier modes
    L = 0.5  # domain length [-1/4, 1/4]
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.5/N/2  # midpoints
    
    def fourier_to_f(params):
        """Convert Fourier params to step function values."""
        a0 = params[0]
        a = params[1:1+K]
        b = params[1+K:1+2*K]
        f = np.full(N, a0)
        for k in range(1, K+1):
            f += a[k-1] * np.cos(2*np.pi*k*x/L)
            f += b[k-1] * np.sin(2*np.pi*k*x/L)
        # Ensure nonnegativity
        f = np.maximum(f, 0.0)
        return f
    
    def objective(params):
        if _time_left() < 5.0:
            return 1e10
        f = fourier_to_f(params)
        return _autoconv_peak_fast(f)
    
    # Initial params: constant function
    # f = 1 means a0 = 1, all others 0
    x0 = np.zeros(1 + 2*K)
    x0[0] = 1.0
    
    # Optimize with L-BFGS-B
    try:
        res = minimize(objective, x0, method='L-BFGS-B', 
                      options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10,
                               'maxfun': 500})
        if res.fun < best_score:
            best_f = fourier_to_f(res.x)
            best_score = res.fun
    except Exception:
        pass
    
    # Strategy 2: Direct optimization on all 600 variables
    # Use L-BFGS-B with nonnegativity constraints
    def obj_direct(f):
        if _time_left() < 3.0:
            return 1e10
        f = np.maximum(f, 0.0)
        return _autoconv_peak_fast(f)
    
    constraints = [{'type': 'ineq', 'fun': lambda f: np.min(f) - 1e-10}]
    bounds = [(0.0, None)] * N
    
    try:
        res2 = minimize(obj_direct, best_f.copy(), method='L-BFGS-B',
                       bounds=bounds, options={'maxiter': 100, 'ftol': 1e-12,
                                               'gtol': 1e-10, 'maxfun': 200})
        if res2.fun < best_score:
            best_f = np.maximum(res2.x, 0.0)
            best_score = res2.fun
    except Exception:
        pass
    
    # Strategy 3: Local search / coordinate descent polish
    # Try small perturbations
    if _time_left() > 10.0:
        step_size = 0.01
        for _ in range(50):
            if _time_left() < 5.0:
                break
            # Random coordinate updates
            for i in range(0, N, 3):  # every 3rd coordinate for speed
                if _time_left() < 5.0:
                    break
                for delta in [step_size, -step_size]:
                    old = best_f[i]
                    best_f[i] = max(0.0, old + delta)
                    new_score = _autoconv_peak_fast(best_f)
                    if new_score < best_score:
                        best_score = new_score
                    else:
                        best_f[i] = old
    
    # Strategy 4: Try different initial shapes
    # Bell-shaped, plateau, etc.
    if _time_left() > 15.0:
        shapes = [
            lambda x: np.exp(-10*(x)**2),  # Gaussian
            lambda x: np.cos(np.pi*x)**2,   # Cosine plateau
            lambda x: np.maximum(0, 1 - 4*x**2),  # Parabolic
        ]
        for shape_fn in shapes:
            if _time_left() < 10.0:
                break
            f_init = shape_fn(x)
            f_init = f_init / np.max(f_init) * np.max(best_f)  # scale to match
            try:
                res3 = minimize(obj_direct, f_init.copy(), method='L-BFGS-B',
                               bounds=bounds, options={'maxiter': 50, 'ftol': 1e-10,
                                                       'gtol': 1e-8, 'maxfun': 100})
                if res3.fun < best_score:
                    best_f = np.maximum(res3.x, 0.0)
                    best_score = res3.fun
            except Exception:
                pass
    
    # Ensure nonnegativity
    best_f = np.maximum(best_f, 0.0)
    
    return list(best_f)
# EVOLVE-BLOCK-END
