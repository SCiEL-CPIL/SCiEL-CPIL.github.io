import numpy as np
import time
import scipy.optimize as opt

N = 600
W = 1.0 / 1200.0
X = np.linspace(-0.25, 0.25, N, endpoint=False)

# Precompute fine grid for fast convolution ratio
FINE_N = 8000
X_FINE = np.linspace(-0.25, 0.25, FINE_N, endpoint=False)
DX = 0.5 / FINE_N
# Map fine grid indices to coarse cell indices
# x in [-0.25, 0.25). Cell i is [x_i, x_{i+1}).
# x_i = -0.25 + i * W
# (x + 0.25) / W = i
CELL = np.clip(((X_FINE + 0.25) / W).astype(np.int64), 0, N - 1)
N_FFT = 1 << (2 * FINE_N - 1).bit_length()

# Number of Fourier coefficients to optimize
N_HARMONICS = 20

def ratio_from_h(h):
    """Compute max(autoconv) / ||h||_2^2 using FFT on fine grid."""
    h = np.asarray(h, dtype=np.float64)
    if h.size != N:
        h = np.resize(h, N)
    h = np.clip(h, 0.0, None)
    s = h.sum()
    if s < 1e-15:
        return float('inf')
    f = h[CELL]
    F = np.fft.rfft(f, N_FFT)
    conv = np.fft.irfft(F * F, N_FFT)[:2 * FINE_N - 1] * DX
    l2 = np.sum(f * f) * DX
    if l2 < 1e-15:
        return float('inf')
    return float(np.max(conv) / l2)

def fourier_to_h(coeffs):
    """Convert Fourier coefficients to step function values on N intervals."""
    # coeffs: array of shape (N_HARMONICS,) containing c0, c1, ..., c_{N_HARMONICS-1}
    # h(x) = sum_{k=0}^{N_HARMONICS-1} c_k * cos(2 * pi * k * x)
    # We use only cosines to enforce even symmetry, which is expected for the optimal solution
    # of minimizing the peak of the autoconvolution of a non-negative function on a symmetric interval.
    h = np.zeros(N)
    for k in range(N_HARMONICS):
        h += coeffs[k] * np.cos(2 * np.pi * k * X)
    return np.clip(h, 0.0, None)

def fourier_ratio(coeffs):
    """Compute the ratio from Fourier coefficients."""
    h = fourier_to_h(coeffs)
    norm = np.linalg.norm(h)
    if norm < 1e-15:
        return float('inf')
    h /= norm
    return ratio_from_h(h)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Initialize with a Gaussian centered at 0
    x_norm = np.linspace(-1.0, 1.0, N)
    best_h = np.exp(-4.0 * x_norm**2)
    best_h /= np.linalg.norm(best_h)
    best_val = ratio_from_h(best_h)
    
    # Strategy: Optimize Fourier coefficients in the frequency domain
    # This enforces global smoothness and avoids high-frequency artifacts.
    
    # Initial guess for Fourier coefficients: Fit a Gaussian
    # We'll use a simple approach: start with a few harmonics that approximate a Gaussian
    
    # Let's try to fit the initial Gaussian with Fourier coefficients
    # Using least squares on the N points
    # Design matrix: 1, cos(2*pi*k*x) for k=1..N_HARMONICS-1
    # Since we want even symmetry, we can drop sine terms.
    
    x_fit = X
    A = np.ones((N, N_HARMONICS))
    for k in range(1, N_HARMONICS):
        A[:, k] = np.cos(2 * np.pi * k * x_fit)
    
    # Least squares fit
    try:
        coeffs_init, _, _, _ = np.linalg.lstsq(A, best_h, rcond=None)
    except:
        coeffs_init = np.zeros(N_HARMONICS)
        coeffs_init[0] = 2.0 * best_h.mean()
    
    # Symmetrize the resulting h (it should be even already due to cosine basis)
    h_init = fourier_to_h(coeffs_init)
    if np.linalg.norm(h_init) > 1e-15:
        h_init /= np.linalg.norm(h_init)
    else:
        h_init = best_h.copy()
        coeffs_init = np.zeros(N_HARMONICS)
    
    best_coeffs = coeffs_init.copy()
    
    # Use Nelder-Mead for optimization
    # This is a derivative-free method, suitable for the low-dimensional Fourier space
    n_restarts = 50
    
    for restart in range(n_restarts):
        if time.time() > deadline - 5.0:
            break
        
        # For the first restart, use the fitted coefficients
        # For subsequent restarts, add random perturbations
        if restart == 0:
            x0 = best_coeffs.copy()
        else:
            # Random perturbation
            scale = 0.1 * (1 + 0.5 * np.random.randn())
            x0 = best_coeffs + scale * np.random.randn(N_HARMONICS)
            # Ensure DC term is positive to maintain non-negativity potential
            x0[0] = abs(x0[0]) + 0.1
        
        # Nelder-Mead options
        options = {
            'maxiter': 500,
            'xatol': 1e-8,
            'fatol': 1e-8,
            'adaptive': True
        }
        
        try:
            result = opt.minimize(fourier_ratio, x0, method='Nelder-Mead', options=options)
            if result.success or result.fun < best_val:
                val = fourier_ratio(result.x)
                if val < best_val:
                    best_val = val
                    best_coeffs = result.x.copy()
                    h_candidate = fourier_to_h(result.x)
                    if np.linalg.norm(h_candidate) > 1e-15:
                        h_candidate /= np.linalg.norm(h_candidate)
                        best_h = h_candidate
        except Exception as e:
            pass
    
    # Final check: ensure best_h is valid
    if np.linalg.norm(best_h) < 1e-15:
        best_h = np.exp(-4.0 * x_norm**2)
        best_h /= np.linalg.norm(best_h)
    
    return list(best_h)
