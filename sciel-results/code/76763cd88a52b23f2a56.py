import numpy as np
import time
import scipy.optimize as opt

N = 600
W = 0.5 / N
X = np.linspace(-0.25, 0.25, N, endpoint=False)

# Fine grid parameters for accurate metric evaluation
# The autoconvolution of a step function is piecewise quadratic? 
# Actually, convolution of two piecewise constant functions is piecewise quadratic.
# To capture the peak accurately, we need a fine grid.
FINE_N = 8000
X_FINE = np.linspace(-0.25, 0.25, FINE_N, endpoint=False)
DX = 0.5 / FINE_N
# Map fine grid points to the original step indices
# Each fine point x_f falls into interval i = floor((x_f + 0.25) / W)
CELL = np.clip(((X_FINE + 0.25) / W).astype(np.int64), 0, N - 1)
N_FFT = 1 << (2 * FINE_N - 1).bit_length()

def ratio_from_h(h):
    """
    Compute the normalized max autoconvolution ratio using a fine grid.
    h: array of shape (N,) with non-negative step heights.
    """
    h = np.asarray(h, dtype=np.float64)
    if h.size != N:
        h = np.resize(h, N)
    h = np.clip(h, 0.0, None)
    
    # Check if zero
    if np.sum(h) < 1e-15:
        return float('inf')
        
    # Upsample h to fine grid by repeating each value
    f = h[CELL]
    
    # Compute autoconvolution via FFT
    F = np.fft.rfft(f, N_FFT)
    conv = np.fft.irfft(F * F, n=N_FFT)
    
    # The linear convolution length is 2*FINE_N - 1.
    # irfft gives a circular convolution of size N_FFT.
    # Since N_FFT >= 2*FINE_N - 1, the first 2*FINE_N - 1 samples are the linear convolution.
    # However, due to periodicity, the tail might wrap around if N_FFT is not large enough.
    # N_FFT is a power of 2 >= 2*FINE_N - 1, so no wrap-around.
    
    # Scale by DX to approximate the integral
    conv_int = conv[:2 * FINE_N - 1] * DX
    
    # L2 norm squared (integral of h^2)
    # sum(f^2) * DX approximates ∫ h^2
    l2_sq = np.sum(f * f) * DX
    
    if l2_sq < 1e-15:
        return float('inf')
        
    peak = np.max(conv_int)
    return peak / l2_sq

# Precompute Fourier basis
K_MAX = 30
COS_BASIS = np.zeros((N, K_MAX))
SIN_BASIS = np.zeros((N, K_MAX))
for k in range(1, K_MAX + 1):
    COS_BASIS[:, k-1] = np.cos(4 * np.pi * k * X)
    SIN_BASIS[:, k-1] = np.sin(4 * np.pi * k * X)

def fourier_to_h_fast(c0, a, b):
    """
    Reconstruct h from Fourier coefficients.
    c0: scalar
    a: array of shape (K_MAX,)
    b: array of shape (K_MAX,)
    """
    h = c0 + a @ COS_BASIS.T + b @ SIN_BASIS.T
    return np.clip(h, 0.0, None)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Initial best: Uniform
    best_h = np.ones(N)
    best_val = ratio_from_h(best_h)
    
    # Phase 1: Fourier Domain Optimization
    # Parameters: [c0, a0...aK-1, b0...bK-1]
    # Total dimension: 1 + 2*K_MAX
    
    def objective_fourier(params):
        c0 = params[0]
        a = params[1:1+K_MAX]
        b = params[1+K_MAX:1+2*K_MAX]
        h = fourier_to_h_fast(c0, a, b)
        return ratio_from_h(h)
    
    rng = np.random.default_rng(42)
    
    best_fourier_h = best_h.copy()
    best_fourier_val = best_val
    
    # Try several random initializations in Fourier space
    n_starts = 15
    for i in range(n_starts):
        if time.time() > deadline - 40.0:
            break
            
        # Initialize c0 to be positive and large enough to keep h non-negative mostly
        # If c0 is too small, clipping becomes very active and non-smooth.
        # A good starting point is a Gaussian-like shape, which has significant low-frequency content.
        
        # Strategy: Random low-frequency perturbations around a base shape
        # Base shape: Uniform (c0=1, others 0) or Gaussian-like
        
        c0_init = 1.0
        a_init = np.zeros(K_MAX)
        b_init = np.zeros(K_MAX)
        
        # Add random perturbations, decaying with frequency
        for k in range(K_MAX):
            scale = 0.5 / (k + 1)
            a_init[k] = rng.normal(0, scale)
            b_init[k] = rng.normal(0, scale)
            
        x0 = np.concatenate([[c0_init], a_init, b_init])
        
        try:
            # Nelder-Mead is robust for this low-dimensional, non-smooth (due to clip) objective
            res = opt.minimize(objective_fourier, x0, method='Nelder-Mead', 
                               options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True})
            
            if res.fun < best_fourier_val:
                h_opt = fourier_to_h_fast(res.x[0], res.x[1:1+K_MAX], res.x[1+K_MAX:1+2*K_MAX])
                v_opt = ratio_from_h(h_opt)
                if v_opt < best_fourier_val:
                    best_fourier_val = v_opt
                    best_fourier_h = h_opt
        except Exception:
            pass
    
    if best_fourier_val < best_val:
        best_val = best_fourier_val
        best_h = best_fourier_h

    # Phase 2: Local Refinement in Spatial Domain
    # Use the best Fourier result as a starting point for spatial local search.
    # We can use L-BFGS-B on a smoothed objective in the spatial domain to polish.
    
    if time.time() < deadline - 10.0:
        current_h = best_h.copy()
        current_val = best_val
        
        # Smooth max approximation parameters
        beta = 50.0
        
        def smooth_spatial_obj(y):
            # y are the log-heights or direct heights?
            # Let's use direct heights but ensure non-negativity via softplus or just clip?
            # L-BFGS-B supports bounds. We can optimize h directly with bounds [0, inf].
            # But the objective has a max, which is non-smooth.
            # Use LogSumExp for smooth max.
            h = y
            # Upsample
            f = h[CELL]
            F = np.fft.rfft(f, N_FFT)
            conv = np.fft.irfft(F * F, n=N_FFT)
            conv_int = conv[:2 * FINE_N - 1] * DX
            
            # Smooth max of conv_int
            # Note: conv_int can be negative? No, h>=0 => h*h >= 0.
            max_conv = np.log(np.sum(np.exp(beta * conv_int))) / beta
            
            l2_sq = np.sum(f * f) * DX
            if l2_sq < 1e-15:
                return 1e12
            return max_conv / l2_sq

        # Use L-BFGS-B with bounds
        bounds = [(0, None) for _ in range(N)]
        
        try:
            res = opt.minimize(smooth_spatial_obj, current_h, method='L-BFGS-B', 
                               bounds=bounds, options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8})
            
            if res.fun < current_val:
                # Verify with true metric
                h_opt = np.clip(res.x, 0, None)
                v_opt = ratio_from_h(h_opt)
                if v_opt < best_val:
                    best_val = v_opt
                    best_h = h_opt
        except Exception:
            pass

    return list(best_h)
