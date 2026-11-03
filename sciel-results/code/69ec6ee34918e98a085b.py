import numpy as np
import time
import scipy.optimize as opt

N = 600
W = 1.0 / 1200.0
X = np.linspace(-0.25, 0.25, N, endpoint=False)

# Precompute fine grid for fast convolution ratio
# Using a slightly higher resolution than default to ensure accuracy of max(conv)
FINE_N = 12000
X_FINE = np.linspace(-0.25, 0.25, FINE_N, endpoint=False)
DX = 0.5 / FINE_N
# Map fine grid points to coarse bins
CELL = np.clip(((X_FINE + 0.25) / W).astype(np.int64), 0, N - 1)
N_FFT = 1 << (2 * FINE_N - 1).bit_length()

# Precompute FFT of the cell mapping to speed up convolution calculation
# We want conv = irfft(fft(f) * fft(f))
# f is the values on the fine grid.
# We can precompute the indices for the FFT.

def ratio_from_h(h):
    """Compute max(autoconv) / ||h||_2^2 using FFT on fine grid."""
    h = np.asarray(h, dtype=np.float64)
    if h.size != N:
        h = np.resize(h, N)
    h = np.clip(h, 0.0, None)
    
    # Check for zero norm
    l2_coarse = np.sum(h * h)
    if l2_coarse < 1e-15:
        return float('inf')
        
    # Map to fine grid
    f = h[CELL]
    
    # Calculate L2 norm on fine grid (approximate)
    l2 = np.sum(f * f) * DX
    if l2 < 1e-15:
        return float('inf')
        
    # FFT
    F = np.fft.rfft(f, N_FFT)
    conv = np.fft.irfft(F * F, N_FFT)[:2 * FINE_N - 1] * DX
    
    # The max of the autoconvolution
    max_conv = np.max(conv)
    
    return float(max_conv / l2)

def quick_ratio(h):
    """Optimized ratio for inner loops."""
    f = h[CELL]
    F = np.fft.rfft(f, N_FFT)
    conv = np.fft.irfft(F * F, N_FFT)[:2 * FINE_N - 1] * DX
    l2 = np.sum(f * f) * DX
    if l2 < 1e-15:
        return float('inf')
    return float(np.max(conv) / l2)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Initial guess: constant function
    best_h = np.ones(N)
    best_val = ratio_from_h(best_h)
    
    def update_best(h, val):
        nonlocal best_h, best_val
        if val < best_val:
            best_val = val
            best_h = h.copy()

    rng = np.random.default_rng(42)

    # === Phase 1: Fourier-domain Global Optimization ===
    # The optimal function is likely smooth. We represent it as a sum of cosines.
    # h[n] = a0 + sum_{k=1}^{K} (a_k cos(2*pi*k*n/N) + b_k sin(2*pi*k*n/N))
    # We clip at 0 to satisfy non-negativity.
    
    N_HARM = 30  # Increase harmonics to allow for more complex shapes
    
    # Precompute cosine and sine matrices for speed
    n_idx = np.arange(N)
    cos_mat = np.zeros((N, N_HARM))
    sin_mat = np.zeros((N, N_HARM))
    for k in range(1, N_HARM + 1):
        cos_mat[:, k-1] = np.cos(2 * np.pi * k * n_idx / N)
        sin_mat[:, k-1] = np.sin(2 * np.pi * k * n_idx / N)
        
    def fourier_to_h(coeffs):
        """
        coeffs = [a0, a1, b1, a2, b2, ..., aK, bK]
        """
        a0 = coeffs[0]
        a_coeffs = coeffs[1::2]
        b_coeffs = coeffs[2::2]
        
        h = a0 + cos_mat @ a_coeffs + sin_mat @ b_coeffs
        return np.clip(h, 0.0, None)
    
    def fourier_ratio(coeffs):
        h = fourier_to_h(coeffs)
        return ratio_from_h(h)
        
    # Multi-start optimization in Fourier space
    # Using Powell's method which is robust for non-smooth functions (due to clipping)
    n_fourier_starts = 40
    
    for start in range(n_fourier_starts):
        if time.time() > deadline - 50.0:
            break
        
        # Initialize with random small coefficients
        # a0 should be positive to keep most of the function positive
        coeffs = np.zeros(2 * N_HARM + 1)
        coeffs[0] = float(rng.uniform(0.5, 2.0))
        for k in range(1, N_HARM + 1):
            # Amplitude of harmonics should be smaller than DC term
            scale = 0.5 / k 
            coeffs[2*k - 1] = float(rng.uniform(-scale, scale))
            coeffs[2*k] = float(rng.uniform(-scale, scale))
            
        try:
            # Powell is often better than Nelder-Mead for high dimensions with clipping
            result = opt.minimize(
                fourier_ratio,
                coeffs,
                method='Powell',
                options={'maxiter': 100, 'xatol': 1e-5, 'fatol': 1e-7, 'maxfev': 500}
            )
            
            h_opt = fourier_to_h(result.x)
            val_opt = result.fun
            update_best(h_opt, val_opt)
            
            # Also try Nelder-Mead on the result of Powell for further refinement
            if time.time() < deadline - 45.0:
                result2 = opt.minimize(
                    fourier_ratio,
                    result.x,
                    method='Nelder-Mead',
                    options={'maxiter': 100, 'xatol': 1e-6, 'fatol': 1e-8}
                )
                if result2.fun < val_opt:
                    h_opt2 = fourier_to_h(result2.x)
                    update_best(h_opt2, result2.fun)
                    
        except Exception:
            pass

    # === Phase 2: Spatial Domain Refinement (Coordinate Descent) ===
    # Now we refine the best solution found in the spatial domain.
    # We use a block coordinate descent approach where we update a small
    # neighborhood of indices to allow for local shape adjustments.
    
    if time.time() < deadline - 10.0:
        cur_h = best_h.copy()
        cur_val = best_val
        
        # Adaptive step size
        step_size = 0.1
        min_step = 0.001
        stagnation_count = 0
        
        max_iters = 5000
        for it in range(max_iters):
            if time.time() > deadline - 5.0:
                break
            
            # Select a random center and a small window
            center = int(rng.integers(0, N))
            window = 5
            start_idx = max(0, center - window)
            end_idx = min(N, center + window + 1)
            
            h_try = cur_h.copy()
            
            # Perturb the window
            # We try to move the values in this window slightly
            # To preserve non-negativity, we scale or shift carefully
            # Simple approach: add small random noise, clip
            noise = rng.normal(0, step_size * np.max(cur_h), size=(end_idx - start_idx))
            h_try[start_idx:end_idx] = np.clip(cur_h[start_idx:end_idx] + noise, 0, None)
            
            v = quick_ratio(h_try)
            
            if v < cur_val:
                cur_val = v
                cur_h = h_try
                update_best(cur_h, cur_val)
                stagnation_count = 0
            else:
                stagnation_count += 1
                if stagnation_count > 100:
                    # Reduce step size and reset stagnation
                    step_size *= 0.9
                    if step_size < min_step:
                        break
                    stagnation_count = 0

    # === Phase 3: Final Polishing with Gradient-Free Local Search ===
    # A final pass with very small perturbations to the entire vector
    if time.time() < deadline - 2.0:
        cur_h = best_h.copy()
        cur_val = best_val
        
        for _ in range(100):
            if time.time() > deadline - 1.0:
                break
            
            # Perturb all elements slightly
            noise = rng.normal(0, 0.01 * np.max(cur_h), size=N)
            h_try = np.clip(cur_h + noise, 0, None)
            
            v = quick_ratio(h_try)
            if v < cur_val:
                cur_val = v
                cur_h = h_try
                update_best(cur_h, cur_val)

    return list(best_h)
