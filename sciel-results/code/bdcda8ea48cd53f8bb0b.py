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
CELL = np.clip(((X_FINE + 0.25) / W).astype(np.int64), 0, N - 1)
N_FFT = 1 << (2 * FINE_N - 1).bit_length()

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

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    best_h = np.ones(N)
    best_val = ratio_from_h(best_h)
    
    def update_best(h, val):
        nonlocal best_h, best_val
        if val < best_val:
            best_val = val
            best_h = h.copy()
    
    # === Phase 1: Fourier-domain optimization with L-BFGS-B ===
    # This is a structural change: using a gradient-based optimizer (L-BFGS-B) 
    # on Fourier coefficients instead of Nelder-Mead.
    
    N_HARM = 30  # Increase harmonics slightly, L-BFGS-B handles higher dims better
    
    # Precompute basis functions for speed
    # h[n] = a0 + sum_k (a_k cos_k[n] + b_k sin_k[n])
    # coeffs = [a0, a1, b1, ..., aN_HARM, bN_HARM]
    
    def fourier_to_h(coeffs):
        """Convert Fourier coefficients to step heights."""
        h = np.full(N, coeffs[0])
        for k in range(1, N_HARM + 1):
            a_k = coeffs[2*k - 1]
            b_k = coeffs[2*k]
            angle = 2 * np.pi * k * np.arange(N) / N
            h += a_k * np.cos(angle) + b_k * np.sin(angle)
        return np.clip(h, 0.0, None)
    
    def fourier_ratio(coeffs):
        h = fourier_to_h(coeffs)
        return ratio_from_h(h)
    
    # Multi-start Fourier optimization
    rng = np.random.default_rng(42)
    n_fourier_starts = 20
    
    # Initial guess: A simple bump
    # h(x) ~ exp(-x^2/sigma^2)
    # We can project this to Fourier coefficients or just start with a0 > 0 and small others
    
    for start in range(n_fourier_starts):
        if time.time() > deadline - 40.0:
            break
        
        # Initialize with low-amplitude random coefficients
        coeffs = np.zeros(2 * N_HARM + 1)
        coeffs[0] = float(rng.uniform(0.5, 1.5))  # DC term positive
        for k in range(1, N_HARM + 1):
            # Decay amplitude with frequency to encourage smoothness
            amp_scale = 1.0 / (1.0 + k)
            coeffs[2*k - 1] = float(rng.uniform(-0.5, 0.5) * amp_scale)
            coeffs[2*k] = float(rng.uniform(-0.5, 0.5) * amp_scale)
        
        # Use L-BFGS-B to optimize Fourier coefficients
        # Bounds: a0 > 0, others unbounded (clipping handles negativity in h)
        # Actually, to ensure h >= 0, we rely on clipping in fourier_to_h.
        # L-BFGS-B works on the coefficients.
        try:
            result = opt.minimize(
                fourier_ratio,
                coeffs,
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-5}
            )
            if result.fun < best_val:
                h_opt = fourier_to_h(result.x)
                update_best(h_opt, result.fun)
        except Exception:
            pass
    
    # === Phase 2: Spatial-domain optimization with L-BFGS-B ===
    # Direct optimization of h. 
    # To handle the max(conv) non-smoothness, we can use a soft-max approximation 
    # or just rely on the fact that L-BFGS-B is robust.
    # However, finite differences for 600 variables is 600 evals per gradient step.
    # This might be too slow. 
    # Instead, we do a "Peak Shaving" heuristic which is cheaper.
    
    if time.time() < deadline - 20.0:
        cur_h = best_h.copy()
        cur_val = best_val
        
        # Compute convolution once to find peaks
        f = cur_h[CELL]
        F = np.fft.rfft(f, N_FFT)
        conv = np.fft.irfft(F * F, N_FFT)[:2 * FINE_N - 1] * DX
        l2 = np.sum(f * f) * DX
        
        max_val = np.max(conv)
        
        # Identify indices in the fine grid where conv is close to max
        # Map back to original h indices? 
        # The convolution peak is at some lag. 
        # A simple heuristic: reduce h at indices that are "aligned" with the peak.
        
        # Let's try a simpler gradient-free refinement: 
        # Randomly perturb a few h values and accept if better.
        rng2 = np.random.default_rng(777)
        n_iter = 5000
        for it in range(n_iter):
            if time.time() > deadline - 5.0:
                break
            
            # Pick a random coordinate
            idx = int(rng2.integers(0, N))
            
            # Try a small change
            # Multiplicative step
            factor = float(rng2.uniform(0.9, 1.1))
            h_try = cur_h.copy()
            h_try[idx] = max(0.0, cur_h[idx] * factor)
            
            v = ratio_from_h(h_try)
            if v < cur_val:
                cur_val = v
                cur_h = h_try
                update_best(cur_h, cur_val)
    
    # === Phase 3: Try hand-crafted shapes ===
    if time.time() < deadline - 5.0:
        x_norm = np.linspace(-1.0, 1.0, N)
        crafted = []
        # More diverse shapes
        crafted.append(np.exp(-2.0 * x_norm**2))
        crafted.append(np.exp(-0.5 * x_norm**2))
        crafted.append(0.5 + 0.5 * np.exp(-8.0 * x_norm**2))
        crafted.append(np.exp(-20.0 * (x_norm - 0.3)**2) + np.exp(-20.0 * (x_norm + 0.3)**2))
        crafted.append(0.1 + np.exp(-30.0 * (x_norm - 0.4)**2) + np.exp(-30.0 * (x_norm + 0.4)**2))
        crafted.append(1.0 - 0.3 * np.exp(-5.0 * x_norm**2))
        crafted.append(0.5 + 0.5 * x_norm**2)
        crafted.append(0.5 + 0.5 * np.cos(np.pi * x_norm))
        # Triangular
        crafted.append(np.clip(1.0 - np.abs(x_norm), 0, None))
        # Square
        crafted.append(np.clip(1.0 - 2.0 * np.abs(x_norm), 0, None))
        
        for h in crafted:
            if time.time() > deadline - 3.0:
                break
            h = np.clip(h, 0, None)
            if h.sum() < 1e-12:
                continue
            v = ratio_from_h(h)
            update_best(h, v)

    return list(best_h)
