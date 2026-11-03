import numpy as np
import time
import scipy.optimize as opt

N = 600
W = 1.0 / 1200.0
X = np.linspace(-0.25, 0.25, N, endpoint=False)

# Precompute for fast convolution ratio
FINE_N = 20000
X_FINE = np.linspace(-0.25, 0.25, FINE_N, endpoint=False)
DX = 0.5 / FINE_N
CELL = np.clip(((X_FINE + 0.25) / W).astype(np.int64), 0, N - 1)
N_FFT = 1 << (2 * FINE_N - 1).bit_length()

def ratio_from_h(h):
    """Compute normalized max autoconvolution ratio using FFT."""
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
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    # Initial best: uniform
    best_h = np.ones(N)
    best_val = ratio_from_h(best_h)
    
    # Strategy: Parametric Shape Search
    # We hypothesize that the optimal shape is a "bump" centered at 0.
    # We model h(x) as a scaled and shifted version of a base profile.
    # Base profiles:
    # 1. Triangle: max(0, 1 - |x|/width)
    # 2. Gaussian: exp(-x^2 / (2*width^2))
    # 3. Sinc-like (main lobe of sinc): max(0, sin(pi*x/width)/(pi*x/width))
    
    # Parameters: [center, width, height, skew]
    # center in [-0.1, 0.1]
    # width in [0.01, 0.2]
    # height in [0.1, 2.0]
    # skew in [-0.5, 0.5] (shifts the peak slightly or modifies asymmetry)
    
    def make_h(params, profile_type):
        c, w, h, s = params
        x = X - c
        # Apply skew by shifting x slightly for one side? 
        # Simpler: just use c as center.
        # Let's stick to symmetric for simplicity, but allow center shift.
        if profile_type == 0: # Triangle
            h_vals = np.maximum(0.0, 1.0 - np.abs(x) / w)
        elif profile_type == 1: # Gaussian
            h_vals = np.exp(-0.5 * (x / w) ** 2)
        elif profile_type == 2: # Sinc lobe
            # Avoid division by zero
            x_safe = np.where(np.abs(x) < 1e-9, 1e-9, x)
            h_vals = np.sin(np.pi * x_safe / w) / (np.pi * x_safe / w)
            h_vals = np.where(np.abs(x) < w, h_vals, 0.0)
            # Ensure non-negative
            h_vals = np.maximum(0.0, h_vals)
        else: # Parabolic
            h_vals = np.maximum(0.0, 1.0 - (x / w) ** 2)
            
        h_vals *= h
        return h_vals
    
    # Define objective for a specific profile type
    def objective(params, ptype):
        h = make_h(params, ptype)
        if h.sum() < 1e-12:
            return 1e6
        return ratio_from_h(h)
    
    # Search over different profile types
    profile_types = [0, 1, 2, 3]
    
    rng = np.random.default_rng(42)
    
    for ptype in profile_types:
        if time.time() > deadline - 10.0:
            break
            
        # Define bounds for [center, width, height]
        # center: [-0.1, 0.1]
        # width: [0.01, 0.2]
        # height: [0.1, 2.0]
        bounds = [(-0.1, 0.1), (0.01, 0.2), (0.1, 2.0)]
        
        # Try multiple random starting points
        n_starts = 5
        for i in range(n_starts):
            if time.time() > deadline - 5.0:
                break
            
            # Random start within bounds
            x0 = np.array([
                rng.uniform(-0.1, 0.1),
                rng.uniform(0.02, 0.15),
                rng.uniform(0.5, 1.5)
            ])
            
            # Use Nelder-Mead for unconstrained optimization (bounds handled by clipping in make_h or by penalty)
            # Since make_h clips to non-negative and defines support, we don't strictly need bounds for the optimizer
            # but keeping parameters in reasonable ranges helps convergence.
            
            try:
                res = opt.minimize(
                    lambda p: objective(p, ptype), 
                    x0, 
                    method='Nelder-Mead', 
                    options={'maxiter': 1000, 'xatol': 1e-5, 'fatol': 1e-6, 'adaptive': True}
                )
                if res.fun < best_val:
                    best_val = res.fun
                    best_h = make_h(res.x, ptype)
            except Exception:
                pass
            
            # Also try a coarse grid search if time permits, to catch global minima missed by NM
            if time.time() < deadline - 20.0:
                # Coarse grid on width and height, center=0
                widths = np.linspace(0.02, 0.2, 10)
                heights = np.linspace(0.2, 1.5, 10)
                for w in widths:
                    for hgt in heights:
                        p = [0.0, w, hgt]
                        val = objective(p, ptype)
                        if val < best_val:
                            best_val = val
                            best_h = make_h(p, ptype)
                            # If we found a new best, refine it
                            try:
                                res_ref = opt.minimize(
                                    lambda p: objective(p, ptype), 
                                    p, 
                                    method='Nelder-Mead', 
                                    options={'maxiter': 500, 'xatol': 1e-5, 'fatol': 1e-6}
                                )
                                if res_ref.fun < best_val:
                                    best_val = res_ref.fun
                                    best_h = make_h(res_ref.x, ptype)
                            except Exception:
                                pass
                        if time.time() > deadline - 5.0:
                            break
                    if time.time() > deadline - 5.0:
                        break
    
    # Final local refinement in spatial domain to break symmetry or adjust edges
    if time.time() < deadline - 5.0:
        rng2 = np.random.default_rng(12345)
        current_h = best_h.copy()
        current_val = best_val
        
        for it in range(500):
            if time.time() > deadline - 2.0:
                break
            
            # Small random perturbation
            h_try = current_h.copy()
            # Add small noise
            noise = rng2.normal(0, 0.01, N)
            h_try += noise
            h_try = np.clip(h_try, 0, None)
            
            # Optional: slight smoothing
            if rng2.random() < 0.5:
                h_try = np.convolve(h_try, np.ones(3)/3, mode='same')
                h_try = np.clip(h_try, 0, None)
            
            v = ratio_from_h(h_try)
            if v < current_val:
                current_val = v
                current_h = h_try
                if current_val < best_val:
                    best_val = current_val
                    best_h = h_try.copy()
    
    return list(best_h)
