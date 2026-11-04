"""Second autocorrelation inequality: maximize ratio of squared L2 norm to L1 norm times sup norm of autoconvolution."""
import numpy as np
from scipy.optimize import minimize
import time

N = 50
TIME_LIMIT = 115  # seconds, leave margin before 120s kill

def _compute_ratio(h):
    """Compute the ratio: ||h||_2^2 / (||h||_1 * ||h*h||_inf)
    where h*h is the autoconvolution (circular convolution for FFT-based, 
    but we use linear convolution on the support).
    
    Actually, for step heights on [-1/4, 1/4], the autoconvolution is the 
    linear convolution of the sequence with itself. The supremum norm is 
    the max of this convolution.
    
    For a sequence of N nonnegative values, the autoconvolution has 2N-1 
    elements. We compute it via FFT or direct.
    """
    h = np.asarray(h, dtype=float)
    if np.any(h < 0):
        return 0.0
    l2_sq = np.sum(h * h)
    l1 = np.sum(h)
    if l1 == 0 or l2_sq == 0:
        return 0.0
    # Autoconvolution: linear convolution of h with itself
    conv = np.convolve(h, h, mode='full')
    sup_norm = np.max(conv)
    if sup_norm <= 0:
        return 0.0
    return l2_sq / (l1 * sup_norm)

def _negative_ratio(h):
    """Negative of the ratio for minimization."""
    return -_compute_ratio(h)

def construct():
    start_time = time.time()
    
    best_h = np.ones(N)
    best_score = _compute_ratio(best_h)
    
    # Try several initializations and optimization strategies
    rng = np.random.default_rng(42)
    
    # Different initial profiles
    initializations = []
    
    # 1. Uniform (baseline)
    initializations.append(np.ones(N))
    
    # 2. Triangle/bell shape
    x = np.linspace(-1, 1, N)
    initializations.append(np.maximum(0, 1 - np.abs(x)))
    initializations.append(np.exp(-2 * x**2))
    initializations.append(np.exp(-5 * x**2))
    
    # 3. Random smooth profiles
    for _ in range(5):
        # Smooth random via low-pass
        raw = rng.random(N)
        # Apply simple smoothing
        kernel = np.ones(5) / 5
        smoothed = np.convolve(raw, kernel, mode='same')
        smoothed = np.maximum(0, smoothed)
        if np.sum(smoothed) > 0:
            initializations.append(smoothed)
    
    # 4. Sparse-like profiles
    for _ in range(3):
        h = np.zeros(N)
        num_nonzero = rng.integers(10, 25)
        idx = rng.choice(N, size=num_nonzero, replace=False)
        h[idx] = rng.uniform(0.5, 2.0, size=num_nonzero)
        initializations.append(h)
    
    # 5. DFT-based: |IDFT(z)|^2 with sparse spectrum
    # Represent h as |IDFT(Z)|^2 where Z has only K active bins
    for K in [2, 3, 4]:
        for _ in range(3):
            # Random sparse spectrum
            Z = np.zeros(N, dtype=complex)
            active_bins = rng.choice(N, size=K, replace=False)
            for b in active_bins:
                # Ensure conjugate symmetry for real output: pair b with N-b
                if b == 0:
                    Z[b] = rng.uniform(1, 5)
                elif b == N // 2 and N % 2 == 0:
                    Z[b] = rng.uniform(1, 5)
                else:
                    mag = rng.uniform(1, 5)
                    phase = rng.uniform(0, 2 * np.pi)
                    Z[b] = mag * np.exp(1j * phase)
                    Z[N - b] = np.conj(Z[b])
            
            h_dft = np.abs(np.fft.ifft(Z))**2
            # Normalize to avoid issues
            if np.sum(h_dft) > 0:
                h_dft = h_dft / np.sum(h_dft) * N  # keep similar scale
                initializations.append(h_dft)
    
    # Optimize each initialization
    for i, h0 in enumerate(initializations):
        if time.time() - start_time > TIME_LIMIT - 10:
            break
        
        h0 = np.asarray(h0, dtype=float)
        h0 = np.maximum(0, h0)
        
        # Normalize to a reasonable scale
        if np.sum(h0) > 0:
            h0 = h0 * (N / np.sum(h0))
        
        # Try L-BFGS-B with bounds
        bounds = [(0, None)] * N
        try:
            result = minimize(
                _negative_ratio,
                h0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            h_opt = np.maximum(0, result.x)
            score = _compute_ratio(h_opt)
            if score > best_score:
                best_score = score
                best_h = h_opt.copy()
        except Exception:
            pass
        
        # Also try a few random restarts from this neighborhood
        for _ in range(2):
            if time.time() - start_time > TIME_LIMIT - 10:
                break
            h_perturb = h0 + rng.normal(0, 0.1, size=N)
            h_perturb = np.maximum(0, h_perturb)
            if np.sum(h_perturb) > 0:
                h_perturb = h_perturb * (N / np.sum(h_perturb))
            try:
                result = minimize(
                    _negative_ratio,
                    h_perturb,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                h_opt = np.maximum(0, result.x)
                score = _compute_ratio(h_opt)
                if score > best_score:
                    best_score = score
                    best_h = h_opt.copy()
            except Exception:
                pass
    
    # Final: try a more global optimization using Nelder-Mead on a subset
    if time.time() - start_time < TIME_LIMIT - 15:
        # Use the best so far as starting point for a final refinement
        try:
            result = minimize(
                _negative_ratio,
                best_h,
                method='L-BFGS-B',
                bounds=[(0, None)] * N,
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            h_final = np.maximum(0, result.x)
            score = _compute_ratio(h_final)
            if score > best_score:
                best_score = score
                best_h = h_final.copy()
        except Exception:
            pass
    
    # Ensure nonnegativity and return
    best_h = np.maximum(0, best_h)
    
    # Check deadline
    if time.time() - start_time > TIME_LIMIT:
        # Return the uniform as fallback (shouldn't happen)
        return [1.0] * N
    
    return [float(x) for x in best_h]
