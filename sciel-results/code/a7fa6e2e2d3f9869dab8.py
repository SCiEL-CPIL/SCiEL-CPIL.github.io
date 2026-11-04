"""Second autocorrelation inequality: maximize ||h||^2_2 / (||h||_1 * ||h*h||_inf).

Structural approach: Represent h_k as the squared magnitude of a sparse complex
polynomial evaluated at the 50th roots of unity. Specifically, fix a small set of
non-zero polynomial coefficients (sparse support) and optimize their complex values.
This exploits the spectral structure: h = |P(omega^k)|^2 where P is a degree-49
polynomial with only a few non-zero terms. Sparse polynomials tend to produce
flatter spectral envelopes, which reduces the autoconvolution peak relative to
the L2 norm, improving the ratio.

Combined with:
- Complex parameterization for smooth non-negativity enforcement
- Smooth max (log-sum-exp) for differentiable supremum norm
- L-BFGS-B local refinement
- Multiple diverse sparse supports and random phases for restarts
- Time management with explicit deadline
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
_DEADLINE = 140.0  # safety margin before 150s kill

_start = time.time()

def _elapsed():
    return time.time() - _start

def _time_left():
    return _DEADLINE - _elapsed()

def _autoconv_peak(h):
    """Compute supremum norm of autoconvolution using FFT."""
    n_fft = 1 << (2 * N - 1).bit_length()
    H = np.fft.rfft(h, n_fft)
    conv = np.fft.irfft(H * H, n_fft)[:2 * N - 1]
    return np.max(conv)

def _ratio(h):
    """Compute ||h||^2_2 / (||h||_1 * ||h*h||_inf)."""
    h = np.asarray(h, dtype=float)
    if np.any(h < -1e-15) or np.sum(h) < 1e-15:
        return 0.0
    h = np.maximum(h, 0)
    l2sq = np.sum(h * h)
    l1 = np.sum(h)
    peak = _autoconv_peak(h)
    if l1 < 1e-15 or peak < 1e-30:
        return 0.0
    return l2sq / (l1 * peak)

def _smooth_max(x, sigma=1.0):
    """Smooth approximation of max using log-sum-exp."""
    x_max = np.max(x)
    if x_max < 1e-30:
        return 1e-30
    x_shifted = x - x_max
    return x_max + sigma * np.log(np.sum(np.exp(x_shifted / sigma))) / len(x)

def _poly_to_h(coeffs, support):
    """
    Evaluate sparse complex polynomial at N-th roots of unity and return |P|^2.
    coeffs: complex array of length len(support)
    support: integer array of exponents
    """
    # omega_k = exp(-2*pi*i*k/N) for k=0..N-1
    k = np.arange(N)
    omega = np.exp(-2j * np.pi * k / N)
    # P(omega^k) = sum_j coeffs[j] * omega^(k * support[j])
    # = sum_j coeffs[j] * exp(-2*pi*i*k*support[j]/N)
    # Build as matrix product
    # M[k, j] = omega^(k * support[j])
    exponents = np.outer(k, support)  # shape (N, len(support))
    M = np.exp(-2j * np.pi * exponents / N)
    P = M @ coeffs  # shape (N,)
    h = np.abs(P) ** 2
    return h

def _objective_sparse(params, support):
    """
    Objective to minimize: negative of the ratio.
    params: real vector of length 2*len(support) representing complex coefficients.
    support: fixed integer array of exponents.
    """
    m = len(support)
    coeffs = params[:2*m].reshape(m, 2)
    c_real = coeffs[:, 0]
    c_imag = coeffs[:, 1]
    c_complex = c_real + 1j * c_imag
    
    h = _poly_to_h(c_complex, support)
    
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    h_norm = h / l1
    
    l2sq = np.sum(h_norm * h_norm)
    
    n_fft = 1 << (2 * N - 1).bit_length()
    H = np.fft.rfft(h_norm, n_fft)
    conv = np.fft.irfft(H * H, n_fft)[:2 * N - 1]
    peak = _smooth_max(conv, sigma=0.05)
    
    if peak < 1e-30:
        return 0.0
    
    return -l2sq / peak

def _optimize_sparse(support, c0_complex, max_iters=300, maxfun=1500):
    """Run L-BFGS-B optimization on complex polynomial coefficients."""
    m = len(support)
    params0 = np.concatenate([c0_complex.real, c0_complex.imag])
    
    result = minimize(
        lambda p: _objective_sparse(p, support),
        params0,
        method='L-BFGS-B',
        options={'maxiter': max_iters, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': maxfun}
    )
    
    params_opt = result.x
    c_real = params_opt[:m]
    c_imag = params_opt[m:2*m]
    c_opt = c_real + 1j * c_imag
    
    h_opt = _poly_to_h(c_opt, support)
    h_opt = np.maximum(h_opt, 0)
    return h_opt

def _generate_supports():
    """Generate diverse sparse supports for the polynomial."""
    supports = []
    
    # Small sets of exponents
    # Try consecutive, spread out, and random patterns
    for size in [3, 4, 5, 6, 7, 8]:
        # Consecutive
        supports.append(np.arange(size))
        # Spread evenly
        supports.append(np.linspace(0, N-1, size, dtype=int))
        # Random
        for seed in [1, 2, 3]:
            rng = np.random.RandomState(seed + size * 10)
            s = np.sort(rng.choice(N, size=size, replace=False))
            supports.append(s)
    
    # Some structured patterns
    supports.append(np.array([0, 1, 2, 3, 4]))
    supports.append(np.array([0, 10, 20, 30, 40]))
    supports.append(np.array([0, 1, 2, 10, 20, 30, 40]))
    supports.append(np.array([0, 2, 4, 6, 8, 10, 12, 14]))
    supports.append(np.array([0, 5, 10, 15, 20, 25, 30, 35, 40, 45]))
    supports.append(np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))
    supports.append(np.array([0, 1, 2, 3, 10, 11, 12, 13]))
    supports.append(np.array([0, 1, 2, 3, 4, 20, 21, 22]))
    
    # Remove duplicates by converting to tuple
    unique_supports = []
    seen = set()
    for s in supports:
        key = tuple(s)
        if key not in seen:
            seen.add(key)
            unique_supports.append(s)
    
    return unique_supports

def construct():
    start_time = time.time()
    
    best_h = np.ones(N)
    best_score = _ratio(best_h)
    
    supports = _generate_supports()
    
    # Time budget per optimization attempt
    time_per_opt = 5.0
    
    for support in supports:
        elapsed = time.time() - start_time
        if elapsed + time_per_opt > _DEADLINE:
            break
        
        m = len(support)
        
        # Try multiple random initial phases/amplitudes
        for seed in [1, 2, 3, 4, 5]:
            elapsed = time.time() - start_time
            if elapsed + time_per_opt > _DEADLINE:
                break
            
            rng = np.random.RandomState(seed)
            
            # Random complex coefficients
            amp = rng.uniform(0.5, 2.0, size=m)
            phase = rng.uniform(0, 2*np.pi, size=m)
            c0 = amp * np.exp(1j * phase)
            
            try:
                h_opt = _optimize_sparse(support, c0, max_iters=200, maxfun=800)
                score = _ratio(h_opt)
                if score > best_score:
                    best_score = score
                    best_h = h_opt.copy()
            except Exception:
                pass
    
    # Also try direct complex parameterization h = |g|^2 for comparison
    elapsed = time.time() - start_time
    if elapsed < _DEADLINE - 15:
        # Generate diverse initializations for g-space
        initializations_h = []
        
        # 1. Uniform
        initializations_h.append(np.ones(N))
        
        # 2. Gaussian-like profiles
        x = np.linspace(-1, 1, N)
        for var in [0.1, 0.5, 1.0, 2.0]:
            initializations_h.append(np.exp(-x**2 / var))
        
        # 3. Bimodal
        for sep in [0.3, 0.5, 0.7]:
            initializations_h.append(np.exp(-((x - sep)**2) / 0.1) + np.exp(-((x + sep)**2) / 0.1))
        
        # 4. Cosine window
        initializations_h.append(0.5 + 0.5 * np.cos(np.pi * x))
        
        # 5. Sidon-set sparse spikes
        for sidon_size in [4, 5, 6, 7]:
            rng = np.random.RandomState(sidon_size)
            S = np.sort(rng.choice(N, size=sidon_size, replace=False))
            h_sidon = np.zeros(N)
            h_sidon[S] = 1.0
            initializations_h.append(h_sidon)
        
        # 6. Random smooth
        for seed in [10, 20, 30]:
            rng_local = np.random.RandomState(seed)
            h_rand = np.abs(rng_local.randn(N))
            kernel = np.ones(5) / 5
            h_smooth = np.convolve(h_rand, kernel, mode='same')
            initializations_h.append(h_smooth)
        
        # 7. DFT-based
        for seed in [40, 50, 60]:
            rng_local = np.random.RandomState(seed)
            z = rng_local.randn(N) + 1j * rng_local.randn(N)
            H = np.fft.fft(z)
            h_dft = np.abs(H)**2
            h_dft = h_dft / (np.max(h_dft) + 1e-15)
            initializations_h.append(h_dft)
        
        time_per_opt2 = 4.0
        
        for i, h0 in enumerate(initializations_h):
            elapsed = time.time() - start_time
            if elapsed + time_per_opt2 > _DEADLINE:
                break
            
            score0 = _ratio(h0)
            if score0 > best_score:
                best_score = score0
                best_h = h0.copy()
            
            # Convert to g-space and optimize
            g_real = np.sqrt(np.maximum(h0, 0))
            g_imag = np.zeros(N)
            g0_flat = np.concatenate([g_real, g_imag])
            
            try:
                def _obj_g(g_flat):
                    g = g_flat.reshape(N, 2)
                    h = g[:, 0]**2 + g[:, 1]**2
                    l1 = np.sum(h)
                    if l1 < 1e-15:
                        return 0.0
                    h_norm = h / l1
                    l2sq = np.sum(h_norm * h_norm)
                    n_fft = 1 << (2 * N - 1).bit_length()
                    H = np.fft.rfft(h_norm, n_fft)
                    conv = np.fft.irfft(H * H, n_fft)[:2 * N - 1]
                    peak = _smooth_max(conv, sigma=0.05)
                    if peak < 1e-30:
                        return 0.0
                    return -l2sq / peak
                
                result = minimize(
                    _obj_g,
                    g0_flat,
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 800}
                )
                g_opt = result.x.reshape(N, 2)
                h_opt = g_opt[:, 0]**2 + g_opt[:, 1]**2
                h_opt = np.maximum(h_opt, 0)
                score = _ratio(h_opt)
                if score > best_score:
                    best_score = score
                    best_h = h_opt.copy()
            except Exception:
                pass
    
    # Final polish: perturb best solution
    elapsed = time.time() - start_time
    if elapsed < _DEADLINE - 8:
        for pert_seed in [100, 200, 300]:
            elapsed = time.time() - start_time
            if elapsed + 4 > _DEADLINE:
                break
            rng_local = np.random.RandomState(pert_seed)
            
            g_best_real = np.sqrt(np.maximum(best_h, 0))
            g_best_imag = np.zeros(N)
            
            pert_real = g_best_real + rng_local.randn(N) * 0.1
            pert_imag = rng_local.randn(N) * 0.1
            
            g_pert_flat = np.concatenate([pert_real, pert_imag])
            
            try:
                def _obj_g2(g_flat):
                    g = g_flat.reshape(N, 2)
                    h = g[:, 0]**2 + g[:, 1]**2
                    l1 = np.sum(h)
                    if l1 < 1e-15:
                        return 0.0
                    h_norm = h / l1
                    l2sq = np.sum(h_norm * h_norm)
                    n_fft = 1 << (2 * N - 1).bit_length()
                    H = np.fft.rfft(h_norm, n_fft)
                    conv = np.fft.irfft(H * H, n_fft)[:2 * N - 1]
                    peak = _smooth_max(conv, sigma=0.05)
                    if peak < 1e-30:
                        return 0.0
                    return -l2sq / peak
                
                result = minimize(
                    _obj_g2,
                    g_pert_flat,
                    method='L-BFGS-B',
                    options={'maxiter': 150, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 600}
                )
                g_opt = result.x.reshape(N, 2)
                h_opt = g_opt[:, 0]**2 + g_opt[:, 1]**2
                h_opt = np.maximum(h_opt, 0)
                score = _ratio(h_opt)
                if score > best_score:
                    best_score = score
                    best_h = h_opt.copy()
            except Exception:
                pass
    
    # Ensure final result is valid
    best_h = np.maximum(best_h, 0)
    if np.sum(best_h) < 1e-15:
        best_h = np.ones(N)
    
    return list(best_h)
