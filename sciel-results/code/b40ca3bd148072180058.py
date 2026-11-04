import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE = 110.0  # Safety margin before 150s kill

def _elapsed(start):
    return time.time() - start

def _time_left(start):
    return DEADLINE - _elapsed(start)

def _autoconv(h):
    """Compute discrete autoconvolution of h (length N) -> length 2N-1."""
    return np.convolve(h, h, mode='full')

def _ratio(h, beta=50.0):
    """Compute ||h||^2_2 / (||h||_1 * ||h*h||_inf) using soft-max for peak."""
    h = np.maximum(h, 0.0)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    h_norm = h / l1
    l2_sq = np.sum(h_norm ** 2)
    conv = _autoconv(h_norm)
    max_val = np.max(conv)
    if max_val <= 0:
        return 0.0
    # Soft-max approximation of max using log-sum-exp
    shifted = beta * (conv - max_val)
    max_shifted = np.max(shifted)
    exp_shifted = np.exp(shifted - max_shifted)
    log_sum_exp = max_shifted + np.log(np.sum(exp_shifted))
    soft_max = (log_sum_exp + beta * max_val) / beta
    if soft_max <= 0:
        return 0.0
    return l2_sq / soft_max

def _ratio_neg(h, beta=50.0):
    return -_ratio(h, beta)

def _dft_parameterize(z):
    """
    Parameterize heights as |DFT(z)|^2 where z is a complex vector.
    This ensures non-negativity and provides a smooth manifold.
    z has length N, we take DFT and get magnitudes squared.
    """
    H = np.fft.fft(z)
    h = np.abs(H) ** 2
    return h

def _dft_inverse(h):
    """Given h = |DFT(z)|^2, find a z (phase choice)."""
    H_mags = np.sqrt(h + 1e-30)
    H = H_mags.astype(complex)
    z = np.fft.ifft(H)
    return z

def _optimize_from_z(z0, start_time, max_iter=100, tol=1e-8):
    """Optimize over complex z to maximize ratio via |DFT(z)|^2."""
    z = z0.astype(complex).copy()
    
    def objective(z_flat):
        z_c = z_flat[:N] + 1j * z_flat[N:]
        h = _dft_parameterize(z_c)
        return _ratio_neg(h, beta=50.0)
    
    z_flat = np.concatenate([z.real, z.imag])
    
    result = minimize(objective, z_flat, method='L-BFGS-B', 
                      options={'maxiter': max_iter, 'gtol': tol, 'maxls': 20})
    
    z_opt = result.x[:N] + 1j * result.x[N:]
    h_opt = _dft_parameterize(z_opt)
    score = _ratio(h_opt)
    return h_opt, score

def _local_optimize(h0, start_time, max_iter=200, tol=1e-9):
    """Direct optimization of h with non-negativity constraint."""
    h = np.maximum(h0, 0.0).copy()
    
    def objective(h_flat):
        return _ratio_neg(h_flat, beta=50.0)
    
    result = minimize(objective, h, method='L-BFGS-B',
                      bounds=[(0, None)] * N,
                      options={'maxiter': max_iter, 'gtol': tol, 'maxls': 20})
    
    h_opt = np.maximum(result.x, 0.0)
    score = _ratio(h_opt)
    return h_opt, score

def _gaussian_profile(center, width, height=1.0):
    """Create a Gaussian-like profile on the 50 bins."""
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.25 / N
    h = height * np.exp(-((x - center) ** 2) / (2 * width ** 2))
    return h

def _lorentzian_profile(center, gamma, height=1.0):
    """Create a Lorentzian-like profile."""
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.25 / N
    h = height * (gamma ** 2) / ((x - center) ** 2 + gamma ** 2)
    return h

def _power_profile(k, side='right'):
    """Power law profile."""
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.25 / N
    h = np.zeros(N)
    if side == 'right':
        mask = x > 0
        h[mask] = x[mask] ** k
    elif side == 'left':
        mask = x < 0
        h[mask] = (-x[mask]) ** k
    else:
        h = np.abs(x) ** k
    return h

def _two_gaussian(c1, w1, c2, w2, h1=1.0, h2=1.0):
    """Sum of two Gaussians."""
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.25 / N
    h1_arr = h1 * np.exp(-((x - c1) ** 2) / (2 * w1 ** 2))
    h2_arr = h2 * np.exp(-((x - c2) ** 2) / (2 * w2 ** 2))
    return h1_arr + h2_arr

def _sinc_like_profile(width, center=0.0):
    """Create a sinc-like profile."""
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.25 / N
    t = (x - center) / width
    h = np.sinc(t) ** 2
    return np.maximum(h, 0)

def construct():
    start_time = time.time()
    
    best_h = np.ones(N)
    best_score = _ratio(best_h)
    
    # Generate diverse initializations
    initializations = []
    
    # Uniform
    initializations.append(np.ones(N))
    
    # Single Gaussians at various positions and widths
    for center in [-0.2, -0.1, 0.0, 0.1, 0.2]:
        for width in [0.03, 0.05, 0.08, 0.1, 0.15, 0.2]:
            initializations.append(_gaussian_profile(center, width, 1.0))
    
    # Lorentzians
    for center in [-0.15, 0.0, 0.15]:
        for gamma in [0.03, 0.05, 0.08, 0.1, 0.15]:
            initializations.append(_lorentzian_profile(center, gamma, 1.0))
    
    # Power laws
    for k in [2, 3, 4, 5, 6, 8]:
        initializations.append(_power_profile(k, 'right'))
        initializations.append(_power_profile(k, 'left'))
        initializations.append(_power_profile(k, 'both'))
    
    # Two Gaussians
    for c1, c2 in [(-0.15, 0.15), (-0.1, 0.1), (-0.2, 0.2), (-0.05, 0.05), (0.0, 0.1)]:
        for w in [0.03, 0.05, 0.08, 0.12]:
            initializations.append(_two_gaussian(c1, w, c2, w, 1.0, 1.0))
    
    # Sinc-like profiles
    for width in [0.05, 0.1, 0.15]:
        for center in [0.0, 0.05, -0.05]:
            initializations.append(_sinc_like_profile(width, center))
    
    # Random initializations
    rng = np.random.RandomState(42)
    for _ in range(10):
        h_rand = rng.random(N)
        initializations.append(h_rand)
    
    # Sort initializations by score, keep top ones for optimization
    scored_inits = []
    for h_init in initializations:
        h_init = np.maximum(h_init, 0.0)
        s = _ratio(h_init)
        scored_inits.append((s, h_init))
    
    scored_inits.sort(key=lambda x: -x[0])
    
    # Update best from initializations
    if scored_inits[0][0] > best_score:
        best_score = scored_inits[0][0]
        best_h = scored_inits[0][1].copy()
    
    # Optimize from top initializations
    top_k = min(15, len(scored_inits))
    for i in range(top_k):
        if _time_left(start_time) < 5.0:
            break
        
        s_init, h_init = scored_inits[i]
        
        # Direct optimization
        h_opt, s_opt = _local_optimize(h_init, start_time, max_iter=200, tol=1e-9)
        if s_opt > best_score:
            best_score = s_opt
            best_h = h_opt.copy()
        
        # DFT-based optimization
        if _time_left(start_time) < 5.0:
            break
        
        z_init = _dft_inverse(h_init)
        h_dft_opt, s_dft = _optimize_from_z(z_init, start_time, max_iter=100, tol=1e-8)
        if s_dft > best_score:
            best_score = s_dft
            best_h = h_dft_opt.copy()
    
    # Final polish: high-precision optimization from best found
    if _time_left(start_time) > 10.0:
        h_polish, s_polish = _local_optimize(best_h, start_time, max_iter=500, tol=1e-12)
        if s_polish > best_score:
            best_score = s_polish
            best_h = h_polish.copy()
        
        # Another DFT polish
        if _time_left(start_time) > 5.0:
            z_polish = _dft_inverse(best_h)
            h_dft_polish, s_dft_polish = _optimize_from_z(z_polish, start_time, max_iter=200, tol=1e-10)
            if s_dft_polish > best_score:
                best_score = s_dft_polish
                best_h = h_dft_polish.copy()
    
    return best_h.tolist()
