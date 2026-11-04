"""Combined program: DFT spectral parameterization + diverse profiles + dual optimization."""
import numpy as np
from scipy.optimize import minimize
import time

N = 50
dx = 0.5 / N  # interval width

def _compute_ratio(h):
    """Compute ||f||_2^2 / (||f||_1 * ||f*f||_inf) for step function on [-1/4, 1/4]."""
    h = np.asarray(h, dtype=float)
    h = np.maximum(0, h)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    l2_sq = np.sum(h ** 2)
    if l2_sq < 1e-30:
        return 0.0
    ac = np.correlate(h, h, mode='full')
    max_ac = np.max(ac)
    if max_ac < 1e-30:
        return 0.0
    # The ratio: ||f||_2^2 / (||f||_1 * ||f*f||_inf)
    # For step functions: ||f||_2^2 = dx * sum(h^2), ||f||_1 = dx * sum(h)
    # ||f*f||_inf approximated by dx * max(ac) (since autoconvolution of step functions)
    # Actually: (f*f)(x) = sum_i h_i * h_j * dx for appropriate overlaps
    # The peak of autoconvolution is dx * max(correlate(h,h))
    # So ratio = (dx * sum(h^2)) / (dx * sum(h) * dx * max(ac))
    #         = sum(h^2) / (dx * sum(h) * max(ac))
    return l2_sq / (dx * l1 * max_ac)

def _ratio_neg(h):
    s = _compute_ratio(h)
    return -s if s > 0 else 1e10

def _dft_param(z):
    """Parameterize heights as |DFT(z)|^2 where z is complex vector of length N."""
    H = np.fft.fft(z)
    h = np.abs(H) ** 2
    return h

def _dft_inverse(h):
    """Given h = |DFT(z)|^2, find a z (phase choice: all positive magnitudes)."""
    H_mags = np.sqrt(np.maximum(h, 0) + 1e-30)
    H = H_mags.astype(complex)
    z = np.fft.ifft(H)
    return z

def _optimize_z(z0, deadline, max_iter=150, tol=1e-9):
    """Optimize over complex z to maximize ratio via |DFT(z)|^2."""
    def objective(z_flat):
        z_c = z_flat[:N] + 1j * z_flat[N:]
        h = _dft_param(z_c)
        return _ratio_neg(h)
    
    z_flat = np.concatenate([z0.real, z0.imag])
    
    try:
        result = minimize(objective, z_flat, method='L-BFGS-B',
                          options={'maxiter': max_iter, 'gtol': tol, 'maxls': 20})
        z_opt = result.x[:N] + 1j * result.x[N:]
        h_opt = _dft_param(z_opt)
        score = _compute_ratio(h_opt)
        return h_opt, score
    except Exception:
        return z0, 0.0

def _optimize_direct(h0, deadline, max_iter=200, tol=1e-9):
    """Direct optimization of h with non-negativity constraint."""
    h = np.maximum(h0, 0.0).copy()
    
    try:
        result = minimize(_ratio_neg, h, method='L-BFGS-B',
                          bounds=[(0, None)] * N,
                          options={'maxiter': max_iter, 'gtol': tol, 'maxls': 20})
        h_opt = np.maximum(result.x, 0.0)
        score = _compute_ratio(h_opt)
        return h_opt, score
    except Exception:
        return h, 0.0

def construct():
    start = time.time()
    deadline = start + 110  # Safety margin below 120s limit

    def time_left():
        return deadline - time.time()

    # Generate diverse initializations
    initializations = []
    
    # Uniform
    initializations.append(np.ones(N))
    
    # Single point spikes
    for i in range(N):
        h = np.zeros(N)
        h[i] = 1.0
        initializations.append(h)
    
    # Two-point spikes
    for i in range(0, N, 3):
        for j in range(i + 1, N, 3):
            for ratio in [0.5, 1.0, 2.0]:
                h = np.zeros(N)
                h[i] = 1.0
                h[j] = ratio
                initializations.append(h)
    
    # Gaussian profiles at various positions and widths
    for center in np.linspace(0, N - 1, 10):
        for width in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0]:
            t = np.arange(N)
            h = np.exp(-0.5 * ((t - center) / width) ** 2)
            initializations.append(h)
    
    # Lorentzian profiles
    for center in np.linspace(0, N - 1, 8):
        for gamma in [0.5, 1.0, 2.0, 3.0, 5.0]:
            t = np.arange(N)
            h = 1.0 / (1.0 + ((t - center) / gamma) ** 2)
            initializations.append(h)
    
    # Power law profiles
    for center in [N // 4, N // 2, 3 * N // 4]:
        for p in [2.0, 3.0, 5.0]:
            t = np.arange(N)
            h = (1.0 + np.abs(t - center)) ** (-p)
            initializations.append(h)
    
    # Box profiles
    for center in np.linspace(2, N - 3, 8):
        for width in [1, 2, 3, 5, 8]:
            h = np.where(np.abs(np.arange(N) - center) <= width, 1.0, 0.0)
            initializations.append(h)
    
    # Sinc-like profiles
    for center in [N // 4, N // 2, 3 * N // 4]:
        for width in [2.0, 4.0, 6.0]:
            t = np.arange(N)
            h = np.sinc((t - center) / width) ** 2
            h = np.maximum(h, 0)
            initializations.append(h)
    
    # Two Gaussian mixtures
    for c1, c2 in [(N//4, 3*N//4), (N//3, 2*N//3), (N//5, 4*N//5)]:
        for w in [1.0, 2.0, 3.0]:
            t = np.arange(N)
            h = np.exp(-0.5 * ((t - c1) / w) ** 2) + np.exp(-0.5 * ((t - c2) / w) ** 2)
            initializations.append(h)
    
    # Random initializations
    rng = np.random.RandomState(42)
    for _ in range(20):
        h_rand = rng.random(N)
        h_rand[h_rand < 0.1] = 0  # Sparsify
        initializations.append(h_rand)
    
    # Score all initializations and sort
    scored_inits = []
    for h_init in initializations:
        h_init = np.maximum(h_init, 0.0)
        s = _compute_ratio(h_init)
        scored_inits.append((s, h_init))
    
    scored_inits.sort(key=lambda x: -x[0])
    
    best_h = scored_inits[0][1].copy()
    best_score = scored_inits[0][0]
    
    # Optimize from top initializations using both methods
    top_k = min(20, len(scored_inits))
    for i in range(top_k):
        if time_left() < 8.0:
            break
        
        s_init, h_init = scored_inits[i]
        
        # Direct optimization
        h_opt, s_opt = _optimize_direct(h_init, deadline, max_iter=150, tol=1e-9)
        if s_opt > best_score:
            best_score = s_opt
            best_h = h_opt.copy()
        
        # DFT-based optimization
        if time_left() < 5.0:
            break
        
        z_init = _dft_inverse(h_init)
        h_dft_opt, s_dft = _optimize_z(z_init, deadline, max_iter=100, tol=1e-9)
        if s_dft > best_score:
            best_score = s_dft
            best_h = h_dft_opt.copy()
    
    # Final polish from best found
    if time_left() > 10.0:
        h_polish, s_polish = _optimize_direct(best_h, deadline, max_iter=300, tol=1e-11)
        if s_polish > best_score:
            best_score = s_polish
            best_h = h_polish.copy()
        
        if time_left() > 5.0:
            z_polish = _dft_inverse(best_h)
            h_dft_polish, s_dft_polish = _optimize_z(z_polish, deadline, max_iter=200, tol=1e-11)
            if s_dft_polish > best_score:
                best_score = s_dft_polish
                best_h = h_dft_polish.copy()
    
    # Coordinate descent polish
    if time_left() > 3.0:
        h = best_h.copy()
        for iteration in range(100):
            if time_left() < 1.0:
                break
            improved = False
            for i in range(N):
                old = h[i]
                for mult in [0.0, 0.2, 0.5, 0.8, 1.2, 1.5, 2.0, 5.0]:
                    h[i] = max(0.0, old * mult) if old > 0 else mult
                    s = _compute_ratio(h)
                    if s > best_score + 1e-15:
                        best_score = s
                        best_h = h.copy()
                        improved = True
                    else:
                        h[i] = old
                        break
                h[i] = old
            if not improved:
                break
    
    # Ensure nonnegativity and return
    best_h = np.maximum(0, best_h)
    if np.sum(best_h) < 1e-15:
        best_h = np.zeros(N)
        best_h[N // 2] = 1.0

    return list(best_h)
