import numpy as np
from scipy.optimize import minimize
import time

N = 50
_DEADLINE = 142.0  # safety margin before 150s kill

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

def _get_sidon_set(n, max_val):
    """Generate a Sidon set of size n within [0, max_val]."""
    S = [0]
    sums = set([0])
    for i in range(1, max_val + 1):
        if len(S) >= n:
            break
        new_sums = set()
        valid = True
        for s in S:
            new_sum = i + s
            if new_sum in sums or new_sum in new_sums:
                valid = False
                break
            new_sums.add(new_sum)
        two_i = 2 * i
        if two_i in sums or two_i in new_sums:
            valid = False
        
        if valid:
            S.append(i)
            sums.update(new_sums)
            sums.add(two_i)
    return S

def _optimize_weights(sidon_indices, time_budget=10.0):
    """Optimize weights on Sidon indices to maximize the ratio."""
    k = len(sidon_indices)
    if k == 0:
        return np.zeros(N)
    
    start = time.time()
    
    # Parameterize: log-weights for positivity, with L1 normalization
    # Use softmax-like parameterization: w_i = exp(x_i) / sum(exp(x_j))
    def _objective(x):
        w = np.exp(x - np.max(x))  # numerical stability
        w = w / np.sum(w)
        h = np.zeros(N)
        for i, idx in enumerate(sidon_indices):
            h[idx] = w[i]
        return _ratio_neg(h)
    
    # Start from uniform weights
    x0 = np.zeros(k)
    
    # Try multiple random restarts
    best_x = x0.copy()
    best_val = _objective(x0)
    
    n_restarts = 5
    for r in range(n_restarts):
        if time.time() - start > time_budget:
            break
        if r == 0:
            x_r = x0.copy()
        else:
            x_r = np.random.randn(k) * 0.5
        
        try:
            res = minimize(_objective, x_r, method='L-BFGS-B',
                          options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass
    
    # Convert optimized weights to h
    w = np.exp(best_x - np.max(best_x))
    w = w / np.sum(w)
    h = np.zeros(N)
    for i, idx in enumerate(sidon_indices):
        h[idx] = w[i]
    return h

def _try_larger_sidon(max_size=15, time_budget=5.0):
    """Try to find a larger Sidon set and optimize on it."""
    start = time.time()
    best_h = None
    best_ratio = 0.0
    
    for size in range(8, max_size + 1):
        if time.time() - start > time_budget:
            break
        sidon = _get_sidon_set(size, N - 1)
        if len(sidon) < size:
            continue
        h = _optimize_weights(sidon, time_budget=max(1.0, time_budget - (time.time() - start)))
        r = _ratio(h)
        if r > best_ratio:
            best_ratio = r
            best_h = h.copy()
    
    return best_h, best_ratio

def _local_optimize(h0, start_time, max_iter=200, tol=1e-9):
    """Direct optimization of h with non-negativity constraint."""
    h = np.maximum(h0, 0.0).copy()
    
    def objective(h_flat):
        return _ratio_neg(h_flat)
    
    try:
        result = minimize(objective, h, method='L-BFGS-B',
                          bounds=[(0, None)] * N,
                          options={'maxiter': max_iter, 'gtol': tol, 'maxls': 20})
        
        h_opt = np.maximum(result.x, 0.0)
        score = _ratio(h_opt)
        return h_opt, score
    except Exception:
        return h0.copy(), _ratio(h0)

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
        return _ratio_neg(h)
    
    z_flat = np.concatenate([z.real, z.imag])
    
    try:
        result = minimize(objective, z_flat, method='L-BFGS-B', 
                          options={'maxiter': max_iter, 'gtol': tol, 'maxls': 20})
        
        z_opt = result.x[:N] + 1j * result.x[N:]
        h_opt = _dft_parameterize(z_opt)
        score = _ratio(h_opt)
        return h_opt, score
    except Exception:
        h0 = _dft_parameterize(z)
        return h0, _ratio(h0)

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
    top_k = min(10, len(scored_inits))
    for i in range(top_k):
        if time.time() - start_time > _DEADLINE - 30:
            break
        
        s_init, h_init = scored_inits[i]
        
        # Direct optimization
        h_opt, s_opt = _local_optimize(h_init, start_time, max_iter=150, tol=1e-9)
        if s_opt > best_score:
            best_score = s_opt
            best_h = h_opt.copy()
        
        # DFT-based optimization
        if time.time() - start_time > _DEADLINE - 25:
            break
        
        z_init = _dft_inverse(h_init)
        h_dft_opt, s_dft = _optimize_from_z(z_init, start_time, max_iter=80, tol=1e-8)
        if s_dft > best_score:
            best_score = s_dft
            best_h = h_dft_opt.copy()
    
    # Try Sidon set approach
    if time.time() - start_time < _DEADLINE - 20:
        sidon_indices = _get_sidon_set(10, N - 1)
        h_sidon = np.zeros(N)
        for idx in sidon_indices:
            h_sidon[idx] = 1.0
        l1 = np.sum(h_sidon)
        if l1 > 0:
            h_sidon = h_sidon / l1
        r_sidon = _ratio(h_sidon)
        
        if r_sidon > best_score:
            best_score = r_sidon
            best_h = h_sidon.copy()
        
        # Try optimizing weights on the same Sidon set
        if time.time() - start_time < _DEADLINE - 15:
            h_opt_sidon = _optimize_weights(sidon_indices, time_budget=10.0)
            r_opt_sidon = _ratio(h_opt_sidon)
            if r_opt_sidon > best_score:
                best_score = r_opt_sidon
                best_h = h_opt_sidon.copy()
        
        # Try larger Sidon sets
        if time.time() - start_time < _DEADLINE - 10:
            h_large, r_large = _try_larger_sidon(max_size=12, time_budget=8.0)
            if h_large is not None and r_large > best_score:
                best_score = r_large
                best_h = h_large.copy()
    
    # Final polish: high-precision optimization from best found
    if time.time() - start_time < _DEADLINE - 8:
        h_polish, s_polish = _local_optimize(best_h, start_time, max_iter=300, tol=1e-12)
        if s_polish > best_score:
            best_score = s_polish
            best_h = h_polish.copy()
        
        # Another DFT polish
        if time.time() - start_time < _DEADLINE - 5:
            z_polish = _dft_inverse(best_h)
            h_dft_polish, s_dft_polish = _optimize_from_z(z_polish, start_time, max_iter=100, tol=1e-10)
            if s_dft_polish > best_score:
                best_score = s_dft_polish
                best_h = h_dft_polish.copy()
    
    # Exact-max polish: re-optimize with high beta (near-exact supremum)
    if time.time() - start_time < _DEADLINE - 3:
        def _ratio_neg_exact(h_flat, beta=1000.0):
            return -_ratio(h_flat, beta)
        
        h = np.maximum(best_h, 0.0).copy()
        try:
            res = minimize(lambda x: _ratio_neg_exact(x), h, method='L-BFGS-B',
                           bounds=[(0, None)] * N,
                           options={'maxiter': 200, 'gtol': 1e-12, 'maxls': 15})
            h_exact = np.maximum(res.x, 0.0)
            s_exact = _ratio(h_exact, beta=1000.0)
            if s_exact > best_score:
                best_score = s_exact
                best_h = h_exact.copy()
        except Exception:
            pass
    
    # Final: ensure nonnegative and valid
    best_h = np.maximum(best_h, 0.0)
    if np.sum(best_h) < 1e-15:
        # Fallback to uniform
        best_h = np.ones(N) / N
    
    return [float(x) for x in best_h]
