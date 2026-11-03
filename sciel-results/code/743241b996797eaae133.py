"""Rewritten program: signed step heights minimizing max|autoconv|/(sum h)^2."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _autoconv_fft(h):
    """Compute linear autoconvolution using FFT."""
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)
    return ac[:size]

def _autoconv_max_abs(h):
    """Compute max absolute value of linear autoconvolution."""
    ac = _autoconv_fft(h)
    return np.max(np.abs(ac))

def _score(h):
    """Compute the objective value: max|autoconv| / (sum h)^2."""
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    return _autoconv_max_abs(h) / (s * s)

def _soft_max_abs(ac, beta=100.0):
    """
    Smooth approximation of max(|ac|) via log-sum-exp.
    """
    abs_ac = np.abs(ac)
    m = np.max(abs_ac)
    if m < 1e-15:
        return 0.0
    scaled = beta * (abs_ac - m)
    scaled = np.clip(scaled, -100, 100)
    log_sum = np.log(np.sum(np.exp(scaled)))
    return m + log_sum / beta

def _objective_smooth(h, beta=100.0):
    """
    Smooth objective for optimization: soft_max_abs(autoconv(h)) / (sum h)^2.
    """
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    ac = _autoconv_fft(h)
    sm = _soft_max_abs(ac, beta)
    return sm / (s * s)

def _epigraph_objective(h, t, beta=200.0):
    """
    Epigraph formulation: minimize t subject to soft_max_abs(ac) <= t * (sum h)^2.
    We use a penalty: t + penalty * max(0, soft_max_abs(ac) - t * s^2)^2
    """
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    ac = _autoconv_fft(h)
    sm = _soft_max_abs(ac, beta)
    s2 = s * s
    violation = max(0.0, sm - t * s2)
    return t + 100.0 * violation * violation

def _epigraph_polish(h, deadline, beta=200.0):
    """
    Polish using epigraph formulation with auxiliary variable t.
    """
    s0 = np.sum(h)
    if abs(s0) < 1e-9:
        return h, _score(h)
    
    h_norm = h / s0
    ac0 = _autoconv_fft(h_norm)
    sm0 = _soft_max_abs(ac0, beta)
    t0 = sm0 + 1e-6  # Initial t slightly above current value
    
    def obj(x):
        h_part = x[:N]
        t_val = x[N]
        return _epigraph_objective(h_part, t_val, beta)
    
    x0 = np.concatenate([h_norm, [t0]])
    
    try:
        res = minimize(obj, x0, method='L-BFGS-B',
                       options={'maxiter': 300, 'disp': False, 'gtol': 1e-7, 'ftol': 1e-9})
        h_opt = res.x[:N]
        if abs(np.sum(h_opt)) < 1e-6:
            return h, _score(h)
        val = _score(h_opt)
        return h_opt, val
    except Exception:
        return h, _score(h)

def _gen_barker_like(n):
    """
    Generate a Barker-like sequence: +1/-1 with good autocorrelation properties.
    We use a pseudo-random sign sequence smoothed slightly.
    """
    rng = np.random.default_rng(123)
    h = rng.choice([-1.0, 1.0], size=n)
    # Light smoothing to help gradient methods
    kernel = np.array([0.2, 0.3, 0.3, 0.2])
    h = np.convolve(h, kernel, mode='same')
    return h

def _gen_chirp(n):
    """Generate a chirp sequence."""
    i = np.arange(n)
    phase = np.pi * i**2 / n
    return np.cos(phase)

def _gen_gaussian(n):
    """Generate a Gaussian-shaped sequence."""
    x = np.arange(n)
    mu = n / 2
    sigma = n / 4
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def _gen_random_smooth(n, seed=42):
    """Generate a random smoothed sequence."""
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(n)
    kernel = np.ones(8) / 8
    return np.convolve(noise, kernel, mode='same')

def _autoconv_after_flip(h, ac_base, i):
    """
    Compute autoconvolution after flipping sign at index i.
    h' = h - 2*h[i]*e_i
    ac(h')_k = ac(h)_k - 2*h[i]*h[i+k] - 2*h[i]*h[i-k] + 4*h[i]^2*delta_{k,0}
    where out-of-range indices contribute 0.
    """
    n = len(h)
    size = 2 * n - 1
    ac = ac_base.copy()
    hi = h[i]
    
    # For lag k (0 to size-1):
    # ac_k = sum_j h_j * h_{j+k} (with h_j = 0 if j out of [0,n-1])
    # After flip: h'_j = h_j - 2*hi*delta_{j,i}
    # ac'_k = sum_j (h_j - 2*hi*delta_{j,i})(h_{j+k} - 2*hi*delta_{j+k,i})
    #       = ac_k - 2*hi*sum_j delta_{j,i}*h_{j+k} - 2*hi*sum_j h_j*delta_{j+k,i} + 4*hi^2*sum_j delta_{j,i}*delta_{j+k,i}
    #       = ac_k - 2*hi*h_{i+k} - 2*hi*h_{i-k} + 4*hi^2*delta_{k,0}
    
    # h_{i+k}: valid when 0 <= i+k < n
    for k in range(size):
        # Term 1: h[i+k]
        if 0 <= i + k < n:
            ac[k] -= 2 * hi * h[i + k]
        # Term 2: h[i-k]
        if 0 <= i - k < n:
            ac[k] -= 2 * hi * h[i - k]
        # Term 3: delta_{k,0}
        if k == 0:
            ac[k] += 4 * hi * hi
    
    return ac

def _local_search(h, max_passes=100, deadline=None):
    """
    Local search with incremental autoconvolution updates for sign flips.
    """
    h = h.copy()
    best_val = _score(h)
    s = np.sum(h)
    rng = np.random.default_rng(42)
    
    # Precompute base autoconv
    ac_base = _autoconv_fft(h)
    
    for _ in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        
        # Try all single sign flips using incremental update
        best_flip_i = -1
        best_flip_val = best_val
        best_flip_ac = None
        best_flip_s = s
        
        for i in range(N):
            ac_new = _autoconv_after_flip(h, ac_base, i)
            s_new = s - 2 * h[i]
            if abs(s_new) < 1e-9:
                continue
            val_new = np.max(np.abs(ac_new)) / (s_new * s_new)
            if val_new < best_flip_val:
                best_flip_val = val_new
                best_flip_i = i
                best_flip_ac = ac_new
                best_flip_s = s_new
        
        if best_flip_i >= 0 and best_flip_val < best_val * (1 - 1e-12):
            h[best_flip_i] = -h[best_flip_i]
            ac_base = best_flip_ac
            s = best_flip_s
            best_val = best_flip_val
            improved = True
        
        # Also try small multiplicative perturbations
        for _ in range(N // 4):
            if deadline is not None and time.time() > deadline:
                break
            i = rng.integers(N)
            orig = h[i]
            for factor in [1.05, 0.95, 1.1, 0.9]:
                if deadline is not None and time.time() > deadline:
                    break
                h[i] = orig * factor
                new_val = _score(h)
                if new_val < best_val:
                    best_val = new_val
                    ac_base = _autoconv_fft(h)
                    s = np.sum(h)
                    improved = True
                    break
                else:
                    h[i] = orig
        
        if not improved:
            break
    
    return h, best_val

def construct():
    t_start = time.time()
    deadline = t_start + 147.0  # Safety margin before 150s kill
    
    best_score = np.inf
    best_h = np.ones(N)
    
    base_score = _score(best_h)
    best_score = base_score
    
    inits = []
    
    # 1. Constant
    inits.append(np.ones(N))
    
    # 2. Gaussian
    inits.append(_gen_gaussian(N))
    
    # 3. Chirp
    inits.append(_gen_chirp(N))
    
    # 4. Barker-like
    inits.append(_gen_barker_like(N))
    
    # 5. Random smoothed
    inits.append(_gen_random_smooth(N, seed=1))
    
    # 6. Another random with different seed
    inits.append(_gen_random_smooth(N, seed=99))
    
    # 7. Alternating signs with envelope
    x = np.arange(N)
    envelope = np.exp(-0.5 * ((x - N/2) / (N/5))**2)
    inits.append(envelope * np.cos(np.pi * x / 5))
    
    # 8. Another chirp variant
    i = np.arange(N)
    inits.append(np.cos(2 * np.pi * i * 3 / N))
    
    for h_init in inits:
        if time.time() > deadline - 20:
            break
        
        s0 = np.sum(h_init)
        if abs(s0) < 1e-6:
            continue
        
        h_norm = h_init / s0
        
        # Try multiple beta values for smooth objective
        for beta in [50, 100, 200]:
            if time.time() > deadline - 12:
                break
            
            try:
                def obj(x, b=beta):
                    return _objective_smooth(x, beta=b)
                
                res = minimize(obj, h_norm, method='L-BFGS-B',
                               options={'maxiter': 250, 'disp': False, 'gtol': 1e-6, 'ftol': 1e-8})
                h_opt = res.x
                
                if abs(np.sum(h_opt)) < 1e-6:
                    continue
                    
                val = _score(h_opt)
                if val < best_score:
                    best_score = val
                    best_h = h_opt.copy()
            except Exception:
                pass
        
        # Also try without normalization (the objective handles it)
        if time.time() > deadline - 10:
            continue
        try:
            def obj_raw(x):
                return _objective_smooth(x, beta=100.0)
            
            res = minimize(obj_raw, h_init, method='L-BFGS-B',
                           options={'maxiter': 150, 'disp': False, 'gtol': 1e-5})
            h_opt = res.x
            if abs(np.sum(h_opt)) < 1e-6:
                continue
            val = _score(h_opt)
            if val < best_score:
                best_score = val
                best_h = h_opt.copy()
        except Exception:
            pass
    
    # Local search on best found - use more passes now that we have incremental updates
    if time.time() < deadline - 8:
        h_refined, val_refined = _local_search(best_h, max_passes=200, deadline=deadline - 4)
        if val_refined < best_score:
            best_score = val_refined
            best_h = h_refined.copy()
    
    # Epigraph polish
    if time.time() < deadline - 3:
        try:
            h_epi, val_epi = _epigraph_polish(best_h, deadline - 1)
            if val_epi < best_score:
                best_score = val_epi
                best_h = h_epi.copy()
        except Exception:
            pass
    
    # High-precision polish: run L-BFGS-B with tighter tolerance on the best result
    if time.time() < deadline - 2:
        try:
            s0 = np.sum(best_h)
            if abs(s0) > 1e-9:
                h_norm = best_h / s0
                res = minimize(lambda x: _objective_smooth(x, beta=200.0), h_norm,
                               method='L-BFGS-B',
                               options={'maxiter': 400, 'disp': False, 'gtol': 1e-8, 'ftol': 1e-10})
                h_opt = res.x
                if abs(np.sum(h_opt)) > 1e-9:
                    val = _score(h_opt)
                    if val < best_score:
                        best_score = val
                        best_h = h_opt.copy()
        except Exception:
            pass
    
    # Final quick local search
    if time.time() < deadline - 1:
        try:
            h_final, val_final = _local_search(best_h, max_passes=50, deadline=deadline - 0.5)
            if val_final < best_score:
                best_score = val_final
                best_h = h_final.copy()
        except Exception:
            pass
    
    # Safety: if somehow we lost track, fall back
    if best_score > base_score:
        best_score = base_score
        best_h = np.ones(N)
    
    return best_h

# EVOLVE-BLOCK-END
