"""Combined: spectral subspace projection + FFT local search + epigraph polish for signed step heights."""
import numpy as np
import time
from scipy.optimize import minimize

N = 400
_FSIZE = 1
while _FSIZE < 2 * N - 1:
    _FSIZE *= 2

# EVOLVE-BLOCK-START

def _autoconv(h):
    """Compute full linear autoconvolution using FFT."""
    H = np.fft.rfft(h, _FSIZE)
    ac = np.fft.irfft(H * H, _FSIZE)
    return ac[:2 * N - 1]

def _raw_value(h):
    """Compute the objective value: max|autoconv| / (sum h)^2."""
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    ac = _autoconv(h)
    return np.max(np.abs(ac)) / (s * s)

def _soft_max_abs(ac, beta=50.0):
    """Smooth approximation of max(|ac|) using log-sum-exp."""
    abs_ac = np.abs(ac)
    m = np.max(abs_ac)
    if m < 1e-15:
        return 0.0
    scaled = beta * (abs_ac - m)
    scaled = np.clip(scaled, -50, 50)
    log_sum = np.log(np.sum(np.exp(scaled)))
    return m + log_sum / beta

def _smooth_objective(h, beta=50.0):
    """Smooth objective: soft_max_abs(autoconv(h)) / (sum(h))^2"""
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    ac = _autoconv(h)
    sm = _soft_max_abs(ac, beta)
    return sm / (s * s)

def _smooth_gradient(h, beta=50.0):
    """Compute analytical gradient of smooth objective."""
    s = np.sum(h)
    if abs(s) < 1e-9:
        return np.zeros_like(h)
    
    ac = _autoconv(h)
    abs_ac = np.abs(ac)
    m = np.max(abs_ac)
    if m < 1e-15:
        return np.zeros_like(h)
    
    scaled = beta * (abs_ac - m)
    scaled = np.clip(scaled, -50, 50)
    weights = np.exp(scaled)
    weights /= np.sum(weights)
    
    # d(soft_max)/d(ac[j]) = beta * weights[j] * sign(ac[j])
    d_sm_d_ac = beta * weights * np.sign(ac)
    
    # ac = h * h (autoconvolution), so d(ac)/d(h[i]) is a shifted version of h
    # ac[k] = sum_j h[j] h[k-j]
    # d(ac[k])/d(h[i]) = h[k-i] + h[i-k] (for valid indices)
    # This is a convolution of delta with h, plus its time-reversed version
    
    # Gradient of soft_max w.r.t. h:
    # d(soft_max)/d(h[i]) = sum_k d_sm_d_ac[k] * d(ac[k])/d(h[i])
    # = sum_k d_sm_d_ac[k] * (h[k-i] + h[i-k])
    # This is a convolution of d_sm_d_ac with h, plus d_sm_d_ac with flipped h
    
    # Use FFT for efficiency
    H = np.fft.rfft(h, _FSIZE)
    D = np.fft.rfft(d_sm_d_ac, _FSIZE)
    # ac = ifft(fft(h)^2), so d(ac)/d(h[i]) corresponds to h convolved with delta
    # Actually, d(ac)/d(h) is: for each i, the derivative is h shifted
    # Let's compute: grad_sm[i] = sum_k d_sm_d_ac[k] * (h[k-i] + h[i-k])
    # = (d_sm_d_ac * h)[i] + (d_sm_d_ac * flip(h))[i]
    
    # Convolution via FFT
    conv1 = np.fft.irfft(D * H, _FSIZE)[:2 * N - 1]
    h_flip = h[::-1]
    HF = np.fft.rfft(h_flip, _FSIZE)
    conv2 = np.fft.irfft(D * HF, _FSIZE)[:2 * N - 1]
    
    grad_sm = conv1 + conv2
    
    # Objective: soft_max / s^2
    # d(obj)/d(h[i]) = (d_sm_d_h[i] * s^2 - soft_max * 2 * s) / s^4
    # = d_sm_d_h[i] / s^2 - 2 * soft_max / s^3
    
    sm = m + np.log(np.sum(np.exp(scaled))) / beta
    grad = grad_sm / (s * s) - 2.0 * sm / (s * s * s)
    
    return grad

def _lbfgs_optimize(h_init, beta_schedule=[15, 30, 60, 100, 150], deadline=None):
    """Multi-stage L-BFGS optimization with smooth surrogate."""
    h = h_init.copy()
    s0 = np.sum(h)
    if abs(s0) < 1e-6:
        return h, _raw_value(h)
    h = h / s0
    
    best_val = _raw_value(h)
    best_h = h.copy()
    
    for beta in beta_schedule:
        if deadline is not None and time.time() > deadline:
            break
        
        def obj(x):
            return _smooth_objective(x, beta=beta)
        
        def grad(x):
            return _smooth_gradient(x, beta=beta)
        
        try:
            res = minimize(obj, h, jac=grad, method='L-BFGS-B',
                           options={'maxiter': 200, 'disp': False, 'gtol': 1e-7})
            h_opt = res.x
            s_opt = np.sum(h_opt)
            if abs(s_opt) > 1e-6:
                h_opt = h_opt / s_opt
                val = _raw_value(h_opt)
                if val < best_val:
                    best_val = val
                    best_h = h_opt.copy()
                h = h_opt
        except Exception:
            pass
    
    return best_h, best_val

def _single_flip_local_search(h, max_passes=10, deadline=None):
    """Single-bit flip local search with incremental FFT."""
    h = np.sign(h.copy())
    h[h == 0] = 1.0
    s = np.sum(h)
    if abs(s) < 1e-9:
        return h, 1e10
    best_val = _raw_value(h)
    
    H = np.fft.rfft(h, _FSIZE)
    ac = np.fft.irfft(H * H, _FSIZE)[:2 * N - 1]
    
    for _ in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        for i in range(N):
            if deadline is not None and time.time() > deadline:
                break
            
            old_h_i = h[i]
            delta = -2.0 * old_h_i
            h[i] = -old_h_i
            
            k_vals = np.arange(_FSIZE // 2 + 1)
            phase = np.exp(-2j * np.pi * i * k_vals / _FSIZE)
            H += delta * phase
            
            ac = np.fft.irfft(H * H, _FSIZE)[:2 * N - 1]
            new_max_abs = np.max(np.abs(ac))
            
            new_s = s - 2.0 * old_h_i
            if abs(new_s) < 1e-9:
                new_val = 1e10
            else:
                new_val = new_max_abs / (new_s * new_s)
            
            if new_val < best_val:
                best_val = new_val
                improved = True
                s = new_s
            else:
                h[i] = old_h_i
                H -= delta * phase
        
        if not improved:
            break
    return h, best_val

def _block_sign_flip_local_search(h, deadline=None, block_sizes=[3, 5, 7, 10, 15]):
    """Block flip local search."""
    h = np.sign(h.copy())
    h[h == 0] = 1.0
    best_val = _raw_value(h)
    improved = True
    while improved and (deadline is None or time.time() < deadline):
        improved = False
        for bs in block_sizes:
            for start in range(0, N - bs + 1, max(1, (N - bs) // 100)):
                if deadline is not None and time.time() > deadline:
                    break
                h[start:start+bs] = -h[start:start+bs]
                new_val = _raw_value(h)
                if new_val < best_val:
                    best_val = new_val
                    improved = True
                else:
                    h[start:start+bs] = -h[start:start+bs]
    return h, best_val

def _spectral_subspace_optimize(h, deadline=None, num_modes=20):
    """Optimize in a spectral subspace spanned by top autoconvolution modes."""
    h = np.sign(h.copy())
    h[h == 0] = 1.0
    s = np.sum(h)
    if abs(s) < 1e-6:
        return h, _raw_value(h)
    h = h / s
    
    best_val = _raw_value(h)
    best_h = h.copy()
    
    # Get current autoconvolution spectrum
    H = np.fft.rfft(h, _FSIZE)
    ac = np.fft.irfft(H * H, _FSIZE)[:2 * N - 1]
    
    # Identify top magnitude frequencies in autoconvolution
    ac_fft = np.fft.rfft(ac, _FSIZE)
    magnitudes = np.abs(ac_fft)
    top_indices = np.argsort(magnitudes)[::-1][:num_modes]
    
    # Build basis vectors in h-space corresponding to these frequencies
    # For frequency k, the corresponding h-space vector is exp(2j*pi*k*n/FSIZE)
    # But we need real-valued basis, so use cos and sin
    basis = []
    for k in top_indices:
        if k == 0:
            basis.append(np.ones(N))
        else:
            n = np.arange(N)
            cos_part = np.cos(2 * np.pi * k * n / _FSIZE)
            sin_part = np.sin(2 * np.pi * k * n / _FSIZE)
            basis.append(cos_part)
            basis.append(sin_part)
    
    # Orthonormalize basis
    basis = np.array(basis)
    basis = basis / np.linalg.norm(basis, axis=1, keepdims=True)
    
    # Optimize coefficients in this subspace
    def subspace_obj(coeffs):
        h_trial = h + basis.T @ coeffs
        s_trial = np.sum(h_trial)
        if abs(s_trial) < 1e-9:
            return 1e10
        ac_trial = _autoconv(h_trial)
        return np.max(np.abs(ac_trial)) / (s_trial * s_trial)
    
    try:
        res = minimize(subspace_obj, np.zeros(len(top_indices) * 2), method='Nelder-Mead',
                       options={'maxiter': 500, 'xatol': 1e-5, 'fatol': 1e-7})
        h_opt = h + basis.T @ res.x
        s_opt = np.sum(h_opt)
        if abs(s_opt) > 1e-6:
            h_opt = h_opt / s_opt
            val = _raw_value(h_opt)
            if val < best_val:
                best_val = val
                best_h = h_opt.copy()
    except Exception:
        pass
    
    return best_h, best_val

def _epigraph_polish(h, deadline=None):
    """Epigraph formulation polish: minimize t subject to |ac[j]| <= t * (sum h)^2 for all j."""
    h = np.sign(h.copy())
    h[h == 0] = 1.0
    s = np.sum(h)
    if abs(s) < 1e-6:
        return h, _raw_value(h)
    h = h / s
    
    best_val = _raw_value(h)
    best_h = h.copy()
    
    # Use smooth epigraph: minimize t + penalty * sum(max(0, |ac[j]| - t * s^2)^2)
    for penalty in [1.0, 10.0, 100.0]:
        if deadline is not None and time.time() > deadline:
            break
        
        def epigraph_obj(params):
            t = params[0]
            h_trial = params[1:]
            s_trial = np.sum(h_trial)
            if abs(s_trial) < 1e-9:
                return 1e10
            ac_trial = _autoconv(h_trial)
            violations = np.maximum(0, np.abs(ac_trial) - t * s_trial * s_trial)
            return t + penalty * np.sum(violations ** 2)
        
        def epigraph_grad(params):
            t = params[0]
            h_trial = params[1:]
            s_trial = np.sum(h_trial)
            if abs(s_trial) < 1e-9:
                return np.zeros_like(params)
            ac_trial = _autoconv(h_trial)
            violations = np.maximum(0, np.abs(ac_trial) - t * s_trial * s_trial)
            d_violations = 2 * penalty * violations * np.sign(ac_trial)
            
            # Gradient w.r.t. t: 1 - penalty * sum(2 * violations * s^2)
            dt = 1.0 - 2.0 * penalty * np.sum(violations * s_trial * s_trial)
            
            # Gradient w.r.t. h: -penalty * sum(2 * violations * s^2 * d(ac)/d(h)) + 2 * penalty * sum(violations * s * d(ac)/d(h))
            # This is complex; use finite difference approximation for simplicity
            eps = 1e-6
            grad_h = np.zeros(N)
            for i in range(N):
                h_plus = h_trial.copy()
                h_plus[i] += eps
                ac_plus = _autoconv(h_plus)
                s_plus = np.sum(h_plus)
                viol_plus = np.maximum(0, np.abs(ac_plus) - t * s_plus * s_plus)
                f_plus = t + penalty * np.sum(viol_plus ** 2)
                
                h_minus = h_trial.copy()
                h_minus[i] -= eps
                ac_minus = _autoconv(h_minus)
                s_minus = np.sum(h_minus)
                viol_minus = np.maximum(0, np.abs(ac_minus) - t * s_minus * s_minus)
                f_minus = t + penalty * np.sum(viol_minus ** 2)
                
                grad_h[i] = (f_plus - f_minus) / (2 * eps)
            
            return np.concatenate([[dt], grad_h])
        
        x0 = np.concatenate([[best_val], h])
        try:
            res = minimize(epigraph_obj, x0, jac=epigraph_grad, method='L-BFGS-B',
                           options={'maxiter': 100, 'disp': False, 'gtol': 1e-6})
            h_opt = res.x[1:]
            s_opt = np.sum(h_opt)
            if abs(s_opt) > 1e-6:
                h_opt = h_opt / s_opt
                val = _raw_value(h_opt)
                if val < best_val:
                    best_val = val
                    best_h = h_opt.copy()
        except Exception:
            pass
    
    return best_h, best_val

def _gen_chirp(freq):
    t = np.arange(N)
    phase = 2 * np.pi * freq * (t ** 2) / N
    x = np.sign(np.cos(phase))
    x[x == 0] = 1.0
    return x

def _gen_random(seed):
    rng = np.random.default_rng(seed)
    return rng.choice([1.0, -1.0], N)

def _gen_window(seed, sigma_frac=0.25):
    rng = np.random.default_rng(seed)
    t = np.arange(N)
    noise = rng.standard_normal(N)
    mu = N / 2
    sigma = N * sigma_frac
    window = np.exp(-(t - mu) ** 2 / (2 * sigma ** 2))
    x = np.sign(noise * window)
    x[x == 0] = 1.0
    return x

def construct():
    t_start = time.time()
    deadline = t_start + 135.0
    
    best_score = np.inf
    best_h = np.ones(N)
    
    # Baseline
    base_score = _raw_value(best_h)
    if base_score < best_score:
        best_score = base_score
        best_h = best_h.copy()
    
    # Generate diverse candidates
    candidates = []
    candidates.append(np.ones(N))
    
    for freq in [0.5, 1.0, 1.5, 2.0, 3.0, 0.3, 0.7]:
        candidates.append(_gen_chirp(freq))
    
    for seed in [42, 123, 456, 789, 101, 202, 303, 404, 505, 606, 707, 808]:
        candidates.append(_gen_random(seed))
    
    for seed in [11, 22, 33, 44, 55]:
        candidates.append(_gen_window(seed))
    
    # Evaluate all candidates
    for x in candidates:
        val = _raw_value(x)
        if val < best_score:
            best_score = val
            best_h = x.copy()
    
    # L-BFGS optimization from top candidates
    scored = [( _raw_value(x), x) for x in candidates]
    scored.sort(key=lambda t: t[0])
    
    for init_val, h_init in scored[:8]:
        if time.time() > deadline - 40:
            break
        h_opt, val_opt = _lbfgs_optimize(h_init, deadline=deadline - 35)
        if val_opt < best_score:
            best_score = val_opt
            best_h = h_opt.copy()
    
    # Spectral subspace optimization
    if time.time() < deadline - 30:
        h_opt, val_opt = _spectral_subspace_optimize(best_h, deadline=deadline - 25, num_modes=15)
        if val_opt < best_score:
            best_score = val_opt
            best_h = h_opt.copy()
    
    # Epigraph polish
    if time.time() < deadline - 25:
        h_opt, val_opt = _epigraph_polish(best_h, deadline=deadline - 20)
        if val_opt < best_score:
            best_score = val_opt
            best_h = h_opt.copy()
    
    # Binarize and local search
    if time.time() < deadline - 20:
        h_bin = np.sign(best_h)
        h_bin[h_bin == 0] = 1.0
        val_bin = _raw_value(h_bin)
        if val_bin < best_score:
            best_score = val_bin
            best_h = h_bin.copy()
        
        # Single flip local search
        h_opt, val_opt = _single_flip_local_search(best_h, max_passes=15, deadline=deadline - 10)
        if val_opt < best_score:
            best_score = val_opt
            best_h = h_opt.copy()
        
        # Block flip local search
        if time.time() < deadline - 10:
            h_opt, val_opt = _block_sign_flip_local_search(best_h, deadline=deadline - 5,
                                                           block_sizes=[3, 5, 7, 10, 15])
            if val_opt < best_score:
                best_score = val_opt
                best_h = h_opt.copy()
    
    # Random restarts with local search
    if time.time() < deadline - 15:
        rng = np.random.default_rng(999)
        for _ in range(10):
            if time.time() > deadline - 8:
                break
            x_rand = rng.choice([1.0, -1.0], N)
            h_opt, val_opt = _single_flip_local_search(x_rand, max_passes=5, deadline=deadline - 5)
            if val_opt < best_score:
                best_score = val_opt
                best_h = h_opt.copy()
    
    # Final polish
    if time.time() < deadline - 3:
        h_final, val_final = _single_flip_local_search(best_h, max_passes=10, deadline=deadline - 1)
        if val_final < best_score:
            best_score = val_final
            best_h = h_final.copy()
    
    if best_score > base_score:
        best_score = base_score
        best_h = np.ones(N)
    
    return best_h

# EVOLVE-BLOCK-END
