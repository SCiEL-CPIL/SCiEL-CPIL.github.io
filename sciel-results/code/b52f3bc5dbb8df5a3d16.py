import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE = 140.0  # safety margin before 150s kill

# EVOLVE-BLOCK-START
def _value(h):
    """Compute the objective value for given step heights."""
    h = np.asarray(h, dtype=np.float64)
    if np.any(h < -1e-12) or not np.all(np.isfinite(h)):
        return -1.0
    h = np.clip(h, 0.0, None)
    g = np.convolve(h, h)
    m = len(g)
    y = np.zeros(m + 2)
    y[1:m+1] = g
    w = 1.0 / (m + 1)
    terms = y[:-1]**2 + y[:-1]*y[1:] + y[1:]**2
    l2 = (w / 3.0) * np.sum(terms)
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    if norm_inf <= 0 or norm_1 <= 0:
        return -1.0
    return l2 / (norm_1 * norm_inf)

def _h_from_spectrum(spectrum):
    """Convert spectral coefficients to nonnegative step heights via inverse DST."""
    # Use scipy's dst for type II (forward) and type III (inverse)
    # spectrum has N coefficients, we use them as DST-III input to get h
    from scipy.fft import dst, idst
    # idst with type=2 inverts dst with type=3
    # Actually let's use: h = idst(spectrum, type=2) / N * 2 for proper normalization
    h = idst(spectrum, type=2) * 2.0 / N
    h = np.abs(h)  # ensure nonnegativity
    return h

def _spectrum_from_h(h):
    """Convert step heights to spectral coefficients via DST."""
    from scipy.fft import dst
    spec = dst(h, type=2) * N / 2.0
    return spec

def _smooth_max(g, beta=50.0):
    """Log-sum-exp soft approximation of max for differentiability."""
    g_max = np.max(g)
    if g_max <= 0:
        return 0.0
    return g_max + np.log(np.sum(np.exp(beta * (g - g_max)))) / beta

def _neg_value_smooth(spec, beta=50.0):
    """Negative smooth objective for minimization. spec are spectral coefficients."""
    h = _h_from_spectrum(spec)
    if not np.all(np.isfinite(h)) or np.any(h < -1e-12):
        return 1e10
    h = np.clip(h, 0.0, None)
    if h.sum() < 1e-12:
        return 1e10
    g = np.convolve(h, h)
    m = len(g)
    y = np.zeros(m + 2)
    y[1:m+1] = g
    w = 1.0 / (m + 1)
    terms = y[:-1]**2 + y[:-1]*y[1:] + y[1:]**2
    l2 = (w / 3.0) * np.sum(terms)
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf_smooth = _smooth_max(g, beta)
    if norm_inf_smooth <= 0 or norm_1 <= 0:
        return 1e10
    return -(l2 / (norm_1 * norm_inf_smooth))

def _neg_value_direct(spec):
    """Negative exact objective (non-smooth) for final evaluation."""
    h = _h_from_spectrum(spec)
    if not np.all(np.isfinite(h)) or np.any(h < -1e-12):
        return 1e10
    h = np.clip(h, 0.0, None)
    if h.sum() < 1e-12:
        return 1e10
    v = _value(h)
    if v < 0:
        return 1e10
    return -v

def _neg_value_smooth_grad(spec, beta=50.0):
    """Gradient of the smooth negative objective via finite differences."""
    h = _h_from_spectrum(spec)
    h = np.clip(h, 0.0, None)
    g = np.convolve(h, h)
    m = len(g)
    y = np.zeros(m + 2)
    y[1:m+1] = g
    w = 1.0 / (m + 1)
    terms = y[:-1]**2 + y[:-1]*y[1:] + y[1:]**2
    l2 = (w / 3.0) * np.sum(terms)
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf_smooth = _smooth_max(g, beta)
    
    # Compute gradient numerically (central differences)
    grad = np.zeros_like(spec)
    eps = 1e-7
    for i in range(len(spec)):
        spec_plus = spec.copy()
        spec_minus = spec.copy()
        spec_plus[i] += eps
        spec_minus[i] -= eps
        fp = _neg_value_smooth(spec_plus, beta)
        fm = _neg_value_smooth(spec_minus, beta)
        grad[i] = (fp - fm) / (2.0 * eps)
    return grad

def _local_search(h0, max_iters=3000, tol=1e-9):
    """Coordinate ascent on shape (sum=1 normalization after each step)."""
    h = np.array(h0, dtype=float)
    s = h.sum()
    if s <= 0:
        return np.ones(N)/N, -1.0
    h = h / s
    cur_val = _value(h)
    if cur_val < 0:
        return h, cur_val
    step = 0.05
    for it in range(max_iters):
        improved = False
        for i in range(N):
            if h[i] <= 1e-12:
                continue
            for sign in [1.0, -1.0]:
                h_try = h.copy()
                h_try[i] = max(0.0, h[i] + sign * step)
                s_try = h_try.sum()
                if s_try <= 0:
                    continue
                h_try = h_try / s_try
                v = _value(h_try)
                if v > cur_val + 1e-12:
                    h = h_try
                    cur_val = v
                    improved = True
                    break
        if not improved:
            step *= 0.7
            if step < 1e-4:
                break
        if it % 20 == 0:
            for i in range(N):
                if h[i] <= 1e-12:
                    h_try = h.copy()
                    h_try[i] = step * 0.1
                    s_try = h_try.sum()
                    if s_try <= 0:
                        continue
                    h_try = h_try / s_try
                    v = _value(h_try)
                    if v > cur_val + 1e-12:
                        h = h_try
                        cur_val = v
    return h, cur_val

def _construct_hump(power=1.0, shift=0.0, width=1.0, flat_top=0.0):
    """Concave hump centered at shift in [0,1], width relative to N, power controls shape."""
    x = np.linspace(0, 1, N)
    t = np.abs(x - (shift + 0.5)) / (0.5 * width)
    t = np.clip(t, 0, 1)
    if power <= 0:
        f = np.ones(N)
    else:
        f = (1.0 - t**power) ** 1.0
    if flat_top > 0:
        mask = t < flat_top
        f = np.where(mask, 1.0, f * (1.0 - flat_top))
    f = np.maximum(f, 0.0)
    if f.sum() <= 0:
        return np.ones(N) / N
    return f / f.sum()

def construct():
    t0 = time.time()
    deadline = t0 + DEADLINE - 10.0  # safety margin

    best_h = None
    best_val = -1.0

    # Seed
    seed_h = np.ones(N)
    seed_val = _value(seed_h)
    if seed_val > best_val:
        best_val = seed_val
        best_h = seed_h.copy()

    # 1) Analytic humps + local search
    params = []
    for power in [1.5, 2.0, 2.5, 3.0, 4.0]:
        params.append((power, 0.5, 1.0, 0.0))
    for flat in [0.1, 0.2, 0.3]:
        for power in [2.0, 3.0]:
            params.append((power, 0.5, 1.0, flat))
    for power in [2.0, 3.0, 4.0]:
        params.append((power, 0.5, 0.9, 0.0))
    params.append((5.0, 0.5, 1.0, 0.0))

    for power, shift, width, flat in params:
        if time.time() > deadline - 5.0:
            break
        h_init = _construct_hump(power, shift, width, flat)
        h_opt, v = _local_search(h_init, max_iters=2000)
        if v > best_val:
            best_val = v
            best_h = h_opt.copy()

    # 2) DST-based spectral optimization with L-BFGS-B
    rng = np.random.default_rng(42)
    
    # Initialize spectral parameters from good starting points
    spectral_starts = []
    
    # From the best h found so far
    if best_h is not None:
        spec0 = _spectrum_from_h(best_h)
        spec0 = np.maximum(spec0, 0.0)  # non-negative spectral envelope
        spectral_starts.append(spec0)
    
    # From uniform
    spec_uniform = _spectrum_from_h(np.ones(N))
    spec_uniform = np.maximum(spec_uniform, 0.0)
    spectral_starts.append(spec_uniform)
    
    # Random spectral envelopes (non-negative)
    for r in range(6):
        if time.time() > deadline - 10.0:
            break
        # Create a smooth non-negative spectral envelope
        # Low frequencies should be larger for a broad peak
        freqs = np.linspace(0, 1, N)
        decay = rng.uniform(1.0, 4.0)
        spec_rand = np.exp(-decay * freqs) + rng.random(N) * 0.3
        spec_rand = np.maximum(spec_rand, 0.0)
        spectral_starts.append(spec_rand)
    
    for spec_init in spectral_starts:
        if time.time() > deadline - 10.0:
            break
        
        # Use L-BFGS-B on the smooth surrogate
        res = minimize(
            _neg_value_smooth,
            spec_init,
            method='L-BFGS-B',
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 2000}
        )
        
        if res.success or res.fun < 0:
            h_opt = _h_from_spectrum(res.x)
            h_opt = np.clip(h_opt, 0.0, None)
            s = h_opt.sum()
            if s > 1e-12:
                h_opt = h_opt / s
                v = _value(h_opt)
                if v > best_val:
                    best_val = v
                    best_h = h_opt.copy()
        
        # Also try with higher beta (sharper max approximation)
        if time.time() > deadline - 5.0:
            break
        res2 = minimize(
            lambda s: _neg_value_smooth(s, beta=100.0),
            spec_init,
            method='L-BFGS-B',
            options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 1000}
        )
        if res2.success or res2.fun < 0:
            h_opt = _h_from_spectrum(res2.x)
            h_opt = np.clip(h_opt, 0.0, None)
            s = h_opt.sum()
            if s > 1e-12:
                h_opt = h_opt / s
                v = _value(h_opt)
                if v > best_val:
                    best_val = v
                    best_h = h_opt.copy()

    # 3) Random restarts with coordinate ascent (keep from original)
    for r in range(5):
        if time.time() > deadline - 5.0:
            break
        x = np.linspace(0, 1, N)
        center = rng.uniform(0.4, 0.6)
        width = rng.uniform(0.7, 1.0)
        base = (1.0 - np.clip(np.abs(x - center) / (0.5 * width), 0, 1)) ** rng.uniform(1.5, 4.0)
        noise = rng.random(N) * 0.1
        h_init = base + noise
        h_init = np.maximum(h_init, 0.0)
        if h_init.sum() <= 0:
            h_init = np.ones(N)
        h_init = h_init / h_init.sum()
        h_opt, v = _local_search(h_init, max_iters=1500)
        if v > best_val:
            best_val = v
            best_h = h_opt.copy()

    # 4) Final fine-tune on the best
    if best_h is not None and time.time() < deadline - 10.0:
        h_final, v_final = _local_search(best_h, max_iters=5000)
        if v_final > best_val:
            best_val = v_final
            best_h = h_final

    # 5) One more spectral polish on the best
    if best_h is not None and time.time() < deadline - 5.0:
        spec_best = _spectrum_from_h(best_h)
        spec_best = np.maximum(spec_best, 0.0)
        res_final = minimize(
            lambda s: _neg_value_smooth(s, beta=200.0),
            spec_best,
            method='L-BFGS-B',
            options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10, 'maxfun': 3000}
        )
        if res_final.success or res_final.fun < 0:
            h_opt = _h_from_spectrum(res_final.x)
            h_opt = np.clip(h_opt, 0.0, None)
            s = h_opt.sum()
            if s > 1e-12:
                h_opt = h_opt / s
                v = _value(h_opt)
                if v > best_val:
                    best_val = v
                    best_h = h_opt.copy()

    if best_h is None:
        best_h = np.ones(N)

    # Ensure nonnegative and finite
    best_h = np.clip(best_h, 0.0, None)
    best_h = np.where(np.isfinite(best_h), best_h, 0.0)
    
    return [float(x) for x in best_h]
# EVOLVE-BLOCK-END
