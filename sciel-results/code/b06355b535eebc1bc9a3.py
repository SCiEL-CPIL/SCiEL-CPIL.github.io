"""Second autocorrelation inequality - log-softmax parametric optimization with diverse shapes."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50

def _compute_value(h):
    h = np.asarray(h, dtype=float)
    g = np.convolve(h, h)
    m = len(g)
    y = np.concatenate(([0.0], g, [0.0]))
    w = 1.0 / (m + 1)
    y_sq = y[:m+1]**2 + y[:m+1]*y[1:m+2] + y[1:m+2]**2
    l2 = (w / 3.0) * np.sum(y_sq)
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    if norm_1 * norm_inf < 1e-300:
        return 0.0
    return l2 / (norm_1 * norm_inf)

def _h_from_log_params(params):
    """Convert log-params to normalized nonnegative heights via softmax."""
    p = np.asarray(params, dtype=float)
    p_shift = p - np.max(p)
    e = np.exp(p_shift)
    s = e.sum()
    if s < 1e-300:
        return np.ones(N) / N
    return e / s

def _neg_value(params):
    h = _h_from_log_params(params)
    return -_compute_value(h)

def _coordinate_descent_log(params, max_sweeps=20, tol=1e-12, deadline=None):
    """Coordinate descent in log-space with golden section search per coordinate."""
    params = np.array(params, dtype=float)
    h = _h_from_log_params(params)
    best = _compute_value(h)
    idx = np.arange(N)
    
    for sweep in range(max_sweeps):
        if deadline is not None and time.time() > deadline - 3:
            break
        improved = False
        for i in range(N):
            if deadline is not None and time.time() > deadline - 3:
                break
            old_val = _compute_value(_h_from_log_params(params))
            
            # Search bounds in log-space around current value
            cur_log = params[i]
            lo = cur_log - 5.0
            hi = cur_log + 5.0
            
            gr = (np.sqrt(5) - 1) / 2
            a = lo + (1 - gr) * (hi - lo)
            b = lo + gr * (hi - lo)
            
            params_a = params.copy()
            params_b = params.copy()
            params_a[i] = a
            params_b[i] = b
            fa = _neg_value(params_a)
            fb = _neg_value(params_b)
            
            for _ in range(35):
                if deadline is not None and time.time() > deadline - 3:
                    break
                if fa < fb:
                    hi = b
                    b = a
                    fb = fa
                    a = lo + (1 - gr) * (hi - lo)
                    params_a[i] = a
                    fa = _neg_value(params_a)
                else:
                    lo = a
                    a = b
                    fa = fb
                    b = lo + gr * (hi - lo)
                    params_b[i] = b
                    fb = _neg_value(params_b)
            
            best_i = (a + b) / 2
            params_new = params.copy()
            params_new[i] = best_i
            val_new = -_neg_value(params_new)
            
            if val_new > old_val + tol:
                params = params_new
                improved = True
                best = val_new
        
        if not improved:
            break
    
    return params, best

def _generate_shape_candidates(rng):
    """Generate a diverse set of candidate initial points as log-params."""
    candidates = []
    x = np.linspace(-1, 1, N)
    xh = np.linspace(-0.5, 0.5, N)
    idx = np.arange(N)
    center = (N - 1) / 2.0
    
    def to_log_params(h):
        h = np.maximum(h, 1e-10)
        h = h / h.sum()
        return np.log(h)
    
    # Uniform
    candidates.append(to_log_params(np.ones(N)))
    
    # Gaussian family (various widths)
    for a in [0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256]:
        candidates.append(to_log_params(np.exp(-a * x**2)))
    
    # Super-Gaussian / sub-Gaussian
    for p in [0.5, 1, 1.5, 2, 3, 4, 6, 8]:
        for a in [1, 2, 4, 8, 16]:
            candidates.append(to_log_params(np.exp(-a * np.abs(x)**p)))
    
    # Polynomial / power shapes
    for k in [0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10, 12]:
        candidates.append(to_log_params(np.maximum(0, 1 - np.abs(x)**k) + 1e-8))
    
    # Cosine-based
    for mult in [0.5, 1, 1.5, 2, 2.5, 3]:
        candidates.append(to_log_params(np.maximum(0, np.cos(mult * np.pi * xh)) + 1e-8))
        candidates.append(to_log_params(np.maximum(0, np.cos(mult * np.pi * xh)**2) + 1e-8))
    
    # Sine-based
    for mult in [0.5, 1, 1.5, 2, 2.5, 3]:
        candidates.append(to_log_params(np.maximum(0, np.sin(mult * np.pi * xh)) + 1e-8))
        candidates.append(to_log_params(np.maximum(0, np.sin(mult * np.pi * xh)**2) + 1e-8))
    
    # Bimodal
    for c in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
        for a in [4, 8, 12, 20, 40, 80, 120]:
            candidates.append(to_log_params(np.exp(-a*(x-c)**2) + np.exp(-a*(x+c)**2)))
    
    # Plateau with smooth edges
    for start in [0, 2, 5, 8, 10, 12, 15]:
        for end in [N-start, N-2, N-5, N-8, N-10]:
            if end > start + 4:
                plat = np.zeros(N)
                plat[start:end] = 1.0
                edge = 3
                for j in range(start, min(start+edge, N)):
                    plat[j] = (j - start + 1) / (edge + 1)
                for j in range(max(end-edge, 0), end):
                    plat[j] = (end - j) / (edge + 1)
                candidates.append(to_log_params(plat + 1e-8))
    
    # Triangle
    candidates.append(to_log_params(np.maximum(0, 1 - 2*np.abs(xh)) + 1e-8))
    candidates.append(to_log_params(np.maximum(0, 1 - 4*np.abs(xh)) + 1e-8))
    candidates.append(to_log_params(np.maximum(0, 1 - 1.5*np.abs(xh)) + 1e-8))
    
    # Sech / Lorentzian-like
    for a in [1, 2, 3, 4, 6, 8, 12]:
        candidates.append(to_log_params(1.0 / np.cosh(a * x)))
        candidates.append(to_log_params(1.0 / (1 + (a*x)**2)))
    
    # Asymmetric
    for asym in [0.3, 0.5, 1.0, 2.0, 3.0, 5.0]:
        candidates.append(to_log_params(np.exp(-x**2 * (1 + asym*x)) + 1e-8))
        candidates.append(to_log_params(np.exp(-x**2 * (1 - asym*x)) + 1e-8))
    
    # W-shaped / multi-lobe
    for nlobes in [3, 4, 5]:
        for width in [0.1, 0.15, 0.2, 0.3, 0.4]:
            h_w = np.zeros(N)
            for k in range(nlobes):
                c = -1 + 2*k/(nlobes-1)
                h_w += np.exp(-(x-c)**2/(2*width**2))
            candidates.append(to_log_params(h_w + 1e-8))
    
    # Random smooth (low-frequency Fourier)
    for _ in range(30):
        nfreq = rng.integers(2, 10)
        coeffs = rng.standard_normal(nfreq)
        phases = rng.uniform(0, 2*np.pi, nfreq)
        h_f = np.ones(N)
        for k in range(1, nfreq+1):
            h_f += coeffs[k-1] * np.cos(2*np.pi*k*xh + phases[k-1])
        candidates.append(to_log_params(np.abs(h_f) + 0.01))
    
    # Random (raw)
    for _ in range(20):
        h_r = np.abs(rng.standard_normal(N)) + 0.01
        candidates.append(to_log_params(h_r))
    
    # Concentrated center
    for width in [2, 3, 5, 7, 10, 15, 20]:
        candidates.append(to_log_params(np.exp(-((idx - center)**2) / (2*width**2))))
    
    # Edge-concentrated
    for width in [2, 3, 5, 7, 10]:
        candidates.append(to_log_params(np.exp(-((idx)**2) / (2*width**2)) + np.exp(-((idx - (N-1))**2) / (2*width**2))))
    
    # Single spike
    for pos in [0, 5, 10, 15, 20, 24, 25, 29, 35, 40, 45, 49]:
        spike = np.zeros(N)
        spike[pos] = 1.0
        candidates.append(to_log_params(spike + 1e-6))
    
    # Two spikes
    for p1 in [5, 10, 15, 20, 25]:
        for p2 in [30, 35, 40, 45, 49]:
            if p2 > p1:
                two = np.zeros(N)
                two[p1] = 1.0
                two[p2] = 1.0
                candidates.append(to_log_params(two + 1e-6))
    
    # Step functions
    for cut in [5, 10, 15, 20, 25, 30, 35, 40, 45]:
        step = np.zeros(N)
        step[cut:] = 1.0
        candidates.append(to_log_params(step + 1e-6))
        step2 = np.zeros(N)
        step2[:cut] = 1.0
        candidates.append(to_log_params(step2 + 1e-6))
    
    # Power-law decay from center
    for alpha in [0.5, 1, 1.5, 2, 3, 4]:
        dist = np.abs(idx - center)
        candidates.append(to_log_params(1.0 / (1 + dist**alpha)))
    
    # Alternating
    for freq in [1, 2, 3, 5]:
        alt = 1 + 0.5*np.cos(2*np.pi*freq*idx/N)
        candidates.append(to_log_params(np.maximum(alt, 1e-6)))
    
    return candidates

def construct():
    t0 = time.time()
    deadline = t0 + 145.0
    
    best_params = np.zeros(N)
    best_val = _compute_value(_h_from_log_params(best_params))
    
    def try_optimize(params0, method='L-BFGS-B', maxiter=500, **kwargs):
        nonlocal best_params, best_val
        if time.time() > deadline - 5:
            return
        try:
            if method == 'L-BFGS-B':
                opts = {'maxiter': maxiter, 'gtol': 1e-10, 'maxfun': 5000}
            elif method == 'Nelder-Mead':
                opts = {'maxiter': maxiter, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            elif method == 'Powell':
                opts = {'maxiter': maxiter, 'xtol': 1e-12, 'ftol': 1e-14}
            else:
                opts = {}
            res = minimize(_neg_value, params0, method=method, options=opts)
            if res.fun < -best_val + 1e-15:
                best_val = -res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    def try_coordinate(params0, max_sweeps=20):
        nonlocal best_params, best_val
        if time.time() > deadline - 3:
            return
        try:
            p_new, val_new = _coordinate_descent_log(params0, max_sweeps=max_sweeps, deadline=deadline)
            if val_new > best_val + 1e-15:
                best_val = val_new
                best_params = p_new
        except Exception:
            pass
    
    rng = np.random.default_rng(50)
    candidates = _generate_shape_candidates(rng)
    
    # Phase 1: L-BFGS-B from diverse candidates
    phase1_deadline = t0 + 55
    for p0 in candidates:
        if time.time() > phase1_deadline:
            break
        try_optimize(p0, 'L-BFGS-B', maxiter=400)
    
    # Phase 2: Coordinate descent on best
    try_coordinate(best_params, max_sweeps=25)
    
    # Phase 3: Powell refinement
    if time.time() < deadline - 35:
        try_optimize(best_params, 'Powell', maxiter=500)
    
    # Phase 4: More random restarts
    phase4_deadline = t0 + 95
    for seed in [1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103]:
        if time.time() > phase4_deadline:
            break
        rng2 = np.random.default_rng(seed)
        for _ in range(2):
            if time.time() > phase4_deadline:
                break
            p0 = rng2.standard_normal(N) * 0.5
            try_optimize(p0, 'L-BFGS-B', maxiter=300)
    
    # Phase 5: Coordinate descent again
    try_coordinate(best_params, max_sweeps=25)
    
    # Phase 6: Nelder-Mead polish
    if time.time() < deadline - 25:
        try_optimize(best_params, 'Nelder-Mead', maxiter=2000)
    
    # Phase 7: Final coordinate descent with tighter tolerance
    try_coordinate(best_params, max_sweeps=30)
    
    # Phase 8: Final Powell with tighter tolerance
    if time.time() < deadline - 15:
        try_optimize(best_params, 'Powell', maxiter=500)
    
    # Phase 9: Final NM polish
    if time.time() < deadline - 8:
        try_optimize(best_params, 'Nelder-Mead', maxiter=3000)
    
    # Final coordinate descent
    try_coordinate(best_params, max_sweeps=15)
    
    # Convert to heights
    best_h = _h_from_log_params(best_params)
    best_h = np.maximum(best_h, 0)
    s = best_h.sum()
    if s < 1e-300:
        best_h = np.ones(N) / N
    else:
        best_h = best_h / s
    return list(best_h)
