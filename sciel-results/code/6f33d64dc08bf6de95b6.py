"""Second autocorrelation inequality - optimized with gradient-based + coordinate descent + multi-start."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50

def _h_from_tent(params):
    """
    Convert tent-basis params to 50 nonnegative heights.
    params: [knot1, knot2, ..., knotK, amp1, amp2, ..., ampK]
    where K is the number of knots (5-7).
    Knots are positions in [0, N-1] (sorted), amps are positive amplitudes.
    """
    K = len(params) // 2
    knots = np.sort(params[:K])
    amps = np.abs(params[K:]) + 1e-6
    
    # Ensure knots are strictly increasing with minimum spacing
    min_spacing = 1.0
    for i in range(1, K):
        if knots[i] - knots[i-1] < min_spacing:
            knots[i] = knots[i-1] + min_spacing
    # Clamp to [0, N-1]
    knots = np.clip(knots, 0, N-1)
    
    # Build tent function: sum of triangular bumps
    idx = np.arange(N, dtype=float)
    h = np.zeros(N)
    for k in range(K):
        center = knots[k]
        # Tent: linear rise and fall around center
        # Width: use spacing to neighbors or default
        if k > 0:
            left_width = (center - knots[k-1]) / 2.0
        else:
            left_width = min(center, (N-1)/4.0)
        if k < K - 1:
            right_width = (knots[k+1] - center) / 2.0
        else:
            right_width = min((N-1) - center, (N-1)/4.0)
        
        left_width = max(left_width, 0.5)
        right_width = max(right_width, 0.5)
        
        # Left side: linear from 0 at center-left_width to amp at center
        left_mask = (idx >= center - left_width) & (idx < center)
        h[left_mask] += amps[k] * (idx[left_mask] - (center - left_width)) / left_width
        
        # Center
        h[int(round(center))] += amps[k]
        
        # Right side: linear from amp at center to 0 at center+right_width
        right_mask = (idx > center) & (idx <= center + right_width)
        h[right_mask] += amps[k] * ((center + right_width) - idx[right_mask]) / right_width
    
    h = np.maximum(h, 0.0)
    s = h.sum()
    if s < 1e-300:
        return np.ones(N) / N
    return h / s

def _neg_value_tent(params):
    h = _h_from_tent(params)
    return -_compute_value(h)

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

def _neg_value(params):
    h = _h_from_params(params)
    return -_compute_value(h)

def _h_from_params(params):
    e = np.exp(params - np.max(params))
    h = e / e.sum()
    return h

def _gradient(params):
    """Compute gradient of -value w.r.t. log-params via finite differences."""
    h = _h_from_params(params)
    base_val = _compute_value(h)
    grad = np.zeros(N)
    eps = 1e-6
    for i in range(N):
        params_p = params.copy()
        params_p[i] += eps
        h_p = _h_from_params(params_p)
        val_p = _compute_value(h_p)
        grad[i] = -(val_p - base_val) / eps
    return grad

def _coordinate_descent(h, max_sweeps=20, tol=1e-10, deadline=None):
    """Coordinate descent: optimize each h[i] by golden section search."""
    h = np.array(h, dtype=float)
    h = np.maximum(h, 1e-10)
    h = h / h.sum()
    best = _compute_value(h)
    
    for sweep in range(max_sweeps):
        if deadline is not None and time.time() > deadline - 2:
            break
        improved = False
        for i in range(N):
            if deadline is not None and time.time() > deadline - 2:
                break
            old_val = _compute_value(h)
            old_hi = h[i]
            
            lo = 1e-8
            hi = max(2.0 * old_hi + 0.1, 0.5)
            
            gr = (np.sqrt(5) - 1) / 2
            a = lo + (1 - gr) * (hi - lo)
            b = lo + gr * (hi - lo)
            fa = _compute_value(np.where(np.arange(N) == i, a, h))
            fb = _compute_value(np.where(np.arange(N) == i, b, h))
            
            for _ in range(30):
                if deadline is not None and time.time() > deadline - 2:
                    break
                if fa < fb:
                    hi = b
                    b = a
                    fb = fa
                    a = lo + (1 - gr) * (hi - lo)
                    fa = _compute_value(np.where(np.arange(N) == i, a, h))
                else:
                    lo = a
                    a = b
                    fa = fb
                    b = lo + gr * (hi - lo)
                    fb = _compute_value(np.where(np.arange(N) == i, b, h))
            
            best_i = (a + b) / 2
            val_i = _compute_value(np.where(np.arange(N) == i, best_i, h))
            
            if val_i > old_val + tol:
                h[i] = best_i
                h = h / h.sum()
                improved = True
                best = _compute_value(h)
        
        if not improved:
            break
    
    return h, best

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 145.0

    best_h = np.ones(N)
    best_val = _compute_value(best_h)

    def try_optimize(params0, method='Nelder-Mead', maxiter=3000, **kwargs):
        nonlocal best_h, best_val
        if time.time() > deadline - 5:
            return
        try:
            if method == 'Nelder-Mead':
                opts = {'maxiter': maxiter, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            elif method == 'L-BFGS-B':
                opts = {'maxiter': maxiter, 'gtol': 1e-7}
            elif method == 'Powell':
                opts = {'maxiter': maxiter, 'xtol': 1e-9, 'ftol': 1e-12}
            elif method == 'CG':
                opts = {'maxiter': maxiter, 'gtol': 1e-7}
            else:
                opts = {}
            res = minimize(_neg_value, params0, method=method, options=opts)
            if res.fun < -best_val:
                best_val = -res.fun
                best_h = _h_from_params(res.x)
        except Exception:
            pass

    def try_coordinate(h0, max_sweeps=20):
        nonlocal best_h, best_val
        if time.time() > deadline - 3:
            return
        try:
            h_new, val_new = _coordinate_descent(h0, max_sweeps=max_sweeps, deadline=deadline)
            if val_new > best_val:
                best_val = val_new
                best_h = h_new
        except Exception:
            pass

    rng = np.random.default_rng(50)
    x = np.linspace(-1, 1, N)
    xh = np.linspace(-0.5, 0.5, N)

    # Generate diverse initial candidates
    candidates = []
    # Uniform
    candidates.append(np.ones(N))
    # Various Gaussians
    for a in [0.3, 0.5, 1, 2, 4, 8, 16, 32, 64, 100]:
        candidates.append(np.exp(-a * x**2))
    # Super-Gaussians
    for p in [1, 1.5, 2, 3, 4, 6]:
        for a in [1, 2, 4, 8]:
            candidates.append(np.exp(-a * np.abs(x)**p))
    # Polynomials
    for k in [1.5, 2, 2.5, 3, 4, 5, 6, 8, 10]:
        candidates.append(np.maximum(0, 1 - np.abs(x)**k))
    # Cosine
    candidates.append(np.maximum(0, np.cos(np.pi * xh)))
    candidates.append(np.maximum(0, np.cos(2 * np.pi * xh)))
    candidates.append(np.maximum(0, np.cos(np.pi * xh)**2))
    candidates.append(np.maximum(0, np.cos(1.5 * np.pi * xh)))
    candidates.append(np.maximum(0, np.cos(0.5 * np.pi * xh)))
    # Sine
    candidates.append(np.maximum(0, np.sin(np.pi * xh)))
    candidates.append(np.maximum(0, np.sin(np.pi * xh)**2))
    candidates.append(np.maximum(0, np.sin(2 * np.pi * xh)))
    # Bimodal
    for c in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
        for a in [4, 8, 12, 20, 40, 80]:
            candidates.append(np.exp(-a*(x-c)**2) + np.exp(-a*(x+c)**2))
    # Plateau-like with smooth edges
    for start, end in [(0,50),(2,48),(5,45),(10,40),(15,35),(3,47),(7,43)]:
        plat = np.zeros(N)
        plat[start:end] = 1.0
        for j in range(start, min(start+3, N)):
            plat[j] = (j - start + 1) / 4.0
        for j in range(max(end-3, 0), end):
            plat[j] = (end - j) / 4.0
        candidates.append(np.maximum(plat, 1e-6))
    # Triangle
    candidates.append(np.maximum(0, 1 - 2*np.abs(xh)))
    candidates.append(np.maximum(0, 1 - 4*np.abs(xh)))
    # Sech-like
    candidates.append(1.0 / np.cosh(2 * x))
    candidates.append(1.0 / np.cosh(4 * x))
    candidates.append(1.0 / np.cosh(6 * x))
    # Asymmetric
    for asym in [0.5, 1.0, 2.0]:
        candidates.append(np.exp(-x**2 * (1 + asym*x)))
    # Random
    for _ in range(30):
        h_r = np.abs(rng.standard_normal(N)) + 0.01
        candidates.append(h_r)
    # Smooth random via low-frequency Fourier
    for _ in range(20):
        coeffs = rng.standard_normal(8)
        h_f = np.ones(N)
        for k in range(1, 8):
            h_f += coeffs[k-1] * np.cos(2*np.pi*k*xh)
        h_f = np.abs(h_f) + 0.01
        candidates.append(h_f)
    # Concentrated in center
    for width in [3, 5, 8, 10, 15, 20]:
        center = N // 2
        idx = np.arange(N)
        candidates.append(np.exp(-((idx - center)**2) / (2*width**2)))
    # Edge-concentrated
    for width in [3, 5, 8]:
        idx = np.arange(N)
        candidates.append(np.exp(-((idx - 0)**2) / (2*width**2)) + np.exp(-((idx - (N-1))**2) / (2*width**2)))

    # Phase 1: Quick NM from diverse candidates
    for h0 in candidates:
        if time.time() > deadline - 35:
            break
        h0 = np.asarray(h0, dtype=float)
        h0 = np.maximum(h0, 1e-6)
        h0 = h0 / h0.sum()
        params0 = np.log(h0)
        try_optimize(params0, 'Nelder-Mead', maxiter=1200)

    # Phase 2: Coordinate descent on best
    try_coordinate(best_h, max_sweeps=30)

    # Phase 3: Powell refinement on best
    if time.time() < deadline - 20:
        try_optimize(np.log(best_h), 'Powell', maxiter=500)

    # Phase 4: More NM restarts with different seeds
    for seed in [1, 2, 3, 7, 13, 42, 99, 123, 200, 300, 500, 777]:
        if time.time() > deadline - 20:
            break
        rng2 = np.random.default_rng(seed)
        for _ in range(4):
            if time.time() > deadline - 15:
                break
            h0 = np.abs(rng2.standard_normal(N)) + 0.01
            h0 = h0 / h0.sum()
            params0 = np.log(h0)
            try_optimize(params0, 'Nelder-Mead', maxiter=1000)

    # Phase 5: Coordinate descent again
    try_coordinate(best_h, max_sweeps=25)

    # Phase 6: L-BFGS-B polish
    if time.time() < deadline - 10:
        try_optimize(np.log(best_h), 'L-BFGS-B', maxiter=1000)

    # Phase 7: Final coordinate descent
    try_coordinate(best_h, max_sweeps=20)

    # Phase 8: Final Powell + coordinate
    if time.time() < deadline - 5:
        try_optimize(np.log(best_h), 'Powell', maxiter=300)
    try_coordinate(best_h, max_sweeps=15)

    # Phase 9: One more NM from best
    if time.time() < deadline - 3:
        try_optimize(np.log(best_h), 'Nelder-Mead', maxiter=800)

    # Phase 10: Tent-basis optimization with few knots
    if time.time() < deadline - 10:
        try:
            rng_tent = np.random.default_rng(42)
            for K in [4, 5, 6, 7]:
                if time.time() > deadline - 8:
                    break
                for trial in range(5):
                    if time.time() > deadline - 7:
                        break
                    # Initialize knots spread across [0, N-1]
                    knots0 = np.sort(rng_tent.uniform(2, N-3, K))
                    amps0 = rng_tent.uniform(0.5, 2.0, K)
                    params0 = np.concatenate([knots0, amps0])
                    
                    res = minimize(_neg_value_tent, params0, method='Nelder-Mead',
                                   options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
                    h_tent = _h_from_tent(res.x)
                    val_tent = _compute_value(h_tent)
                    if val_tent > best_val:
                        best_val = val_tent
                        best_h = h_tent
                        # Polish with full coordinate descent
                        try_coordinate(best_h, max_sweeps=15)
                        
                        # Try L-BFGS-B polish in log-space
                        if time.time() < deadline - 5:
                            try_optimize(np.log(np.maximum(best_h, 1e-10)), 'L-BFGS-B', maxiter=500)
        except Exception:
            pass

    # Phase 11: Final polish
    if time.time() < deadline - 3:
        try_coordinate(best_h, max_sweeps=10)
        try_optimize(np.log(np.maximum(best_h, 1e-10)), 'Powell', maxiter=200)
        try_coordinate(best_h, max_sweeps=8)

    best_h = np.maximum(best_h, 0)
    s = best_h.sum()
    if s < 1e-300:
        best_h = np.ones(N) / N
    else:
        best_h = best_h / s
    return list(best_h)
# EVOLVE-BLOCK-END
