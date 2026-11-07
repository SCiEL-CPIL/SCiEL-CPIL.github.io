"""Second autocorrelation inequality - soft-max surrogate + coordinate refinement."""
import numpy as np
import time

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

def _h_from_params(params):
    e = np.exp(params - np.max(params))
    h = e / e.sum()
    return h

def _logsumexp(x, beta):
    """Compute log-sum-exp of x with temperature 1/beta (soft-max)."""
    x = np.asarray(x, dtype=float)
    xm = np.max(x)
    if xm < -1e300:
        return -np.inf
    return xm + np.log(np.sum(np.exp(-beta * (x - xm)))) / beta

def _soft_value(params, beta):
    """Smooth surrogate: replace max|g| with soft-max of |g|."""
    h = _h_from_params(params)
    g = np.convolve(h, h)
    m = len(g)
    y = np.concatenate(([0.0], g, [0.0]))
    w = 1.0 / (m + 1)
    y_sq = y[:m+1]**2 + y[:m+1]*y[1:m+2] + y[1:m+2]**2
    l2 = (w / 3.0) * np.sum(y_sq)
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    abs_g = np.abs(g)
    soft_max = _logsumexp(abs_g, beta)  # log of soft-max
    # We want l2 / (norm_1 * max|g|). Using soft-max as surrogate for max|g|:
    # value ≈ l2 / (norm_1 * exp(soft_max))
    if norm_1 < 1e-300:
        return 0.0
    return l2 / (norm_1 * np.exp(soft_max))

def _neg_soft_value(params, beta):
    return -_soft_value(params, beta)

def _coordinate_refine(h, n_iter=200, tol=1e-10):
    """Coordinate descent on shape (scale-invariant, so fix sum=1)."""
    h = np.asarray(h, dtype=float).copy()
    h = np.maximum(h, 1e-15)
    h = h / h.sum()
    best_val = _compute_value(h)
    for it in range(n_iter):
        improved = False
        for i in range(N):
            for delta in [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, -0.01, -0.02, -0.05, -0.1]:
                h_new = h.copy()
                h_new[i] = max(h_new[i] + delta, 1e-15)
                h_new = h_new / h_new.sum()
                v = _compute_value(h_new)
                if v > best_val + tol:
                    h = h_new
                    best_val = v
                    improved = True
                    break
        if not improved:
            break
    return h, best_val

def _generate_candidates(rng):
    """Generate diverse initial shape candidates (all normalized to sum=1)."""
    candidates = []
    x = np.linspace(-1, 1, N)
    xh = np.linspace(-0.5, 0.5, N)
    
    # Uniform
    candidates.append(np.ones(N) / N)
    
    # Gaussians with varying widths
    for a in [0.5, 1, 2, 4, 8, 16, 32, 64]:
        g = np.exp(-a * x**2)
        candidates.append(g / g.sum())
    
    # Polynomials (tent-like)
    for k in [1, 2, 3, 4, 6, 8, 12]:
        p = np.maximum(0, 1 - np.abs(x)**k)
        if p.sum() > 0:
            candidates.append(p / p.sum())
    
    # Cosine bumps
    candidates.append(np.maximum(0, np.cos(np.pi * xh)) / np.maximum(0, np.cos(np.pi * xh)).sum())
    candidates.append(np.maximum(0, np.cos(2 * np.pi * xh)) / np.maximum(0, np.cos(2 * np.pi * xh)).sum())
    candidates.append(np.maximum(0, np.cos(np.pi * xh)**2) / np.maximum(0, np.cos(np.pi * xh)**2).sum())
    
    # Sine bumps
    s = np.maximum(0, np.sin(np.pi * xh))
    candidates.append(s / s.sum())
    s2 = np.maximum(0, np.sin(np.pi * xh)**2)
    candidates.append(s2 / s2.sum())
    
    # Bimodal Gaussians
    for c in [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]:
        for a in [4, 8, 12, 20, 40]:
            b = np.exp(-a*(x-c)**2) + np.exp(-a*(x+c)**2)
            candidates.append(b / b.sum())
    
    # Plateau (box) functions
    for start, end in [(0,50),(5,45),(10,40),(15,35),(20,30),(10,30),(15,35)]:
        plat = np.zeros(N)
        plat[start:end] = 1.0
        candidates.append(plat / plat.sum())
    
    # Triangle
    tri = np.maximum(0, 1 - 2*np.abs(xh))
    if tri.sum() > 0:
        candidates.append(tri / tri.sum())
    
    # Random smooth (low-frequency Fourier)
    for _ in range(15):
        coeffs = rng.standard_normal(8)
        h_f = np.ones(N)
        for k in range(1, 9):
            h_f += coeffs[k-1] * np.cos(2*np.pi*k*xh/N * N)  # k cycles over [-0.5,0.5]
        h_f = np.abs(h_f) + 0.01
        candidates.append(h_f / h_f.sum())
    
    # Pure random
    for _ in range(10):
        h_r = np.abs(rng.standard_normal(N)) + 0.01
        candidates.append(h_r / h_r.sum())
    
    # Sparse peaks
    for n_peaks in [1, 2, 3, 5]:
        for _ in range(5):
            h_sp = np.zeros(N)
            for _ in range(n_peaks):
                center = rng.integers(0, N)
                width = rng.uniform(1, 4)
                h_sp += np.exp(-((np.arange(N) - center)**2) / (2*width**2))
            h_sp = np.maximum(h_sp, 1e-6)
            candidates.append(h_sp / h_sp.sum())
    
    return candidates

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 145.0  # safety margin before 150s kill
    
    best_h = np.ones(N) / N
    best_val = _compute_value(best_h)
    
    rng = np.random.default_rng(50)
    candidates = _generate_candidates(rng)
    
    from scipy.optimize import minimize
    
    def try_soft_optimize(params0, beta, maxiter=2000, method='L-BFGS-B'):
        nonlocal best_h, best_val
        if time.time() > deadline - 5:
            return
        try:
            if method == 'L-BFGS-B':
                opts = {'maxiter': maxiter, 'gtol': 1e-8, 'maxls': 50}
            elif method == 'Powell':
                opts = {'maxiter': maxiter, 'xtol': 1e-9, 'ftol': 1e-12}
            else:
                opts = {'maxiter': maxiter}
            res = minimize(lambda p: _neg_soft_value(p, beta), params0, method=method, options=opts)
            h_new = _h_from_params(res.x)
            v = _compute_value(h_new)  # use exact value, not soft surrogate
            if v > best_val:
                best_val = v
                best_h = h_new
        except Exception:
            pass
    
    def try_nm(params0, maxiter=2000):
        nonlocal best_h, best_val
        if time.time() > deadline - 5:
            return
        try:
            opts = {'maxiter': maxiter, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True}
            res = minimize(lambda p: -_compute_value(_h_from_params(p)), params0, method='Nelder-Mead', options=opts)
            h_new = _h_from_params(res.x)
            v = _compute_value(h_new)
            if v > best_val:
                best_val = v
                best_h = h_new
        except Exception:
            pass
    
    # Phase 1: Soft-max optimization from diverse candidates
    # Use decreasing beta (coarse to fine)
    betas = [5.0, 10.0, 20.0, 50.0, 100.0, 200.0]
    
    for h0 in candidates:
        if time.time() > deadline - 30:
            break
        h0 = np.asarray(h0, dtype=float)
        h0 = np.maximum(h0, 1e-15)
        h0 = h0 / h0.sum()
        params0 = np.log(h0)
        
        # Coarse to fine soft-max
        for beta in betas:
            if time.time() > deadline - 10:
                break
            try_soft_optimize(params0, beta, maxiter=1000, method='L-BFGS-B')
            params0 = np.log(best_h)
        
        # Coordinate refinement
        if time.time() < deadline - 8:
            h_ref, v_ref = _coordinate_refine(best_h, n_iter=100)
            if v_ref > best_val:
                best_val = v_ref
                best_h = h_ref
    
    # Phase 2: Nelder-Mead from best + diverse restarts
    if time.time() < deadline - 20:
        try_nm(np.log(best_h), maxiter=3000)
    
    for seed in [1, 2, 3, 7, 13, 42, 99, 123, 200, 500]:
        if time.time() > deadline - 15:
            break
        rng2 = np.random.default_rng(seed)
        for _ in range(3):
            if time.time() > deadline - 10:
                break
            h0 = np.abs(rng2.standard_normal(N)) + 0.01
            h0 = h0 / h0.sum()
            params0 = np.log(h0)
            for beta in [10.0, 50.0, 200.0]:
                if time.time() > deadline - 5:
                    break
                try_soft_optimize(params0, beta, maxiter=800, method='L-BFGS-B')
                params0 = np.log(best_h)
            if time.time() < deadline - 5:
                try_nm(params0, maxiter=1500)
    
    # Phase 3: Final polish - coordinate descent + tight soft-max
    if time.time() < deadline - 10:
        h_ref, v_ref = _coordinate_refine(best_h, n_iter=300, tol=1e-12)
        if v_ref > best_val:
            best_val = v_ref
            best_h = h_ref
    
    if time.time() < deadline - 8:
        for beta in [500.0, 1000.0, 2000.0]:
            if time.time() > deadline - 5:
                break
            try_soft_optimize(np.log(best_h), beta, maxiter=2000, method='L-BFGS-B')
    
    if time.time() < deadline - 5:
        h_ref, v_ref = _coordinate_refine(best_h, n_iter=200, tol=1e-13)
        if v_ref > best_val:
            best_val = v_ref
            best_h = h_ref
    
    # Final: Nelder-Mead tight
    if time.time() < deadline - 3:
        try_nm(np.log(best_h), maxiter=5000)
    
    best_h = np.maximum(best_h, 0)
    s = best_h.sum()
    if s < 1e-300:
        best_h = np.ones(N) / N
    else:
        best_h = best_h / s
    return list(best_h)
# EVOLVE-BLOCK-END
