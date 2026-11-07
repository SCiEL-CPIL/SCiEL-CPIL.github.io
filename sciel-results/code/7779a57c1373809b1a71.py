"""Second autocorrelation inequality - optimized with refined soft-max and robust local polish."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
N_TENT_KNOTS = 6

def _h_from_tent(knots, amps):
    """Piecewise linear tent basis with variable knots and amplitudes.
    knots: array of shape (K,) with values in [0, N-1], sorted
    amps: array of shape (K,) with positive values
    Returns h of shape (N,).
    """
    h = np.zeros(N)
    x = np.arange(N, dtype=float)
    K = len(knots)
    if K == 0:
        return np.ones(N) / N
    for i in range(K):
        k = knots[i]
        a = amps[i]
        # Tent centered at k with support [k-1, k+1]
        # Actually let's use a wider tent: support [k-2, k+2] for smoother
        # Let's use a simple tent: linear from k-1 to k (rising) and k to k+1 (falling)
        # But to ensure non-negativity and better coverage, use a wider basis
        # Let's use: h[j] += a * max(0, 1 - abs(j - k)) for j in [k-1, k+1]
        # This gives a triangular shape
        for j in range(max(0, int(k)-1), min(N, int(k)+2)):
            dist = abs(j - k)
            if dist <= 1.0:
                h[j] += a * (1.0 - dist)
    # Add a small baseline to ensure non-negativity and avoid zeros
    h += 1e-8
    return h / h.sum()

def _tent_value(params):
    """Objective for tent parameterization.
    params: first K are knot positions (in [0, N-1]), next K are log amplitudes.
    """
    K = N_TENT_KNOTS
    knots = params[:K]
    log_amps = params[K:2*K]
    amps = np.exp(log_amps - np.max(log_amps))
    h = _h_from_tent(knots, amps)
    return _compute_value(h)

def _neg_tent_value(params):
    return -_tent_value(params)

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
    """Softmax parameterization for non-negative, normalized h."""
    p = params - np.max(params)
    e = np.exp(p)
    h = e / e.sum()
    return h

def _logsumexp(x, beta):
    """Compute log-sum-exp of x with temperature 1/beta."""
    x = np.asarray(x, dtype=float)
    if len(x) == 0: return -np.inf
    xm = np.max(x)
    if xm < -1e300:
        return -np.inf
    z = beta * (x - xm)
    s = np.sum(np.exp(z))
    if s == 0: return -np.inf
    return xm + np.log(s) / beta

def _soft_value(params, beta):
    """Smooth surrogate for the objective. Replaces max|g| with soft-max(|g|)."""
    h = _h_from_params(params)
    g = np.convolve(h, h)
    m = len(g)
    y = np.concatenate(([0.0], g, [0.0]))
    w = 1.0 / (m + 1)
    y_sq = y[:m+1]**2 + y[:m+1]*y[1:m+2] + y[1:m+2]**2
    l2 = (w / 3.0) * np.sum(y_sq)
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    abs_g = np.abs(g)
    soft_inf = _logsumexp(abs_g, beta)
    if norm_1 < 1e-300:
        return 0.0
    val = l2 / (norm_1 * np.exp(soft_inf))
    return val

def _neg_soft_value(params, beta):
    return -_soft_value(params, beta)

def _neg_value(params):
    h = _h_from_params(params)
    return -_compute_value(h)

def construct():
    t0 = time.time()
    deadline = t0 + 148.0

    best_h = np.ones(N)
    best_val = _compute_value(best_h)
    
    def update_best(h_new, v_new):
        nonlocal best_h, best_val
        if v_new > best_val:
            best_val = v_new
            best_h = h_new.copy()

    def try_optimize(params0, method='L-BFGS-B', maxiter=1000, use_soft=True, beta=50.0, **kwargs):
        nonlocal best_h, best_val
        if time.time() > deadline - 10:
            return
        
        if use_soft:
            objective = lambda p: _neg_soft_value(p, beta)
        else:
            objective = _neg_value
            
        try:
            if method == 'L-BFGS-B':
                res = minimize(objective, params0, method='L-BFGS-B', 
                               options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-10, 'disp': False})
            elif method == 'Nelder-Mead':
                res = minimize(objective, params0, method='Nelder-Mead',
                               options={'maxiter': maxiter, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
            elif method == 'Powell':
                res = minimize(objective, params0, method='Powell',
                               options={'maxiter': maxiter, 'xtol': 1e-10, 'ftol': 1e-12})
            else:
                return
                
            h_new = _h_from_params(res.x)
            v_new = _compute_value(h_new)
            update_best(h_new, v_new)
        except Exception:
            pass

    def coordinate_refine(h, n_iter=50):
        nonlocal best_h, best_val
        h = np.asarray(h, dtype=float).copy()
        h = np.maximum(h, 1e-15)
        h = h / h.sum()
        v = _compute_value(h)
        update_best(h, v)
        
        step_sizes = [0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]
        
        for _ in range(n_iter):
            if time.time() > deadline - 10:
                break
            improved = False
            for i in range(N):
                for delta_frac in step_sizes:
                    delta = delta_frac * h[i]
                    if delta < 1e-15: continue
                    
                    h_new = h.copy()
                    h_new[i] += delta
                    h_new = np.maximum(h_new, 1e-15)
                    h_new = h_new / h_new.sum()
                    v_new = _compute_value(h_new)
                    if v_new > best_val + 1e-14:
                        h = h_new
                        best_val = v_new
                        improved = True
                        break
                        
                    h_new = h.copy()
                    h_new[i] -= delta
                    if h_new[i] < 1e-15:
                        h_new[i] = 1e-15
                    h_new = h_new / h_new.sum()
                    v_new = _compute_value(h_new)
                    if v_new > best_val + 1e-14:
                        h = h_new
                        best_val = v_new
                        improved = True
                        break
            if not improved:
                break
        return best_h, best_val

    candidates = []
    x = np.linspace(-1, 1, N)
    xh = np.linspace(-0.5, 0.5, N)

    candidates.append(np.ones(N))
    
    for a in [0.3, 0.5, 1, 2, 4, 8, 16, 32, 64]:
        candidates.append(np.exp(-a * x**2))
        
    for c in [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
        for a in [2, 4, 8, 16, 32]:
            candidates.append(np.exp(-a*(x-c)**2) + np.exp(-a*(x+c)**2))
            
    for k in [1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8]:
        candidates.append(np.maximum(0, 1 - np.abs(x)**k))
        
    for freq in [0.5, 1, 1.5, 2, 2.5, 3]:
        candidates.append(np.maximum(0, np.cos(freq * np.pi * xh)))
    
    for start, end in [(0, 50), (3, 47), (5, 45), (8, 42), (10, 40), (12, 38), (15, 35), (20, 30), (25, 25)]:
        plat = np.zeros(N)
        plat[start:end] = 1.0
        candidates.append(plat)
        
    rng = np.random.default_rng(42)
    for _ in range(25):
        coeffs = rng.standard_normal(8)
        h_f = np.ones(N)
        for k in range(1, 9):
            h_f += coeffs[k-1] * np.cos(2*np.pi*k*xh) * (1/k)
        h_f = np.abs(h_f) + 0.05
        candidates.append(h_f)
        
    for _ in range(15):
        h_r = np.zeros(N)
        n_spikes = rng.integers(3, 8)
        idx = rng.choice(N, size=n_spikes, replace=False)
        h_r[idx] = 1.0
        from scipy.ndimage import gaussian_filter1d
        h_r = gaussian_filter1d(h_r, rng.uniform(1.0, 4.0))
        h_r = np.maximum(h_r, 1e-5)
        candidates.append(h_r)
    
    # Add some asymmetric shapes
    for _ in range(10):
        h_asym = np.exp(-3 * (x + rng.uniform(-0.2, 0.2))**2) * np.exp(-1 * (x - rng.uniform(-0.3, 0.3))**2)
        candidates.append(h_asym)

    # Add tent-basis optimized candidates
    # First, try symmetric tent configurations
    for k_sym in [2, 3, 4, 5]:
        if k_sym <= N_TENT_KNOTS:
            knots_sym = np.linspace(5, N-6, k_sym)
            amps_sym = np.ones(k_sym) * 0.5
            # Pad to N_TENT_KNOTS with zeros at edges
            knots_full = np.zeros(N_TENT_KNOTS)
            amps_full = np.zeros(N_TENT_KNOTS)
            knots_full[:k_sym] = knots_sym
            amps_full[:k_sym] = amps_sym
            # Add some at edges
            if k_sym < N_TENT_KNOTS:
                knots_full[k_sym] = 2
                amps_full[k_sym] = 0.3
                if k_sym < N_TENT_KNOTS - 1:
                    knots_full[k_sym+1] = N-3
                    amps_full[k_sym+1] = 0.3
            
            h_tent = _h_from_tent(knots_full, amps_full)
            candidates.append(h_tent)
    
    # Random tent configurations
    for _ in range(20):
        knots_rand = np.sort(rng.uniform(3, N-4, size=N_TENT_KNOTS))
        amps_rand = rng.uniform(0.2, 1.0, size=N_TENT_KNOTS)
        h_tent = _h_from_tent(knots_rand, amps_rand)
        candidates.append(h_tent)

    betas = [5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 400.0]
    
    for h0 in candidates:
        if time.time() > deadline - 30:
            break
            
        h0 = np.asarray(h0, dtype=float)
        h0 = np.maximum(h0, 1e-6)
        h0 = h0 / h0.sum()
        params0 = np.log(h0)
        
        for beta in betas:
            if time.time() > deadline - 12:
                break
            try_optimize(params0, 'L-BFGS-B', maxiter=500, use_soft=True, beta=beta)
            params0 = np.log(best_h)
            
        if time.time() < deadline - 12:
            try_optimize(np.log(best_h), 'Nelder-Mead', maxiter=3000, use_soft=False)
            
        if time.time() < deadline - 8:
            try_optimize(np.log(best_h), 'Powell', maxiter=500, use_soft=False)

    if time.time() < deadline - 8:
        coordinate_refine(best_h, n_iter=100)
        
    if time.time() < deadline - 5:
        try_optimize(np.log(best_h), 'Nelder-Mead', maxiter=10000, use_soft=False)
        
    if time.time() < deadline - 2:
        coordinate_refine(best_h, n_iter=50)

    # Tent-basis optimization pass
    if time.time() < deadline - 5:
        # Try optimizing tent parameters
        for _ in range(5):
            if time.time() > deadline - 5:
                break
            knots_init = np.sort(rng.uniform(3, N-4, size=N_TENT_KNOTS))
            amps_init = rng.uniform(0.2, 1.0, size=N_TENT_KNOTS)
            params_tent = np.concatenate([knots_init, np.log(amps_init)])
            
            try:
                res_tent = minimize(_neg_tent_value, params_tent, method='Nelder-Mead',
                                   options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True})
                knots_opt = res_tent.x[:N_TENT_KNOTS]
                amps_opt = np.exp(res_tent.x[N_TENT_KNOTS:2*N_TENT_KNOTS])
                h_tent_opt = _h_from_tent(knots_opt, amps_opt)
                v_tent = _compute_value(h_tent_opt)
                update_best(h_tent_opt, v_tent)
            except Exception:
                pass

    best_h = np.maximum(best_h, 0)
    s = best_h.sum()
    if s < 1e-300:
        best_h = np.ones(N) / N
    else:
        best_h = best_h / s
        
    return list(best_h)

# EVOLVE-BLOCK-END
