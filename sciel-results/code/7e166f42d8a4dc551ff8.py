"""Second autocorrelation inequality - DCT parameterization + smooth surrogate + DE global search."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.fft import dct, idct

N = 50
NDC = 12  # Number of DCT coefficients to use (low-dimensional parameter space)

def _compute_value(h):
    h = np.asarray(h, dtype=float)
    if not np.all(np.isfinite(h)) or np.any(h < -1e-15):
        return 0.0
    h = np.maximum(h, 0.0)
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

def _h_from_dct(c):
    """Reconstruct nonnegative heights from DCT coefficients."""
    c = np.asarray(c, dtype=float)
    # Pad to full length
    c_full = np.zeros(N)
    c_full[:len(c)] = c
    h = idct(c_full, type=2, norm='ortho')
    # Enforce nonnegativity via softplus-like transform
    h = np.maximum(h, 0.0)
    s = h.sum()
    if s < 1e-12:
        h = np.ones(N) / N
    else:
        h = h / s
    return h

def _dct_from_h(h):
    """Extract DCT coefficients from heights."""
    h = np.asarray(h, dtype=float)
    c = dct(h, type=2, norm='ortho')
    return c[:NDC]

def _smooth_max(g, beta=50.0):
    """Soft-max approximation using log-sum-exp."""
    g = np.asarray(g, dtype=float)
    gmax = np.max(g)
    if gmax < 1e-15:
        return 0.0
    # Log-sum-exp for numerical stability
    shifted = g - gmax
    log_sum = np.log(np.sum(np.exp(beta * shifted))) / beta
    return gmax + log_sum

def _aux_obj_smooth(h, lam, beta=50.0):
    """Smooth auxiliary: l2 - lam * norm_1 * soft_max(g)."""
    h = np.maximum(np.asarray(h, dtype=float), 0.0)
    g = np.convolve(h, h)
    m = len(g)
    y = np.concatenate(([0.0], g, [0.0]))
    w = 1.0 / (m + 1)
    y_sq = y[:m+1]**2 + y[:m+1]*y[1:m+2] + y[1:m+2]**2
    l2 = (w / 3.0) * np.sum(y_sq)
    norm_1 = np.sum(g) / (m + 1)
    ninf_smooth = _smooth_max(g, beta)
    return l2 - lam * norm_1 * ninf_smooth

def _neg_value_dct(c, lam, beta=50.0):
    """Negative smooth auxiliary in DCT coefficient space."""
    h = _h_from_dct(c)
    return -_aux_obj_smooth(h, lam, beta)

def _dinkelbach_dct(c0, max_iter=50, tol=1e-10, deadline=None, beta=50.0):
    """Dinkelbach in DCT coefficient space with smooth surrogate."""
    c = np.asarray(c0, dtype=float).copy()
    h = _h_from_dct(c)
    lam = _compute_value(h)
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline - 3:
            break
        
        # Optimize the smooth auxiliary in DCT space
        try:
            res = minimize(
                _neg_value_dct, c.copy(), args=(lam, beta),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            c = res.x
        except Exception:
            pass
        
        h = _h_from_dct(c)
        new_val = _compute_value(h)
        if abs(new_val - lam) < tol * max(1.0, abs(lam)):
            break
        lam = new_val
    
    return c, _compute_value(_h_from_dct(c))

def _nm_polish(h0, deadline, maxiter=400):
    """Nelder-Mead polish on the ratio directly."""
    h = np.maximum(np.asarray(h0, dtype=float), 0.0)
    h = h / max(h.sum(), 1e-12)
    def negv(p):
        hh = np.maximum(p, 1e-12)
        hh = hh / hh.sum()
        return -_compute_value(hh)
    try:
        res = minimize(negv, h.copy(), method='Nelder-Mead',
                       options={'maxiter': maxiter, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
        hh = np.maximum(res.x, 0.0)
        s = hh.sum()
        if s > 1e-12:
            hh = hh / s
        v = _compute_value(hh)
        if v > _compute_value(h):
            return hh, v
    except Exception:
        pass
    return h, _compute_value(h)

def _de_global(rng, pop_size=40, max_gen=30, deadline=None, beta=50.0):
    """Differential evolution in DCT coefficient space."""
    # Initial population
    pop = []
    for _ in range(pop_size):
        c = rng.standard_normal(NDC) * 0.5
        pop.append(c)
    
    # Evaluate initial population
    fitness = np.array([_compute_value(_h_from_dct(c)) for c in pop])
    best_idx = np.argmax(fitness)
    best_c = pop[best_idx].copy()
    best_val = fitness[best_idx]
    
    CR = 0.8
    F = 0.5
    
    for gen in range(max_gen):
        if deadline is not None and time.time() > deadline - 5:
            break
        
        for i in range(pop_size):
            if deadline is not None and time.time() > deadline - 5:
                break
            
            # Select three distinct individuals
            a, b, d = rng.choice(pop_size, size=3, replace=False)
            while a == b or a == d or b == d:
                a, b, d = rng.choice(pop_size, size=3, replace=False)
            
            # Differential mutation
            v = pop[a] + F * (pop[b] - pop[d])
            
            # Clip to reasonable range
            v = np.clip(v, -3.0, 3.0)
            
            # Evaluate trial
            trial_val = _compute_value(_h_from_dct(v))
            
            # Selection
            if trial_val >= fitness[i]:
                pop[i] = v
                fitness[i] = trial_val
                if trial_val > best_val:
                    best_val = trial_val
                    best_c = v.copy()
        
        # Elitism: ensure best is preserved
        if np.max(fitness) < best_val:
            pop[np.argmax(fitness)] = best_c.copy()
            fitness[np.argmax(fitness)] = best_val
    
    return best_c, best_val

def _perturb_polish(h0, deadline, n_perturb=20, scale=0.1):
    """Perturbation-based local polish."""
    h = np.maximum(np.asarray(h0, dtype=float), 0.0)
    h = h / max(h.sum(), 1e-12)
    best_h = h.copy()
    best_val = _compute_value(h)
    rng = np.random.default_rng(12345)
    
    for _ in range(n_perturb):
        if deadline is not None and time.time() > deadline - 3:
            break
        
        # Random perturbation
        pert = rng.standard_normal(N) * scale
        h_pert = np.maximum(h + pert, 0.0)
        s = h_pert.sum()
        if s < 1e-12:
            continue
        h_pert = h_pert / s
        
        # Quick NM polish from perturbed point
        try:
            def negv(p):
                hh = np.maximum(p, 1e-12)
                hh = hh / hh.sum()
                return -_compute_value(hh)
            res = minimize(negv, h_pert.copy(), method='Nelder-Mead',
                           options={'maxiter': 100, 'xatol': 1e-10, 'fatol': 1e-12})
            hh = np.maximum(res.x, 0.0)
            s = hh.sum()
            if s > 1e-12:
                hh = hh / s
            v = _compute_value(hh)
            if v > best_val:
                best_val = v
                best_h = hh.copy()
        except Exception:
            pass
    
    return best_h, best_val

def _gen_starts_dct(rng):
    """Generate diverse DCT coefficient starts."""
    starts = []
    # Uniform in DCT space
    for _ in range(30):
        starts.append(rng.standard_normal(NDC) * 0.5)
    # Low-frequency emphasis
    for _ in range(20):
        c = np.zeros(NDC)
        for k in range(NDC):
            c[k] = rng.standard_normal() * (1.0 / (k + 1))
        starts.append(c)
    # Symmetric patterns
    for _ in range(10):
        c = rng.standard_normal(NDC)
        # Enforce approximate symmetry by averaging
        c_sym = (c + c[::-1]) / 2
        starts.append(c_sym)
    return starts

def construct():
    t0 = time.time()
    deadline = t0 + 140.0

    rng = np.random.default_rng(50)
    
    best_h = np.ones(N) / N
    best_val = _compute_value(best_h)
    
    def try_dct_candidate(c0):
        nonlocal best_h, best_val
        if time.time() > deadline - 5:
            return
        c0 = np.asarray(c0, dtype=float)
        # Dinkelbach in DCT space
        try:
            c_d, v_d = _dinkelbach_dct(c0, max_iter=40, deadline=deadline, beta=50.0)
            h_d = _h_from_dct(c_d)
            if v_d > best_val:
                best_val = v_d
                best_h = h_d.copy()
        except Exception:
            pass
        # NM polish on current best
        try:
            h_n, v_n = _nm_polish(best_h, deadline, maxiter=300)
            if v_n > best_val:
                best_val = v_n
                best_h = h_n.copy()
        except Exception:
            pass

    # Phase 1: DE global search in DCT space
    if time.time() < deadline - 60:
        try:
            c_de, v_de = _de_global(rng, pop_size=30, max_gen=20, deadline=deadline, beta=50.0)
            h_de = _h_from_dct(c_de)
            if v_de > best_val:
                best_val = v_de
                best_h = h_de.copy()
        except Exception:
            pass

    # Phase 2: Dinkelbach from diverse DCT starts
    starts_dct = _gen_starts_dct(rng)
    for c0 in starts_dct:
        if time.time() > deadline - 20:
            break
        try_dct_candidate(c0)

    # Phase 3: Perturbation polish
    if time.time() < deadline - 10:
        try:
            h_p, v_p = _perturb_polish(best_h, deadline, n_perturb=30, scale=0.15)
            if v_p > best_val:
                best_val = v_p
                best_h = h_p.copy()
        except Exception:
            pass

    # Phase 4: Final high-precision Dinkelbach
    if time.time() < deadline - 5:
        try:
            c_best = _dct_from_h(best_h)
            c_f, v_f = _dinkelbach_dct(c_best, max_iter=60, tol=1e-12, deadline=deadline, beta=100.0)
            h_f = _h_from_dct(c_f)
            if v_f > best_val:
                best_val = v_f
                best_h = h_f.copy()
        except Exception:
            pass

    # Phase 5: Final NM polish
    if time.time() < deadline - 3:
        try:
            h_n, v_n = _nm_polish(best_h, deadline, maxiter=500)
            if v_n > best_val:
                best_val = v_n
                best_h = h_n.copy()
        except Exception:
            pass

    # Ensure validity
    best_h = np.maximum(np.asarray(best_h, dtype=float), 0.0)
    s = best_h.sum()
    if s < 1e-12:
        best_h = np.ones(N) / N
    else:
        best_h = best_h / s
    return list(best_h)
# EVOLVE-BLOCK-END
