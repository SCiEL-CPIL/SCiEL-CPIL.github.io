"""Second autocorrelation inequality - Dinkelbach + DCT parameterization + NM polish."""
import numpy as np
import time
from scipy.optimize import minimize, minimize_scalar
from scipy.fftpack import dct, idct

N = 50
M = N + N - 1  # 99
W = 1.0 / (M + 1)  # 1/100

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

def _aux_obj(h, lam):
    """Auxiliary: l2 - lam * norm_1 * norm_inf (for Dinkelbach)."""
    h = np.maximum(np.asarray(h, dtype=float), 0.0)
    g = np.convolve(h, h)
    m = len(g)
    y = np.concatenate(([0.0], g, [0.0]))
    w = 1.0 / (m + 1)
    y_sq = y[:m+1]**2 + y[:m+1]*y[1:m+2] + y[1:m+2]**2
    l2 = (w / 3.0) * np.sum(y_sq)
    norm_1 = np.sum(g) / (m + 1)
    norm_inf = np.max(g)
    return l2 - lam * norm_1 * norm_inf

def _dct_to_h(c):
    """Convert DCT coefficients to nonnegative heights via softplus + normalization."""
    h = idct(c, type=2, norm='ortho')
    # Softplus for smooth non-negativity
    h = np.log1p(np.exp(np.clip(h, -50, 50)))
    s = h.sum()
    if s < 1e-12:
        return np.ones(N) / N
    return h / s

def _h_to_dct(h):
    """Convert heights to DCT coefficients (for initialization)."""
    h = np.asarray(h, dtype=float)
    s = h.sum()
    if s < 1e-12:
        h = np.ones(N)
        s = N
    h = h / s
    return dct(h, type=2, norm='ortho')

def _nm_polish_dct(c0, deadline, maxiter=300):
    """Nelder-Mead polish in DCT coefficient space."""
    c = np.asarray(c0, dtype=float)
    def negv(c):
        h = _dct_to_h(c)
        return -_compute_value(h)
    try:
        res = minimize(negv, c.copy(), method='Nelder-Mead',
                       options={'maxiter': maxiter, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
        h = _dct_to_h(res.x)
        v = _compute_value(h)
        return h, v
    except Exception:
        h = _dct_to_h(c)
        return h, _compute_value(h)

def _nm_polish_h(h0, deadline, maxiter=300):
    """Nelder-Mead polish in height space directly."""
    h = np.maximum(np.asarray(h0, dtype=float), 0.0)
    s = h.sum()
    if s < 1e-12:
        h = np.ones(N) / N
    else:
        h = h / s
    def negv(p):
        hh = np.maximum(p, 1e-12)
        s2 = hh.sum()
        if s2 < 1e-12:
            return 0.0
        hh = hh / s2
        return -_compute_value(hh)
    try:
        res = minimize(negv, h.copy(), method='Nelder-Mead',
                       options={'maxiter': maxiter, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
        hh = np.maximum(res.x, 0.0)
        s2 = hh.sum()
        if s2 > 1e-12:
            hh = hh / s2
        v = _compute_value(hh)
        if v > _compute_value(h):
            return hh, v
    except Exception:
        pass
    return h, _compute_value(h)

def _coord_step(h, lam, max_sweeps=15, tol=1e-12, deadline=None):
    """Coordinate maximization for the auxiliary via 1D bracketed search."""
    h = np.maximum(np.asarray(h, dtype=float), 0.0)
    best_aux = _aux_obj(h, lam)
    N = len(h)

    for sweep in range(max_sweeps):
        if deadline is not None and time.time() > deadline - 2:
            break
        improved = False
        for i in range(N):
            if deadline is not None and time.time() > deadline - 2:
                break
            old_t = h[i]
            h_save = h.copy()

            # Sample a grid to bracket the maximum
            t_max = max(3.0 * old_t + 1.0, 1.0)
            t_grid = np.concatenate([[0.0], np.linspace(0.0, t_max, 25)])
            vals = np.array([
                _aux_obj(np.where(np.arange(N) == i, tv, h_save), lam)
                for tv in t_grid
            ])
            idx_best = int(np.argmax(vals))
            t_lo = t_grid[max(0, idx_best - 1)]
            t_hi = t_grid[min(len(t_grid) - 1, idx_best + 1)]

            # Golden section refine
            gr = (np.sqrt(5) - 1) / 2
            a = t_lo + (1 - gr) * (t_hi - t_lo)
            b = t_lo + gr * (t_hi - t_lo)
            fa = _aux_obj(np.where(np.arange(N) == i, a, h_save), lam)
            fb = _aux_obj(np.where(np.arange(N) == i, b, h_save), lam)
            for _ in range(20):
                if fa < fb:
                    t_hi = b; b = a; fb = fa
                    a = t_lo + (1 - gr) * (t_hi - t_lo)
                    fa = _aux_obj(np.where(np.arange(N) == i, a, h_save), lam)
                else:
                    t_lo = a; a = b; fa = fb
                    b = t_lo + gr * (t_hi - t_lo)
                    fb = _aux_obj(np.where(np.arange(N) == i, b, h_save), lam)
            t_new = (a + b) / 2

            candidates_t = [0.0, old_t, t_new, t_lo, t_hi]
            best_t = old_t
            best_v = _aux_obj(h_save, lam)
            for tv in candidates_t:
                if tv < 0:
                    tv = 0.0
                v = _aux_obj(np.where(np.arange(N) == i, tv, h_save), lam)
                if v > best_v:
                    best_v = v
                    best_t = tv
            h[i] = best_t

            if best_v > best_aux + tol:
                best_aux = best_v
                improved = True
        if not improved:
            break

    return h, best_aux

def _dinkelbach(h0, max_iter=30, tol=1e-9, deadline=None):
    """Dinkelbach's algorithm for the fractional program l2/(n1*ninf)."""
    h = np.maximum(np.asarray(h0, dtype=float), 0.0)
    s = h.sum()
    if s < 1e-12:
        h = np.ones(N) / N
    else:
        h = h / s
    lam = _compute_value(h)
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline - 3:
            break
        h, aux = _coord_step(h, lam, max_sweeps=12, deadline=deadline)
        new_val = _compute_value(h)
        if abs(new_val - lam) < tol * max(1.0, abs(lam)):
            break
        lam = new_val
    return h, _compute_value(h)

def _gen_structured_starts():
    """Generate diverse structured starts."""
    starts = []
    idx = np.arange(N)
    x = idx / (N - 1) - 0.5
    # Uniform
    starts.append(np.ones(N))
    # Gaussians
    for s in [0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.8]:
        starts.append(np.exp(-x**2 / (2 * s**2)))
    # Super/sub-Gaussian
    for p in [0.5, 1, 1.5, 2, 3, 4]:
        for s in [0.1, 0.2, 0.4, 0.7]:
            starts.append(np.exp(-np.abs(x) ** p / s))
    # Compact support
    for w in [0.1, 0.2, 0.3, 0.5]:
        starts.append(np.maximum(0, 1 - np.abs(x) / w))
    # Plateau
    for w in [0.1, 0.2, 0.3]:
        starts.append(np.where(np.abs(x) < w, 1.0, 0.0) + 1e-6)
    # Bimodal
    for c in [0.15, 0.25, 0.35]:
        for s in [0.05, 0.1, 0.15]:
            starts.append(np.exp(-(x - c)**2 / (2*s**2)) + np.exp(-(x + c)**2 / (2*s**2)))
    # Tri-modal
    for s in [0.08, 0.12, 0.2]:
        starts.append(np.exp(-(x - 0.4)**2/(2*s**2)) + np.exp(-x**2/(2*s**2)) + np.exp(-(x + 0.4)**2/(2*s**2)))
    # Lorentzian
    for a in [2, 5, 10, 20]:
        starts.append(1.0 / (1 + (a * x)**2))
    # Power-law
    for a in [0.5, 1, 2, 3]:
        starts.append(1.0 / (1 + np.abs(x) / 0.2) ** a)
    return starts

def _gen_random_starts(rng, n=30):
    """Generate random smooth and random starts."""
    starts = []
    idx = np.arange(N)
    x = idx / (N - 1) - 0.5
    # Random smooth (Fourier)
    for _ in range(15):
        nfreq = int(rng.integers(2, 8))
        coeffs = rng.standard_normal(nfreq)
        phases = rng.uniform(0, 2*np.pi, nfreq)
        h_f = np.ones(N)
        for k in range(1, nfreq + 1):
            h_f += coeffs[k-1] * np.cos(2*np.pi*k*x + phases[k-1])
        starts.append(np.abs(h_f) + 0.02)
    # Random
    for _ in range(15):
        starts.append(np.abs(rng.standard_normal(N)) + 0.02)
    return starts

def construct():
    t0 = time.time()
    deadline = t0 + 140.0

    rng = np.random.default_rng(50)

    best_h = np.ones(N) / N
    best_val = _compute_value(best_h)

    def try_candidate(h0):
        nonlocal best_h, best_val
        if time.time() > deadline - 5:
            return
        h0 = np.maximum(np.asarray(h0, dtype=float), 0.0)
        s = h0.sum()
        if s < 1e-12:
            return
        h0 = h0 / s

        # Dinkelbach in height space
        try:
            h_d, v_d = _dinkelbach(h0, max_iter=25, deadline=deadline)
            if v_d > best_val:
                best_val = v_d
                best_h = h_d.copy()
        except Exception:
            pass

        # DCT-space NM polish on the current best
        if time.time() < deadline - 4:
            try:
                c0 = _h_to_dct(best_h)
                h_c, v_c = _nm_polish_dct(c0, deadline, maxiter=250)
                if v_c > best_val:
                    best_val = v_c
                    best_h = h_c.copy()
            except Exception:
                pass

        # Direct NM polish
        if time.time() < deadline - 3:
            try:
                h_n, v_n = _nm_polish_h(best_h, deadline, maxiter=200)
                if v_n > best_val:
                    best_val = v_n
                    best_h = h_n.copy()
            except Exception:
                pass

    # Phase 1: Structured starts
    starts = _gen_structured_starts()
    for h0 in starts:
        if time.time() > deadline - 20:
            break
        try_candidate(h0)

    # Phase 2: Random starts
    for h0 in _gen_random_starts(rng, 20):
        if time.time() > deadline - 12:
            break
        try_candidate(h0)

    # Phase 3: DCT-space random restarts
    for seed in [1, 2, 3, 7, 11, 13, 17, 19, 23, 29]:
        if time.time() > deadline - 8:
            break
        rng2 = np.random.default_rng(seed)
        for _ in range(2):
            if time.time() > deadline - 8:
                break
            # Random DCT coefficients
            c_rand = rng2.standard_normal(N) * 0.5
            h0 = _dct_to_h(c_rand)
            try_candidate(h0)

    # Phase 4: Final Dinkelbach polish
    if time.time() < deadline - 5:
        try:
            h_f, v_f = _dinkelbach(best_h, max_iter=35, deadline=deadline)
            if v_f > best_val:
                best_val = v_f
                best_h = h_f.copy()
        except Exception:
            pass

    # Phase 5: Final DCT NM polish (high precision)
    if time.time() < deadline - 4:
        try:
            c0 = _h_to_dct(best_h)
            h_c, v_c = _nm_polish_dct(c0, deadline, maxiter=400)
            if v_c > best_val:
                best_val = v_c
                best_h = h_c.copy()
        except Exception:
            pass

    # Phase 6: Final direct NM polish (high precision)
    if time.time() < deadline - 3:
        try:
            h_n, v_n = _nm_polish_h(best_h, deadline, maxiter=400)
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
