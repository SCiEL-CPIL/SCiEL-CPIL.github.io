"""Second autocorrelation inequality - Dinkelbach with DCT parameterization."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.fft import dct, idct

N = 50

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

def _coord_step(h, lam, max_sweeps=20, tol=1e-12, deadline=None):
    """Exact coordinate maximization for the auxiliary l2 - lam * n1 * ninf.

    For each coordinate i, with the rest fixed, the auxiliary as a function of t=h_i is:
      A(t) = a*t^2 + b*t + c  -  lam * (p*t + q) * max(r*t + s, C)
    where a>0, so the quadratic part is convex. The maximum on [0, ∞) is at:
      - t = 0
      - the vertex of the quadratic on the branch where g_i is the max (t <= t0)
      - the vertex on the branch where g_i < C (t >= t0)
      - the breakpoint t0
      - a far boundary
    We evaluate all candidates and pick the best.
    """
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
            # Build a, b, c for A(t) = a t^2 + b t + c
            # a = h[i]^2
            # b = 2*h[i]*h_rest[i] + 2*h[i]   (the +2h[i] is the h[i]*h[i] term in g_i)
            # c = h_rest[i]^2 + 2*h_rest[i] + sum_j h_rest[j]^2 + 2*sum_j h_rest[j]
            # but only the parts that depend on t matter for the maximization; we can
            # use a simple direct evaluation at candidates instead.

            # Determine g_i(t) and C = max_{j != i} g_j
            # g_i = sum_j h[j] * h[i + j]  (for convolution with full support)
            #      = sum_j h[j] * h[i+j] for j = 0..N-1 (with h out of range = 0)
            #      = sum over j where 0 <= i+j < N of h[j] * h[i+j]
            # When we vary t = h[i], g_i = sum_j h[j]*h[i+j] where j or i+j may equal i.
            # Split: g_i(t) = (sum over pairs where neither index is i) + t * (sum over pairs where exactly one index is i) + t^2 * (indicator that i pairs with i)
            # Pairs (j, i+j) with j != i and i+j != i  => contribution independent of t
            # Pairs (j, i+j) with j == i (i+j = 2i) or i+j == i (j=0)  => contribution t * h[k]
            #   j==i: t * h[2i] if 2i < N
            #   j==0: t * h[i] = t^2  (this is the "i pairs with i" case)
            # So g_i(t) = t^2 + t * (h[2i] if 2i<N else 0) + (g_i_fixed)
            #   where g_i_fixed = sum_{j: 0<=i+j<N, j!=i, i+j!=i} h[j]*h[i+j]

            # Compute g_i_fixed
            # Pairs (j, i+j) with j in 0..N-1, 0<=i+j<N, j != i, i+j != i (i.e., j != 0)
            # => j in {1..N-1} with i+j < N
            g_i_fixed = 0.0
            for j in range(1, N):
                k = i + j
                if k < N:
                    g_i_fixed += h[j] * h[k]
            # Note: j == i term (h[i]*h[2i]) is in this sum when i>=1 and 2i<N; we must subtract it
            if i >= 1 and 2 * i < N:
                g_i_fixed -= h[i] * h[2*i]
            # The j=0 term is h[0]*h[i] = t * h[0]; this is the "i pairs with i" part only when i=0? No.
            # Actually j=0 gives pair (0, i), contribution h[0]*h[i] = h[0]*t. This is a "t * h[0]" term.
            # So g_i(t) = t^2 (from j=0, i+j=i, pair (0,i) is h[0]*h[i] = h[0]*t, NOT t^2!)
            # Wait, I need to redo this. g_i = sum_{j: 0<=j<N, 0<=i+j<N} h[j]*h[i+j].
            # If j=0: h[0]*h[i] = h[0]*t  (linear in t, unless i=0 where it's t*t=t^2)
            # If j=i (and i>0, 2i<N): h[i]*h[2i] = t*h[2i] (linear in t)
            # If i=0: j=0 gives t^2.
            # So:
            #   if i == 0: g_0(t) = t^2 + t * sum_{j=1, j<N} h[j] + sum_{j=1, 2j<N} h[j]*h[2j]... 
            # This is getting complicated. Let me just use a simpler parameterization.

            # SIMPLER: directly evaluate the auxiliary at candidate t values.
            # The auxiliary A(t) = l2(t) - lam * n1(t) * ninf(t) where:
            #   l2(t) is quadratic in t (convex since l2 is a sum of squares)
            #   n1(t) = (1/100) * sum_j g_j(t)  is linear in t (n1 = (1/100)*(N*t + S) where S = sum of g_j not involving h_i)
            #   ninf(t) = max(g_i(t), C) where g_i(t) is quadratic in t, C = max_{j!=i} g_j is constant
            #
            # So A(t) is piecewise: on the region where g_i(t) <= C, ninf = C, so
            #   A(t) = quadratic(t) - lam * C * (linear in t)  => quadratic in t
            # On the region where g_i(t) > C, ninf = g_i(t), so
            #   A(t) = quadratic(t) - lam * (linear in t) * (quadratic in t)  => quartic in t
            #
            # The maximum could be at: t=0, the boundary g_i(t)=C, or a stationary point.
            # Given the complexity, use a 1D bracketed search with a good initial bracket.

            # Quick bracketing search
            old_t = h[i]
            h_save = h.copy()

            # Sample a grid to get a rough idea of where the max is
            t_grid = np.concatenate([
                [0.0],
                np.linspace(0.0, max(3.0 * old_t + 1.0, 1.0), 30),
            ])
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
            for _ in range(25):
                if fa < fb:
                    hi = b; b = a; fb = fa
                    a = t_lo + (1 - gr) * (t_hi - t_lo)
                    fa = _aux_obj(np.where(np.arange(N) == i, a, h_save), lam)
                else:
                    lo = a; a = b; fa = fb
                    b = t_lo + gr * (t_hi - t_lo)
                    fb = _aux_obj(np.where(np.arange(N) == i, b, h_save), lam)
            t_new = (a + b) / 2
            # Also consider t=0 and the old value
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

def _dinkelbach(h0, max_iter=40, tol=1e-9, deadline=None):
    """Dinkelbach's algorithm for the fractional program l2/(n1*ninf)."""
    h = np.maximum(np.asarray(h0, dtype=float), 0.0)
    h = h / max(h.sum(), 1e-12)
    lam = _compute_value(h)
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline - 3:
            break
        h, aux = _coord_step(h, lam, max_sweeps=15, deadline=deadline)
        new_val = _compute_value(h)
        if abs(new_val - lam) < tol * max(1.0, abs(lam)):
            break
        lam = new_val
    return h, _compute_value(h)

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

def _dct_param_optimize(h0, deadline, maxiter=200):
    """Optimize in DCT coefficient space for global shape exploration."""
    h = np.maximum(np.asarray(h0, dtype=float), 0.0)
    h = h / max(h.sum(), 1e-12)
    
    # Use DCT-II to get coefficients
    # We'll optimize the first K DCT coefficients (low frequency = global shape)
    K = 8  # Number of DCT coefficients to optimize
    
    # Initial DCT coefficients
    dct_coeffs_full = dct(h, type=2, norm='ortho')
    # Keep only first K coefficients, set rest to small value
    init_coeffs = np.zeros(N)
    init_coeffs[:K] = dct_coeffs_full[:K]
    
    def coeffs_to_h(c):
        """Convert DCT coefficients to nonnegative heights."""
        # Take absolute value to enforce nonnegativity
        h_raw = np.abs(idct(c, type=2, norm='ortho'))
        s = h_raw.sum()
        if s < 1e-12:
            return np.ones(N) / N
        return h_raw / s
    
    def negv_from_coeffs(c):
        hh = coeffs_to_h(c)
        return -_compute_value(hh)
    
    try:
        res = minimize(negv_from_coeffs, init_coeffs.copy(), method='Nelder-Mead',
                       options={'maxiter': maxiter, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
        hh = coeffs_to_h(res.x)
        v = _compute_value(hh)
        if v > _compute_value(h):
            return hh, v
    except Exception:
        pass
    return h, _compute_value(h)

def _gen_starts(rng):
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
    # Random smooth
    for _ in range(25):
        nfreq = int(rng.integers(2, 8))
        coeffs = rng.standard_normal(nfreq)
        phases = rng.uniform(0, 2*np.pi, nfreq)
        h_f = np.ones(N)
        for k in range(1, nfreq + 1):
            h_f += coeffs[k-1] * np.cos(2*np.pi*k*x + phases[k-1])
        starts.append(np.abs(h_f) + 0.02)
    # Random
    for _ in range(25):
        starts.append(np.abs(rng.standard_normal(N)) + 0.02)
    return starts

def construct():
    t0 = time.time()
    deadline = t0 + 140.0

    rng = np.random.default_rng(50)
    starts = _gen_starts(rng)

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
        # Dinkelbach
        try:
            h_d, v_d = _dinkelbach(h0, max_iter=30, deadline=deadline)
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

    # Phase 1: Dinkelbach + NM from structured starts
    for h0 in starts:
        if time.time() > deadline - 15:
            break
        try_candidate(h0)

    # Phase 2: More random restarts
    for seed in [1, 2, 3, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]:
        if time.time() > deadline - 12:
            break
        rng2 = np.random.default_rng(seed)
        for _ in range(3):
            if time.time() > deadline - 12:
                break
            h0 = np.abs(rng2.standard_normal(N)) + 0.02
            try_candidate(h0)

    # Phase 3: DCT parameterization optimization
    if time.time() < deadline - 15:
        try:
            h_dct, v_dct = _dct_param_optimize(best_h, deadline, maxiter=200)
            if v_dct > best_val:
                best_val = v_dct
                best_h = h_dct.copy()
        except Exception:
            pass

    # Phase 4: Final Dinkelbach polish on the best
    if time.time() < deadline - 5:
        try:
            h_f, v_f = _dinkelbach(best_h, max_iter=40, deadline=deadline)
            if v_f > best_val:
                best_val = v_f
                best_h = h_f.copy()
        except Exception:
            pass

    # Phase 5: Final NM polish
    if time.time() < deadline - 3:
        try:
            h_n, v_n = _nm_polish(best_h, deadline, maxiter=300)
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
