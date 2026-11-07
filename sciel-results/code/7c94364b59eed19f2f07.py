"""Second autocorrelation inequality - Dinkelbach with vectorized coordinate updates."""
import numpy as np
import time
from scipy.optimize import minimize

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

def _aux_obj_smooth(h, lam, beta=50.0):
    """Auxiliary with soft-max surrogate for norm_inf."""
    h = np.maximum(np.asarray(h, dtype=float), 0.0)
    g = np.convolve(h, h)
    m = len(g)
    y = np.concatenate(([0.0], g, [0.0]))
    w = 1.0 / (m + 1)
    y_sq = y[:m+1]**2 + y[:m+1]*y[1:m+2] + y[1:m+2]**2
    l2 = (w / 3.0) * np.sum(y_sq)
    norm_1 = np.sum(g) / (m + 1)
    # Soft-max: (1/beta) * log(sum(exp(beta * g_j)))
    g_max = np.max(g)
    soft_max = g_max + np.log(np.sum(np.exp(beta * (g - g_max)))) / beta
    return l2 - lam * norm_1 * soft_max

def _precompute_coeffs(h):
    """Precompute coefficients for fast coordinate updates.
    
    For coordinate i, g_i(t) = t^2 + t * lin_i + const_i
    where lin_i and const_i depend on other h values.
    
    Also precompute:
    - For l2: quadratic coefficients in t
    - For norm_1: linear coefficient in t
    - For norm_inf: the max of other g_j (C_i)
    """
    h = np.maximum(np.asarray(h, dtype=float), 0.0)
    g = np.convolve(h, h)
    m = len(g)
    
    # For each i, decompose g_i(t) = t^2 + t * lin_i + const_i
    # g_i = sum_{j: 0<=j<N, 0<=i+j<N} h[j]*h[i+j]
    # Terms involving h[i]=t:
    #   j=0: h[0]*h[i] = h[0]*t  (if i>0) or t^2 (if i=0)
    #   j=i: h[i]*h[2i] = t*h[2i] (if i>0 and 2i<N)
    #   i+j=i => j=0: already counted
    # So:
    #   if i == 0: g_0(t) = t^2 + t * sum_{j=1}^{N-1} h[j] + sum_{j=1, 2j<N} h[j]*h[2j]
    #   if i > 0:  g_i(t) = t * (h[0] + (h[2i] if 2i<N else 0)) + const_i
    #   where const_i = sum_{j: 0<=i+j<N, j!=0, j!=i} h[j]*h[i+j]
    
    lin = np.zeros(N)
    const = np.zeros(N)
    
    for i in range(N):
        # Linear coefficient
        if i == 0:
            lin[i] = np.sum(h[1:N])
        else:
            lin[i] = h[0]
            if 2 * i < N:
                lin[i] += h[2 * i]
        
        # Constant part: sum over j where 0<=i+j<N, j != 0, j != i (or j != 0 if i==0)
        c = 0.0
        for j in range(N):
            if j == 0 or j == i:
                continue
            k = i + j
            if 0 <= k < N:
                c += h[j] * h[k]
        const[i] = c
    
    # For l2: y = [0, g_0, ..., g_{m-1}, 0], w = 1/(m+1)
    # l2 = (w/3) * sum_{i=0}^{m} (y_i^2 + y_i*y_{i+1} + y_{i+1}^2)
    # y_0 = 0, y_{m+1} = 0
    # y_i = g_{i-1} for i=1..m
    # So l2 = (w/3) * [g_0^2 + g_0*g_1 + g_1^2 + g_1^2 + g_1*g_2 + g_2^2 + ... + g_{m-2}^2 + g_{m-2}*g_{m-1} + g_{m-1}^2 + g_{m-1}^2 + g_{m-1}*0 + 0]
    #      = (w/3) * [g_0^2 + 2*g_0*g_1 + 2*g_1^2 + 2*g_1*g_2 + 2*g_2^2 + ... + 2*g_{m-2}*g_{m-1} + 2*g_{m-1}^2]
    # Wait let me redo: 
    # i=0: y_0^2 + y_0*y_1 + y_1^2 = 0 + 0 + g_0^2
    # i=1: y_1^2 + y_1*y_2 + y_2^2 = g_0^2 + g_0*g_1 + g_1^2
    # i=2: y_2^2 + y_2*y_3 + y_3^2 = g_1^2 + g_1*g_2 + g_2^2
    # ...
    # i=m-1: y_{m-1}^2 + y_{m-1}*y_m + y_m^2 = g_{m-2}^2 + g_{m-2}*g_{m-1} + g_{m-1}^2
    # i=m: y_m^2 + y_m*y_{m+1} + y_{m+1}^2 = g_{m-1}^2 + 0 + 0
    # Sum = g_0^2 + (g_0^2 + g_0*g_1 + g_1^2) + (g_1^2 + g_1*g_2 + g_2^2) + ... + (g_{m-2}^2 + g_{m-2}*g_{m-1} + g_{m-1}^2) + g_{m-1}^2
    #      = 2*g_0^2 + 2*g_0*g_1 + 2*g_1^2 + 2*g_1*g_2 + 2*g_2^2 + ... + 2*g_{m-2}*g_{m-1} + 2*g_{m-1}^2
    #      = 2 * sum_{k=0}^{m-1} g_k^2 + 2 * sum_{k=0}^{m-2} g_k * g_{k+1}
    #      = 2 * (sum g_k^2 + sum g_k*g_{k+1})
    # where m = 2N-1 = 99
    
    # l2 = (w/3) * 2 * (sum g_k^2 + sum g_k*g_{k+1})
    #    = (2w/3) * (sum g_k^2 + sum g_k*g_{k+1})
    
    # For coordinate i, g_i(t) = t^2 + lin_i*t + const_i
    # sum g_k^2 = g_i(t)^2 + sum_{k!=i} g_k^2
    # sum g_k*g_{k+1} = g_i(t)*g_{i-1} + g_i(t)*g_{i+1} + sum_{k: k!=i, k+1!=i} g_k*g_{k+1}
    #   (with g_{-1}=g_m=0)
    
    # So l2(t) = (2w/3) * [ (t^2 + lin_i*t + const_i)^2 + S2_rest + (t^2 + lin_i*t + const_i)*(g_{i-1} + g_{i+1}) + S1_rest ]
    # where S2_rest = sum_{k!=i} g_k^2, S1_rest = sum_{k: k!=i, k+1!=i} g_k*g_{k+1}
    
    # This is a quartic in t. For the auxiliary l2 - lam * n1 * ninf:
    # n1(t) = (1/(m+1)) * (sum_{k!=i} g_k + g_i(t)) = (1/(m+1)) * (n1_rest + t^2 + lin_i*t + const_i)
    # ninf(t) = max(g_i(t), C_i) where C_i = max_{k!=i} g_k
    
    # For the smooth version, ninf_smooth(t) = soft_max of all g's, which is more complex.
    # For exact: piecewise analysis.
    
    # Precompute S2_rest, S1_rest, C_i, n1_rest for each i
    S2 = np.sum(g**2)
    S1 = np.sum(g[:-1] * g[1:])  # sum g_k * g_{k+1} for k=0..m-2
    n1_total = np.sum(g) / (m + 1)
    
    S2_rest = np.zeros(N)
    S1_rest = np.zeros(N)
    C_i = np.zeros(N)
    n1_rest = np.zeros(N)
    g_neighbors = np.zeros(N)  # g_{i-1} + g_{i+1}
    
    for i in range(N):
        S2_rest[i] = S2 - g[i]**2
        # S1_rest: remove terms involving g_i
        # Terms with g_i: g_{i-1}*g_i (if i>0) and g_i*g_{i+1} (if i<m-1)
        s1r = S1
        if i > 0:
            s1r -= g[i-1] * g[i]
        if i < m - 1:
            s1r -= g[i] * g[i+1]
        S1_rest[i] = s1r
        
        C_i[i] = np.max(np.delete(g, i))
        n1_rest[i] = (n1_total * (m + 1) - g[i]) / (m + 1)
        
        gn = 0.0
        if i > 0:
            gn += g[i-1]
        if i < m - 1:
            gn += g[i+1]
        g_neighbors[i] = gn
    
    w = 1.0 / (m + 1)
    factor = 2.0 * w / 3.0
    
    return g, lin, const, S2_rest, S1_rest, C_i, n1_rest, g_neighbors, factor, m, w

def _coord_step_fast(h, lam, max_sweeps=20, tol=1e-12, deadline=None, use_smooth=False, beta=50.0):
    """Fast coordinate maximization using precomputed coefficients.
    
    For each coordinate i, the auxiliary as a function of t=h_i is:
      A(t) = (2w/3) * [(t^2 + lin_i*t + const_i)^2 + S2_rest_i + (t^2 + lin_i*t + const_i)*gn_i + S1_rest_i]
             - lam * (n1_rest_i + (t^2 + lin_i*t + const_i)/(m+1)) * max(t^2 + lin_i*t + const_i, C_i)
    
    This is piecewise (due to max). On each branch it's a quartic.
    We find candidates: t=0, t where derivative=0 on each branch, and the breakpoint.
    """
    h = np.maximum(np.asarray(h, dtype=float), 0.0)
    best_aux = _aux_obj(h, lam)
    N = len(h)
    
    g, lin, const, S2_rest, S1_rest, C_i, n1_rest, g_neighbors, factor, m, w = _precompute_coeffs(h)
    
    for sweep in range(max_sweeps):
        if deadline is not None and time.time() > deadline - 2:
            break
        improved = False
        for i in range(N):
            if deadline is not None and time.time() > deadline - 2:
                break
            
            old_t = h[i]
            li = lin[i]
            ci = const[i]
            s2r = S2_rest[i]
            s1r = S1_rest[i]
            cmax = C_i[i]
            n1r = n1_rest[i]
            gn = g_neighbors[i]
            
            # g_i(t) = t^2 + li*t + ci
            # l2(t) = factor * [(t^2 + li*t + ci)^2 + s2r + (t^2 + li*t + ci)*gn + s1r]
            # n1(t) = n1r + (t^2 + li*t + ci)/(m+1)
            # ninf(t) = max(t^2 + li*t + ci, cmax)
            
            # Evaluate at several candidates
            candidates = [0.0, old_t]
            
            # Find where g_i(t) = cmax: t^2 + li*t + ci = cmax
            # t^2 + li*t + (ci - cmax) = 0
            disc = li*li - 4*(ci - cmax)
            if disc >= 0:
                sqrt_disc = np.sqrt(disc)
                t1 = (-li + sqrt_disc) / 2.0
                t2 = (-li - sqrt_disc) / 2.0
                if t1 >= 0:
                    candidates.append(t1)
                if t2 >= 0:
                    candidates.append(t2)
            
            # On branch 1: g_i(t) <= cmax, so ninf = cmax
            # A1(t) = factor * [(t^2 + li*t + ci)^2 + s2r + (t^2 + li*t + ci)*gn + s1r] - lam * (n1r + (t^2 + li*t + ci)/(m+1)) * cmax
            # = factor * [t^4 + 2*li*t^3 + (li^2 + 2*ci)*t^2 + 2*li*ci*t + ci^2 + s2r + gn*t^2 + li*gn*t + ci*gn + s1r]
            #   - lam * cmax * n1r - lam * cmax * (t^2 + li*t + ci)/(m+1)
            
            # A1(t) is quartic. Find critical points by taking derivative and solving.
            # A1'(t) = factor * [4t^3 + 6*li*t^2 + 2*(li^2 + 2*ci + gn)*t + 2*li*ci + li*gn]
            #          - lam * cmax * (2t + li)/(m+1)
            
            # This is cubic. Use np.roots.
            # A1'(t) = 4*factor*t^3 + 6*factor*li*t^2 + 2*factor*(li^2 + 2*ci + gn)*t + 2*factor*(li*ci + li*gn) - lam*cmax*(2t+li)/(m+1)
            # Multiply by (m+1):
            # (m+1)*4*factor*t^3 + (m+1)*6*factor*li*t^2 + (m+1)*2*factor*(li^2+2*ci+gn)*t + (m+1)*2*factor*(li*ci+li*gn) - lam*cmax*2t - lam*cmax*li = 0
            
            mc1 = (m+1) * 4 * factor
            mc2 = (m+1) * 6 * factor * li
            mc3 = (m+1) * 2 * factor * (li*li + 2*ci + gn) - lam * cmax * 2
            mc4 = (m+1) * 2 * factor * (li*ci + li*gn) - lam * cmax * li
            
            if abs(mc1) > 1e-15:
                roots = np.roots([mc1, mc2, mc3, mc4])
                for r in roots:
                    if np.isreal(r):
                        rv = float(np.real(r))
                        if rv >= 0:
                            # Check if on this branch: g_i(rv) <= cmax
                            gi_rv = rv*rv + li*rv + ci
                            if gi_rv <= cmax + 1e-10:
                                candidates.append(rv)
            
            # On branch 2: g_i(t) > cmax, so ninf = g_i(t)
            # A2(t) = factor * [(t^2 + li*t + ci)^2 + s2r + (t^2 + li*t + ci)*gn + s1r] - lam * (n1r + (t^2 + li*t + ci)/(m+1)) * (t^2 + li*t + ci)
            
            # This is degree 6. Too complex for exact roots. Use numerical search.
            # Sample on a grid in the region where g_i(t) > cmax.
            # g_i(t) > cmax means t < t2 or t > t1 (from above)
            
            # Just do a local search around old_t and the breakpoint
            if disc >= 0:
                sqrt_disc = np.sqrt(disc)
                t_bp = max(0, (-li - sqrt_disc) / 2.0)  # lower breakpoint
                # Search in [t_bp, t_bp + max(3*old_t, 1)]
                t_search_max = max(t_bp + 3*old_t + 1, t_bp + 1)
                grid = np.linspace(t_bp, t_search_max, 20)
                for tv in grid:
                    if tv >= 0:
                        candidates.append(tv)
            
            # Also add some exploration points
            candidates.extend([old_t * 0.5, old_t * 1.5, old_t * 2, old_t * 0.1, old_t * 10])
            candidates.append(1.0)
            candidates.append(0.1)
            
            # Evaluate all candidates
            best_t = old_t
            best_v = best_aux
            for tv in candidates:
                if tv < 0 or not np.isfinite(tv):
                    continue
                h[i] = tv
                v = _aux_obj(h, lam)
                if v > best_v:
                    best_v = v
                    best_t = tv
            
            # Local refinement around best_t using golden section
            if best_t > 0:
                span = max(best_t * 0.3, 0.01)
                a = max(0, best_t - span)
                b = best_t + span
                gr = (np.sqrt(5) - 1) / 2
                c = a + (1 - gr) * (b - a)
                d = a + gr * (b - a)
                h[c_idx] = c  # placeholder
                fc = _aux_obj(h, lam)
                h[i] = d
                fd = _aux_obj(h, lam)
                for _ in range(15):
                    if fc < fd:
                        b = d; d = c; fd = fc
                        c = a + (1 - gr) * (b - a)
                        h[i] = c
                        fc = _aux_obj(h, lam)
                    else:
                        a = c; c = d; fc = fd
                        d = a + gr * (b - a)
                        h[i] = d
                        fd = _aux_obj(h, lam)
                h[i] = (c + d) / 2
                v = _aux_obj(h, lam)
                if v > best_v:
                    best_v = v
                    best_t = (c + d) / 2
            
            h[i] = best_t
            
            if best_v > best_aux + tol:
                best_aux = best_v
                improved = True
                # Update precomputed coefficients
                g, lin, const, S2_rest, S1_rest, C_i, n1_rest, g_neighbors, factor, m, w = _precompute_coeffs(h)
        
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
        h, aux = _coord_step_fast(h, lam, max_sweeps=15, deadline=deadline)
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

def _slsqp_polish(h0, deadline, maxiter=200):
    """SLSQP polish with nonnegativity constraints."""
    h = np.maximum(np.asarray(h0, dtype=float), 0.0)
    h = h / max(h.sum(), 1e-12)
    
    def negv(p):
        hh = np.maximum(p, 0.0)
        s = hh.sum()
        if s < 1e-12:
            return 0.0
        hh = hh / s
        return -_compute_value(hh)
    
    def negv_grad(p):
        # Numerical gradient
        eps = 1e-7
        g = np.zeros(len(p))
        for i in range(len(p)):
            p_plus = p.copy()
            p_plus[i] += eps
            p_minus = p.copy()
            p_minus[i] = max(0, p_minus[i] - eps)
            g[i] = (negv(p_plus) - negv(p_minus)) / (p_plus[i] - p_minus[i] + 1e-15)
        return g
    
    constraints = [{'type': 'ineq', 'fun': lambda p: p, 'jac': lambda p: np.eye(len(p))}]
    
    try:
        res = minimize(negv, h.copy(), method='SLSQP', jac=negv_grad,
                       bounds=[(0, None)] * N, constraints=constraints,
                       options={'maxiter': maxiter, 'ftol': 1e-12})
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
        if time.time() > deadline - 8:
            break
        rng2 = np.random.default_rng(seed)
        for _ in range(3):
            if time.time() > deadline - 8:
                break
            h0 = np.abs(rng2.standard_normal(N)) + 0.02
            try_candidate(h0)

    # Phase 3: Final Dinkelbach polish on the best
    if time.time() < deadline - 5:
        try:
            h_f, v_f = _dinkelbach(best_h, max_iter=40, deadline=deadline)
            if v_f > best_val:
                best_val = v_f
                best_h = h_f.copy()
        except Exception:
            pass

    # Phase 4: Final NM polish
    if time.time() < deadline - 3:
        try:
            h_n, v_n = _nm_polish(best_h, deadline, maxiter=300)
            if v_n > best_val:
                best_val = v_n
                best_h = h_n.copy()
        except Exception:
            pass

    # Phase 5: SLSQP polish
    if time.time() < deadline - 2:
        try:
            h_s, v_s = _slsqp_polish(best_h, deadline, maxiter=100)
            if v_s > best_val:
                best_val = v_s
                best_h = h_s.copy()
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
