"""Second autocorrelation inequality - Dinkelbach with analytic coordinates + SA."""
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

def _analytic_coord_max(h, lam, max_sweeps=15, tol=1e-10, deadline=None):
    """Coordinate maximization using analytic polynomial forms.
    
    For varying h[i]=t with rest fixed:
    - g_i(t) = F*t^2 + G*t + H (quadratic)
    - n1(t) = (S0 + D*t) / m (linear)  
    - l2(t) = l2_0 + A1*t + A2*t^2 (quadratic)
    - ninf(t) = max(g_i(t), C) where C = max_{j!=i} g_j
    
    A(t) = l2(t) - lam * n1(t) * ninf(t)
    Piecewise: quadratic where g_i <= C, quartic where g_i > C.
    """
    h = np.maximum(np.asarray(h, dtype=float), 0.0)
    best_aux = _aux_obj(h, lam)
    
    for sweep in range(max_sweeps):
        if deadline is not None and time.time() > deadline - 2:
            break
        improved = False
        
        g = np.convolve(h, h)
        m = len(g)
        
        for i in range(N):
            if deadline is not None and time.time() > deadline - 2:
                break
            
            old_t = h[i]
            
            # g_i(t) coefficients: g_i = sum_j h[j]*h[i+j]
            # Pairs where h[i] appears: j=0 (h[0]*h[i]) and j=i (h[i]*h[2i])
            if i == 0:
                F_i = 1.0
                G_i = 0.0
                H_i = np.sum(h[1:]**2)
            else:
                F_i = 0.0
                G_i = h[0] + (h[2*i] if 2*i < N else 0.0)
                H_i = 0.0
                for j in range(1, N):
                    if j == i:
                        continue
                    k = i + j
                    if 0 <= k < N:
                        H_i += h[j] * h[k]
            
            # C = max of g_j for j != i
            if m > 1:
                C_i = np.max(np.delete(g, i))
            else:
                C_i = 0.0
            
            # l2(t) = l2_0 + A1*t + A2*t^2
            # Compute by evaluating at t=0, t=1, t=2 with h[i] replaced
            h0 = h.copy(); h0[i] = 0.0
            h1 = h.copy(); h1[i] = 1.0
            h2 = h.copy(); h2[i] = 2.0
            
            g0 = np.convolve(h0, h0)
            g1 = np.convolve(h1, h1)
            g2 = np.convolve(h2, h2)
            
            y0 = np.concatenate(([0.0], g0, [0.0]))
            y1 = np.concatenate(([0.0], g1, [0.0]))
            y2 = np.concatenate(([0.0], g2, [0.0]))
            w = 1.0 / (m + 1)
            
            l2_0 = (w/3.0) * np.sum(y0[:m+1]**2 + y0[:m+1]*y0[1:m+2] + y0[1:m+2]**2)
            l2_1 = (w/3.0) * np.sum(y1[:m+1]**2 + y1[:m+1]*y1[1:m+2] + y1[1:m+2]**2)
            l2_2 = (w/3.0) * np.sum(y2[:m+1]**2 + y2[:m+1]*y2[1:m+2] + y2[1:m+2]**2)
            
            A1 = 2*(l2_1 - l2_0) - (l2_2 - l2_0)
            A2 = (l2_2 - l2_0) - 2*(l2_1 - l2_0)
            
            # n1(t) = (sum_g_0 + D*t) / m
            sum_g_0 = np.sum(g0)
            D = np.sum(g1) - sum_g_0  # linear coefficient
            
            # Find breakpoints where g_i(t) = C_i
            # F_i*t^2 + G_i*t + (H_i - C_i) = 0
            breakpoints = []
            if F_i > 1e-15:
                disc = G_i**2 - 4*F_i*(H_i - C_i)
                if disc >= 0:
                    sq = np.sqrt(disc)
                    tb1 = (-G_i - sq) / (2*F_i)
                    tb2 = (-G_i + sq) / (2*F_i)
                    for tb in [tb1, tb2]:
                        if tb > 1e-12:
                            breakpoints.append(tb)
            elif abs(G_i) > 1e-15:
                tb = -(H_i - C_i) / G_i
                if tb > 1e-12:
                    breakpoints.append(tb)
            
            # Candidate t values
            candidates = [0.0, old_t]
            candidates.extend(breakpoints)
            
            # Vertex of quadratic branch (where g_i <= C_i)
            # A(t) = l2_0 + A1*t + A2*t^2 - lam * ((sum_g_0 + D*t)/m) * C_i
            #     = (A2 - lam*C_i*D/m)*t^2 + (A1 - lam*C_i*sum_g_0/m)*t + const
            a_q = A2 - lam*C_i*D/m
            b_q = A1 - lam*C_i*sum_g_0/m
            if abs(a_q) > 1e-15:
                tv = -b_q / (2*a_q)
                if tv > 0:
                    candidates.append(tv)
            
            # For quartic branch, sample a grid
            t_max = max(3*old_t + 1.0, 1.5)
            grid = np.linspace(0, t_max, 15)
            for tg in grid:
                gi_tg = F_i*tg**2 + G_i*tg + H_i
                if gi_tg > C_i + 1e-12:
                    candidates.append(tg)
            
            # Evaluate all candidates
            best_t = old_t
            best_v = _aux_obj(h, lam)
            
            for tv in candidates:
                if tv < -1e-12 or tv > 20*t_max:
                    continue
                tv = max(0.0, tv)
                h_trial = h.copy()
                h_trial[i] = tv
                v = _aux_obj(h_trial, lam)
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
        h, aux = _analytic_coord_max(h, lam, max_sweeps=12, deadline=deadline)
        new_val = _compute_value(h)
        if abs(new_val - lam) < tol * max(1.0, abs(lam)):
            break
        lam = new_val
    return h, _compute_value(h)

def _sa_refine(h0, deadline, max_iters=3000):
    """Simulated annealing to escape local optima."""
    h = np.maximum(np.asarray(h0, dtype=float), 0.0)
    s = h.sum()
    if s < 1e-12:
        h = np.ones(N) / N
    else:
        h = h / s
    
    best_h = h.copy()
    best_val = _compute_value(h)
    cur_val = best_val
    
    T_start = 0.1
    T_end = 1e-6
    n_iter = max_iters
    
    for it in range(n_iter):
        if deadline is not None and time.time() > deadline - 3:
            break
        T = T_start * (T_end / T_start) ** (it / max(n_iter - 1, 1))
        
        # Random perturbation: modify 1-3 random coordinates
        n_mod = np.random.randint(1, 4)
        idx = np.random.choice(N, n_mod, replace=False)
        h_new = h.copy()
        for i in idx:
            h_new[i] = max(0.0, h[i] * np.exp(np.random.normal(0, 0.3)))
        
        s_new = h_new.sum()
        if s_new < 1e-12:
            continue
        h_new = h_new / s_new
        
        v_new = _compute_value(h_new)
        
        if v_new > cur_val or np.random.random() < np.exp((v_new - cur_val) / max(T, 1e-12)):
            h = h_new
            cur_val = v_new
            if v_new > best_val:
                best_val = v_new
                best_h = h.copy()
    
    return best_h, best_val

def _nm_polish(h0, deadline, maxiter=300):
    h = np.maximum(np.asarray(h0, dtype=float), 0.0)
    s = h.sum()
    if s < 1e-12:
        h = np.ones(N) / N
    else:
        h = h / s
    
    def negv(p):
        hh = np.maximum(p, 1e-12)
        ss = hh.sum()
        if ss < 1e-12:
            return 0.0
        hh = hh / ss
        return -_compute_value(hh)
    
    try:
        res = minimize(negv, h.copy(), method='Nelder-Mead',
                       options={'maxiter': maxiter, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
        hh = np.maximum(res.x, 0.0)
        ss = hh.sum()
        if ss > 1e-12:
            hh = hh / ss
        v = _compute_value(hh)
        if v > _compute_value(h):
            return hh, v
    except Exception:
        pass
    return h, _compute_value(h)

def _gen_starts(rng):
    starts = []
    idx = np.arange(N)
    x = idx / (N - 1) - 0.5
    
    starts.append(np.ones(N))
    for s in [0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.8]:
        starts.append(np.exp(-x**2 / (2 * s**2)))
    for p in [0.5, 1, 1.5, 2, 3, 4]:
        for s in [0.1, 0.2, 0.4, 0.7]:
            starts.append(np.exp(-np.abs(x) ** p / s))
    for w in [0.1, 0.2, 0.3, 0.5]:
        starts.append(np.maximum(0, 1 - np.abs(x) / w))
    for w in [0.1, 0.2, 0.3]:
        starts.append(np.where(np.abs(x) < w, 1.0, 0.0) + 1e-6)
    for c in [0.15, 0.25, 0.35]:
        for s in [0.05, 0.1, 0.15]:
            starts.append(np.exp(-(x - c)**2 / (2*s**2)) + np.exp(-(x + c)**2 / (2*s**2)))
    for s in [0.08, 0.12, 0.2]:
        starts.append(np.exp(-(x - 0.4)**2/(2*s**2)) + np.exp(-x**2/(2*s**2)) + np.exp(-(x + 0.4)**2/(2*s**2)))
    for a in [2, 5, 10, 20]:
        starts.append(1.0 / (1 + (a * x)**2))
    for a in [0.5, 1, 2, 3]:
        starts.append(1.0 / (1 + np.abs(x) / 0.2) ** a)
    for _ in range(20):
        nfreq = int(rng.integers(2, 8))
        coeffs = rng.standard_normal(nfreq)
        phases = rng.uniform(0, 2*np.pi, nfreq)
        h_f = np.ones(N)
        for k in range(1, nfreq + 1):
            h_f += coeffs[k-1] * np.cos(2*np.pi*k*x + phases[k-1])
        starts.append(np.abs(h_f) + 0.02)
    for _ in range(20):
        starts.append(np.abs(rng.standard_normal(N)) + 0.02)
    return starts

def construct():
    t0 = time.time()
    deadline = t0 + 145.0
    
    rng = np.random.default_rng(50)
    starts = _gen_starts(rng)
    
    best_h = np.ones(N) / N
    best_val = _compute_value(best_h)
    
    def try_candidate(h0, do_sa=False):
        nonlocal best_h, best_val
        if time.time() > deadline - 8:
            return
        h0 = np.maximum(np.asarray(h0, dtype=float), 0.0)
        s = h0.sum()
        if s < 1e-12:
            return
        h0 = h0 / s
        
        try:
            h_d, v_d = _dinkelbach(h0, max_iter=25, deadline=deadline)
            if v_d > best_val:
                best_val = v_d
                best_h = h_d.copy()
        except Exception:
            pass
        
        if do_sa and time.time() < deadline - 15:
            try:
                h_s, v_s = _sa_refine(best_h, deadline, max_iters=2000)
                if v_s > best_val:
                    best_val = v_s
                    best_h = h_s.copy()
            except Exception:
                pass
        
        if time.time() < deadline - 10:
            try:
                h_n, v_n = _nm_polish(best_h, deadline, maxiter=250)
                if v_n > best_val:
                    best_val = v_n
                    best_h = h_n.copy()
            except Exception:
                pass
    
    # Phase 1: Dinkelbach from structured starts
    for h0 in starts:
        if time.time() > deadline - 20:
            break
        try_candidate(h0)
    
    # Phase 2: Random restarts with SA
    for seed in [1, 2, 3, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
        if time.time() > deadline - 15:
            break
        rng2 = np.random.default_rng(seed)
        for _ in range(2):
            if time.time() > deadline - 15:
                break
            h0 = np.abs(rng2.standard_normal(N)) + 0.02
            try_candidate(h0, do_sa=True)
    
    # Phase 3: SA on current best
    if time.time() < deadline - 12:
        try:
            h_s, v_s = _sa_refine(best_h, deadline, max_iters=3000)
            if v_s > best_val:
                best_val = v_s
                best_h = h_s.copy()
        except Exception:
            pass
    
    # Phase 4: Final Dinkelbach + NM
    if time.time() < deadline - 8:
        try:
            h_f, v_f = _dinkelbach(best_h, max_iter=35, deadline=deadline)
            if v_f > best_val:
                best_val = v_f
                best_h = h_f.copy()
        except Exception:
            pass
    
    if time.time() < deadline - 5:
        try:
            h_n, v_n = _nm_polish(best_h, deadline, maxiter=400)
            if v_n > best_val:
                best_val = v_n
                best_h = h_n.copy()
        except Exception:
            pass
    
    best_h = np.maximum(np.asarray(best_h, dtype=float), 0.0)
    s = best_h.sum()
    if s < 1e-12:
        best_h = np.ones(N) / N
    else:
        best_h = best_h / s
    return list(best_h)
