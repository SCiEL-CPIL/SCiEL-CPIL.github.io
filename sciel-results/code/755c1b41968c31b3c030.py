import numpy as np
import time
from scipy.optimize import minimize

N = 50
DELTA = 1.0 / N

# Precompute overlap matrix once
_n = N
_num_points = 2 * _n - 1
_i_idx = np.arange(_n)
_j_idx = np.arange(_n)
_k_idx = np.arange(_num_points)
_I, _J, _K = np.meshgrid(_i_idx, _j_idx, _k_idx, indexing='ij')
_lo = np.maximum(_I, _K - _J)
_hi = np.minimum(_I + 1, _K - _J + 1)
_OVERLAPS = np.maximum(0, _hi - _lo)  # shape (n, n, 2n-1)
# Also precompute for faster einsum: reshape to (n*n, 2n-1)
_OVERLAPS_2D = _OVERLAPS.reshape(_n * _n, _num_points)

def _autoconv_sup(h):
    """Compute sup norm of autoconvolution of step function with heights h (vectorized)."""
    hh = h[:, None] * h[None, :]  # (n, n)
    ff_vals = DELTA * np.dot(hh.ravel(), _OVERLAPS_2D)
    return np.max(ff_vals)

def _autoconv_all(h):
    """Compute all autoconv values."""
    hh = h[:, None] * h[None, :]  # (n, n)
    ff_vals = DELTA * np.dot(hh.ravel(), _OVERLAPS_2D)
    return ff_vals

def _ratio(h):
    """Compute the ratio ||h||_2^2 / (||h||_1 * ||h*h||_inf)."""
    h = np.maximum(h, 0)
    sum_h2 = np.sum(h**2)
    sum_h = np.sum(h)
    if sum_h < 1e-15:
        return 0.0
    sup_ff = _autoconv_sup(h)
    if sup_ff < 1e-15:
        return 0.0
    return sum_h2 / (sum_h * sup_ff)

def _smooth_autoconv_sup(h, beta=100.0):
    """Compute smooth approximation of sup norm using log-sum-exp."""
    ff_vals = _autoconv_all(h)
    max_val = np.max(ff_vals)
    if max_val == 0:
        return 0.0
    return max_val + np.log(np.sum(np.exp((ff_vals - max_val) * beta))) / beta

def _smooth_ratio(h, beta=100.0):
    """Compute smooth ratio."""
    h = np.maximum(h, 0)
    sum_h2 = np.sum(h**2)
    sum_h = np.sum(h)
    if sum_h < 1e-15:
        return 0.0
    sup_ff = _smooth_autoconv_sup(h, beta)
    if sup_ff < 1e-15:
        return 0.0
    return sum_h2 / (sum_h * sup_ff)

def generate_mian_chowla(n, max_size):
    """Generate a Mian-Chowla (Sidon) set."""
    sidon = [0]
    used_sums = {0}
    for candidate in range(1, n):
        valid = True
        for i in sidon:
            if i + candidate in used_sums:
                valid = False
                break
        if valid:
            sidon.append(candidate)
            used_sums.add(candidate + candidate)
            for i in sidon[:-1]:
                used_sums.add(i + candidate)
        if len(sidon) >= max_size:
            break
    return sidon

def _epigraph_optimize(h0, start_time, deadline):
    """
    Epigraph formulation: maximize sum_h2/t subject to ff_vals <= t, h >= 0, sum_h = 1.
    Using smooth constraints via log-sum-exp for the max.
    """
    h0 = np.maximum(h0, 0).copy()
    s0 = np.sum(h0)
    if s0 < 1e-15:
        return h0, _ratio(h0)
    h0 = h0 / s0
    
    best_h = h0.copy()
    best_score = _ratio(h0)
    
    beta = 50.0
    
    def neg_objective(x):
        h_part = np.maximum(x[:N], 0)
        t_val = max(x[N], 1e-10)
        s = np.sum(h_part)
        if s < 1e-15:
            return 0.0
        h_norm = h_part / s
        
        ff_vals = _autoconv_all(h_norm)
        sum_h2 = np.sum(h_norm**2)
        
        # Smooth constraint violation: logsumexp((ff_vals - t_val)/beta) * beta
        shifted = (ff_vals - t_val) / beta
        max_shifted = np.max(shifted)
        if max_shifted > 50:  # overflow guard
            return 1e10
        lse = max_shifted + np.log(np.sum(np.exp(shifted - max_shifted)))
        penalty = lse * beta
        
        ratio_val = sum_h2 / t_val
        return -(ratio_val - penalty)
    
    t_init = _autoconv_sup(h0) * 1.05
    x0 = np.concatenate([h0, [t_init]])
    bounds = [(0, None)] * N + [(1e-8, None)]
    
    try:
        result = minimize(neg_objective, x0, method='L-BFGS-B',
                         bounds=bounds,
                         options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-9})
        
        h_candidate = np.maximum(result.x[:N], 0)
        s = np.sum(h_candidate)
        if s > 1e-15:
            h_candidate = h_candidate / s
            sc = _ratio(h_candidate)
            if sc > best_score:
                best_score = sc
                best_h = h_candidate.copy()
    except Exception:
        pass
    
    return best_h, best_score

def _coordinate_descent_polish(h, start_time, deadline, max_rounds=80):
    """Coordinate descent with finer golden-section-like search per coordinate."""
    best_h = h.copy()
    best_score = _ratio(best_h)
    
    for round_num in range(max_rounds):
        if time.time() - start_time > deadline - 3:
            break
        
        improved = False
        for i in range(N):
            if time.time() - start_time > deadline - 3:
                break
            
            old_val = best_h[i]
            
            # Generate candidate values: geometric around old_val + some absolutes
            candidates = set()
            if old_val > 1e-10:
                for frac in [0.2, 0.4, 0.6, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4, 1.7, 2.0, 3.0]:
                    candidates.add(old_val * frac)
            # Add absolute scale candidates
            for v in [0.0, 0.01, 0.03, 0.07, 0.15, 0.3, 0.6, 1.0, 1.5, 2.5, 4.0]:
                candidates.add(v)
            
            best_local = old_val
            best_local_score = best_score
            
            for cv in candidates:
                cv = max(0.0, cv)
                test_h = best_h.copy()
                test_h[i] = cv
                s = np.sum(test_h)
                if s < 1e-15:
                    continue
                test_h = test_h / s
                sc = _ratio(test_h)
                if sc > best_local_score + 1e-12:
                    best_local_score = sc
                    best_local = cv
            
            if best_local != old_val:
                best_h[i] = best_local
                s = np.sum(best_h)
                if s > 1e-15:
                    best_h = best_h / s
                    best_score = best_local_score
                improved = True
        
        if not improved:
            break
    
    return best_h, best_score

def construct():
    start_time = time.time()
    DEADLINE = 145.0
    
    best_h = np.ones(N)
    best_score = _ratio(best_h)
    
    np.random.seed(42)
    
    initializations = []
    
    # 1. Mian-Chowla sets of various sizes
    for size in [6, 8, 10, 12, 14, 16, 18]:
        s = generate_mian_chowla(N, size)
        h_mc = np.zeros(N)
        for idx in s:
            h_mc[idx] = 1.0
        initializations.append(h_mc)
        
    # 2. Random sparse vectors
    for _ in range(15):
        h_sp = np.zeros(N)
        num_ones = np.random.randint(4, 14)
        indices = np.random.choice(N, num_ones, replace=False)
        h_sp[indices] = np.random.uniform(0.3, 2.0, num_ones)
        initializations.append(h_sp)
        
    # 3. Random dense vectors
    for _ in range(8):
        h_rd = np.random.uniform(0, 1, N)
        initializations.append(h_rd)
        
    # 4. Alternating
    h_alt = np.zeros(N)
    h_alt[::2] = 1.0
    initializations.append(h_alt)
    
    # 5. Every 3
    h_sp3 = np.zeros(N)
    h_sp3[::3] = 1.0
    initializations.append(h_sp3)
    
    # 6. Gaussian
    t = np.linspace(-1, 1, N)
    initializations.append(np.exp(-t**2/0.3))
    
    # 7. Triangular
    h_tri = np.zeros(N)
    center = N // 2
    for i in range(N):
        h_tri[i] = max(0, 1 - abs(i - center) / (N/2))
    initializations.append(h_tri)
    
    # 8. Two peaks
    h_2p = np.zeros(N)
    h_2p[10] = 1.0
    h_2p[40] = 1.0
    initializations.append(h_2p)
    
    # 9. Cosine bump
    t = np.linspace(-1, 1, N)
    initializations.append(0.5 + 0.5*np.cos(np.pi*t))
    
    # 10. Power law
    t = np.linspace(-1, 1, N)
    initializations.append(np.abs(t)**(-0.5 + 1e-9))
    
    # 11. Uniform with one peak
    h_up = np.full(N, 0.1)
    h_up[N//2] = 1.0
    initializations.append(h_up)
    
    # 12. Sparse random with varying amplitudes
    for _ in range(5):
        h_vr = np.zeros(N)
        num_ones = np.random.randint(6, 12)
        indices = np.random.choice(N, num_ones, replace=False)
        h_vr[indices] = np.random.uniform(0.1, 3.0, num_ones)
        initializations.append(h_vr)
    
    # Time budget allocation: ~70% for main optimization, ~30% for polish
    main_deadline = start_time + 0.7 * (DEADLINE - start_time)
    
    for h0 in initializations:
        if time.time() - start_time > main_deadline:
            break
        
        h0 = np.maximum(h0, 0)
        s0 = np.sum(h0)
        if s0 < 1e-15:
            continue
        h0 = h0 / s0
        
        # Phase 1: Smooth surrogate optimization with 2 beta values (faster)
        for beta in [40.0, 70.0]:
            if time.time() - start_time > main_deadline:
                break
            
            def neg_smooth_ratio(x, b=beta):
                x = np.maximum(x, 0)
                s = np.sum(x)
                if s < 1e-15:
                    return 0.0
                return -_smooth_ratio(x / s, beta=b)
            
            try:
                result = minimize(neg_smooth_ratio, h0.copy(), method='L-BFGS-B',
                                 bounds=[(0, None)] * N,
                                 options={'maxiter': 80, 'ftol': 1e-12, 'gtol': 1e-9})
                
                h_candidate = np.maximum(result.x, 0)
                s = np.sum(h_candidate)
                if s > 1e-15:
                    h_candidate = h_candidate / s
                    sc = _ratio(h_candidate)
                    if sc > best_score:
                        best_score = sc
                        best_h = h_candidate.copy()
            except Exception:
                continue
        
        # Phase 2: Epigraph optimization (only if time remains)
        if time.time() - start_time < main_deadline - 5:
            try:
                h_epi, sc_epi = _epigraph_optimize(best_h.copy(), start_time, main_deadline)
                if sc_epi > best_score:
                    best_score = sc_epi
                    best_h = h_epi.copy()
            except Exception:
                pass
    
    # Polish phase: Nelder-Mead on exact ratio
    if time.time() - start_time < DEADLINE - 15:
        try:
            def neg_ratio(x):
                x = np.maximum(x, 0)
                s = np.sum(x)
                if s < 1e-15:
                    return 0.0
                return -_ratio(x / s)
            
            result = minimize(neg_ratio, best_h.copy(), method='Nelder-Mead',
                             options={'maxiter': 400, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
            
            h_candidate = np.maximum(result.x, 0)
            s = np.sum(h_candidate)
            if s > 1e-15:
                h_candidate = h_candidate / s
                sc = _ratio(h_candidate)
                if sc > best_score:
                    best_score = sc
                    best_h = h_candidate.copy()
        except Exception:
            pass
    
    # Coordinate descent polish
    if time.time() - start_time < DEADLINE - 8:
        try:
            h_cd, sc_cd = _coordinate_descent_polish(best_h.copy(), start_time, DEADLINE)
            if sc_cd > best_score:
                best_score = sc_cd
                best_h = h_cd.copy()
        except Exception:
            pass
    
    # Final Nelder-Mead polish after coordinate descent
    if time.time() - start_time < DEADLINE - 3:
        try:
            def neg_ratio2(x):
                x = np.maximum(x, 0)
                s = np.sum(x)
                if s < 1e-15:
                    return 0.0
                return -_ratio(x / s)
            
            result = minimize(neg_ratio2, best_h.copy(), method='Nelder-Mead',
                             options={'maxiter': 300, 'xatol': 1e-11, 'fatol': 1e-13, 'adaptive': True})
            
            h_candidate = np.maximum(result.x, 0)
            s = np.sum(h_candidate)
            if s > 1e-15:
                h_candidate = h_candidate / s
                sc = _ratio(h_candidate)
                if sc > best_score:
                    best_score = sc
                    best_h = h_candidate.copy()
        except Exception:
            pass
    
    return best_h.tolist()
