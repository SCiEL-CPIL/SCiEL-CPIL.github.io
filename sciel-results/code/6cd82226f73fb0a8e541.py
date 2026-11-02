import numpy as np
import time
from scipy.optimize import minimize

N = 50
DELTA = 1.0 / N

def _autoconv_sup(h):
    """Compute sup norm of autoconvolution of step function with heights h."""
    n = len(h)
    num_points = 2 * n - 1
    ff_vals = np.zeros(num_points)
    
    for k in range(num_points):
        total = 0.0
        i_min = max(0, k - (n - 1))
        i_max = min(n - 1, k)
        for i in range(i_min, i_max + 1):
            j_min = max(0, k - i - 1)
            j_max = min(n - 1, k - i + 1)
            if j_min > j_max:
                continue
            js = np.arange(j_min, j_max + 1)
            lo = np.maximum(i, k - js)
            hi = np.minimum(i + 1, k - js + 1)
            overlaps = np.maximum(0, hi - lo)
            total += np.sum(h[i] * h[js] * overlaps)
        ff_vals[k] = total * DELTA
    
    return np.max(ff_vals)

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
    n = len(h)
    num_points = 2 * n - 1
    ff_vals = np.zeros(num_points)
    
    for k in range(num_points):
        total = 0.0
        i_min = max(0, k - (n - 1))
        i_max = min(n - 1, k)
        for i in range(i_min, i_max + 1):
            j_min = max(0, k - i - 1)
            j_max = min(n - 1, k - i + 1)
            if j_min > j_max:
                continue
            js = np.arange(j_min, j_max + 1)
            lo = np.maximum(i, k - js)
            hi = np.minimum(i + 1, k - js + 1)
            overlaps = np.maximum(0, hi - lo)
            total += np.sum(h[i] * h[js] * overlaps)
        ff_vals[k] = total * DELTA
    
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

def _penalty_optimize(h0, start_time, deadline):
    """
    Penalty method: optimize ratio with Lagrangian relaxation for the
    non-smooth supremum term. We introduce an auxiliary variable t and
    penalize violations of ff_vals <= t.
    """
    h0 = np.maximum(h0, 0).copy()
    s0 = np.sum(h0)
    if s0 < 1e-15:
        return h0, _ratio(h0)
    h0 = h0 / s0
    
    best_h = h0.copy()
    best_score = _ratio(h0)
    
    # Penalty parameter schedule
    mu = 1.0
    for _ in range(5):
        if time.time() - start_time > deadline - 10:
            break
        
        def neg_penalty(x, t):
            x = np.maximum(x, 0)
            s = np.sum(x)
            if s < 1e-15:
                return 0.0
            x_norm = x / s
            
            # Compute autoconv values
            n = len(x_norm)
            num_points = 2 * n - 1
            ff_vals = np.zeros(num_points)
            for k in range(num_points):
                total = 0.0
                i_min = max(0, k - (n - 1))
                i_max = min(n - 1, k)
                for i in range(i_min, i_max + 1):
                    j_min = max(0, k - i - 1)
                    j_max = min(n - 1, k - i + 1)
                    if j_min > j_max:
                        continue
                    js = np.arange(j_min, j_max + 1)
                    lo = np.maximum(i, k - js)
                    hi = np.minimum(i + 1, k - js + 1)
                    overlaps = np.maximum(0, hi - lo)
                    total += np.sum(x_norm[i] * x_norm[js] * overlaps)
                ff_vals[k] = total * DELTA
            
            sum_h2 = np.sum(x_norm**2)
            sum_h = 1.0  # normalized
            
            # Penalty for ff_vals > t
            violation = np.maximum(ff_vals - t, 0)
            penalty = mu * np.sum(violation**2)
            
            # Objective: maximize ratio, penalize constraint violation
            # ratio = sum_h2 / (sum_h * t) = sum_h2 / t (since sum_h=1)
            if t < 1e-15:
                t = 1e-15
            ratio_val = sum_h2 / t
            return -(ratio_val - penalty)
        
        # Initial t estimate
        t_init = _autoconv_sup(h0) * 1.1
        
        # Optimize over (h, t)
        def neg_penalty_full(x):
            h_part = x[:N]
            t_val = x[N]
            return neg_penalty(h_part, t_val)
        
        x0 = np.concatenate([h0, [t_init]])
        bounds = [(0, None)] * N + [(1e-10, None)]
        
        try:
            result = minimize(neg_penalty_full, x0, method='L-BFGS-B',
                             bounds=bounds,
                             options={'maxiter': 80, 'ftol': 1e-10, 'gtol': 1e-7})
            
            h_candidate = np.maximum(result.x[:N], 0)
            s = np.sum(h_candidate)
            if s > 1e-15:
                h_candidate = h_candidate / s
                sc = _ratio(h_candidate)
                if sc > best_score:
                    best_score = sc
                    best_h = h_candidate.copy()
                    h0 = h_candidate.copy()
        except Exception:
            pass
        
        mu *= 5.0
    
    return best_h, best_score

def construct():
    start_time = time.time()
    DEADLINE = 145.0
    
    best_h = np.ones(N)
    best_score = _ratio(best_h)
    
    np.random.seed(42)
    
    initializations = []
    
    # 1. Mian-Chowla sets of various sizes
    for size in [8, 10, 12, 14, 16]:
        s = generate_mian_chowla(N, size)
        h_mc = np.zeros(N)
        for idx in s:
            h_mc[idx] = 1.0
        initializations.append(h_mc)
        
    # 2. Random sparse vectors
    for _ in range(20):
        h_sp = np.zeros(N)
        num_ones = np.random.randint(5, 15)
        indices = np.random.choice(N, num_ones, replace=False)
        h_sp[indices] = np.random.uniform(0.5, 1.5, num_ones)
        initializations.append(h_sp)
        
    # 3. Random dense vectors
    for _ in range(10):
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
    
    for h0 in initializations:
        if time.time() - start_time > DEADLINE - 20:
            break
        
        h0 = np.maximum(h0, 0)
        s0 = np.sum(h0)
        if s0 < 1e-15:
            continue
        h0 = h0 / s0
        
        # First: smooth surrogate optimization
        def neg_smooth_ratio(x):
            x = np.maximum(x, 0)
            s = np.sum(x)
            if s < 1e-15:
                return 0.0
            return -_smooth_ratio(x / s, beta=50.0)
        
        try:
            result = minimize(neg_smooth_ratio, h0.copy(), method='L-BFGS-B',
                             bounds=[(0, None)] * N,
                             options={'maxiter': 80, 'ftol': 1e-12, 'gtol': 1e-8})
            
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
        
        # Then: penalty method refinement
        if time.time() - start_time < DEADLINE - 15:
            try:
                h_pen, sc_pen = _penalty_optimize(best_h.copy(), start_time, DEADLINE)
                if sc_pen > best_score:
                    best_score = sc_pen
                    best_h = h_pen.copy()
            except Exception:
                pass
    
    # Final polish with Nelder-Mead on the exact ratio
    if time.time() - start_time < DEADLINE - 10:
        try:
            def neg_ratio(x):
                x = np.maximum(x, 0)
                s = np.sum(x)
                if s < 1e-15:
                    return 0.0
                return -_ratio(x / s)
            
            result = minimize(neg_ratio, best_h.copy(), method='Nelder-Mead',
                             options={'maxiter': 400, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True})
            
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
    if time.time() - start_time < DEADLINE - 5:
        for _ in range(20):
            improved = False
            for i in range(N):
                old_val = best_h[i]
                # Try a few perturbations
                for delta in [-0.1, -0.05, 0.05, 0.1, -0.2, 0.2]:
                    new_val = max(0, old_val + delta)
                    test_h = best_h.copy()
                    test_h[i] = new_val
                    s = np.sum(test_h)
                    if s < 1e-15:
                        continue
                    test_h = test_h / s
                    sc = _ratio(test_h)
                    if sc > best_score:
                        best_score = sc
                        best_h = test_h.copy()
                        improved = True
                        break
            if not improved:
                break
    
    return best_h.tolist()
