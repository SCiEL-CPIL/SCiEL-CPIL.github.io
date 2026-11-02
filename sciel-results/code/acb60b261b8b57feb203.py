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

def _autoconv_all(h):
    """Compute all autoconvolution values at breakpoints."""
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

def _ratio_with_peak(h, peak_idx):
    """Compute ratio using a specific peak index (for iterative reweighting)."""
    h = np.maximum(h, 0)
    sum_h2 = np.sum(h**2)
    sum_h = np.sum(h)
    if sum_h < 1e-15:
        return 0.0
    ff_vals = _autoconv_all(h)
    if peak_idx < 0 or peak_idx >= len(ff_vals):
        return 0.0
    sup_ff = ff_vals[peak_idx]
    if sup_ff < 1e-15:
        return 0.0
    return sum_h2 / (sum_h * sup_ff)

def _smooth_autoconv_sup(h, beta=50.0):
    """Compute smooth approximation of sup norm using log-sum-exp."""
    ff_vals = _autoconv_all(h)
    max_val = np.max(ff_vals)
    if max_val == 0:
        return 0.0
    return max_val + np.log(np.sum(np.exp((ff_vals - max_val) * beta))) / beta

def _smooth_ratio(h, beta=50.0):
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

def _iterative_reweighting(h0, max_iter=200, tol=1e-8, time_limit=None, start_time=None):
    """Iterative reweighting on fixed support to optimize heights.
    
    For a fixed support S, we optimize the heights h[i] for i in S.
    At each iteration:
    1. Find the peak index k* of the autoconvolution
    2. Use multiplicative updates to adjust heights
    3. Check convergence
    """
    h = np.maximum(h0, 0).copy()
    support = np.where(h > 1e-10)[0]
    
    if len(support) == 0:
        return h, _ratio(h)
    
    best_h = h.copy()
    best_score = _ratio(h)
    
    for it in range(max_iter):
        if time_limit is not None and start_time is not None:
            if time.time() - start_time > time_limit:
                break
        
        # Find the current peak
        ff_vals = _autoconv_all(h)
        peak_idx = np.argmax(ff_vals)
        
        # Compute gradient-like quantities for multiplicative update
        # We want to maximize ratio = sum_h2 / (sum_h * ff[peak_idx])
        # Taking log: log(sum_h2) - log(sum_h) - log(ff[peak_idx])
        # 
        # For multiplicative updates, we adjust h[i] *= exp(eta * dlog/dh[i])
        # where dlog/dh[i] = 2*h[i]/sum_h2 - 1/sum_h - (1/ff[peak_idx]) * d(ff[peak_idx])/dh[i]
        
        sum_h2 = np.sum(h**2)
        sum_h = np.sum(h)
        ff_peak = ff_vals[peak_idx]
        
        if sum_h < 1e-15 or ff_peak < 1e-15:
            break
        
        # Compute d(ff[peak_idx])/dh[i] for i in support
        # ff[peak_idx] = DELTA * sum_{i,j} h[i]*h[j]*overlap(i,j,peak_idx)
        # d(ff)/dh[i] = DELTA * sum_j h[j]*overlap(i,j,peak_idx) + DELTA * sum_j h[j]*overlap(j,i,peak_idx)
        # By symmetry: = 2 * DELTA * sum_j h[j]*overlap(i,j,peak_idx)
        
        k = peak_idx
        n = len(h)
        dff_dh = np.zeros(n)
        for i in range(n):
            if h[i] < 1e-15:
                continue
            j_min = max(0, k - i - 1)
            j_max = min(n - 1, k - i + 1)
            if j_min > j_max:
                continue
            js = np.arange(j_min, j_max + 1)
            lo = np.maximum(i, k - js)
            hi = np.minimum(i + 1, k - js + 1)
            overlaps = np.maximum(0, hi - lo)
            dff_dh[i] = 2.0 * DELTA * np.sum(h[js] * overlaps)
        
        # Compute gradient of log ratio
        grad_log = 2.0 * h / sum_h2 - 1.0 / sum_h - dff_dh / ff_peak
        
        # Multiplicative update with adaptive step size
        # Start with small eta, increase if improving
        eta = 0.1
        for _ in range(10):
            h_new = h * np.exp(eta * grad_log)
            h_new = np.maximum(h_new, 0)
            
            # Normalize to keep sum_h constant (ratio is scale-invariant)
            s_new = np.sum(h_new)
            if s_new < 1e-15:
                break
            h_new = h_new / s_new
            
            score_new = _ratio(h_new)
            if score_new > best_score + 1e-12:
                h = h_new
                best_h = h.copy()
                best_score = score_new
                eta = min(eta * 1.5, 1.0)
                break
            else:
                eta /= 2.0
        
        # Check convergence
        if it > 0 and abs(best_score - _ratio(h)) < tol:
            break
    
    return best_h, best_score

def _coordinate_descent(h0, max_iter=500, tol=1e-10, time_limit=None, start_time=None):
    """Coordinate descent polish: optimize one coordinate at a time."""
    h = np.maximum(h0, 0).copy()
    best_score = _ratio(h)
    n = len(h)
    
    for it in range(max_iter):
        if time_limit is not None and start_time is not None:
            if time.time() - start_time > time_limit:
                break
        
        improved = False
        for i in range(n):
            if h[i] < 1e-15:
                continue
            
            # Try small perturbations
            for delta in [0.01, 0.001, 0.0001, -0.01, -0.001, -0.0001]:
                h_new = h.copy()
                h_new[i] = max(0, h[i] + delta)
                
                # Normalize
                s = np.sum(h_new)
                if s < 1e-15:
                    continue
                h_new = h_new / s
                
                score = _ratio(h_new)
                if score > best_score + 1e-12:
                    h = h_new
                    best_score = score
                    improved = True
                    break
        
        if not improved:
            break
    
    return h, best_score

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # Safety margin before 150s kill
    
    best_h = np.ones(N)
    best_score = _ratio(best_h)
    
    np.random.seed(42)
    
    initializations = []
    
    # 1. Mian-Chowla sets of various sizes
    for size in [6, 8, 10, 12, 14, 16]:
        s = generate_mian_chowla(N, size)
        h_mc = np.zeros(N)
        for idx in s:
            h_mc[idx] = 1.0
        initializations.append(h_mc)
        
    # 2. Random sparse vectors with varying densities
    for _ in range(30):
        h_sp = np.zeros(N)
        num_ones = np.random.randint(4, 18)
        indices = np.random.choice(N, num_ones, replace=False)
        h_sp[indices] = np.random.uniform(0.3, 2.0, num_ones)
        initializations.append(h_sp)
        
    # 3. Random dense vectors
    for _ in range(15):
        h_rd = np.random.uniform(0, 1, N)
        initializations.append(h_rd)
        
    # 4. Structured patterns
    h_alt = np.zeros(N)
    h_alt[::2] = 1.0
    initializations.append(h_alt)
    
    h_sp3 = np.zeros(N)
    h_sp3[::3] = 1.0
    initializations.append(h_sp3)
    
    h_sp4 = np.zeros(N)
    h_sp4[::4] = 1.0
    initializations.append(h_sp4)
    
    # 5. Gaussian and other smooth shapes
    t = np.linspace(-1, 1, N)
    initializations.append(np.exp(-t**2/0.3))
    initializations.append(np.exp(-t**2/0.5))
    initializations.append(np.maximum(1 - t**2, 0.01))
    initializations.append(np.maximum(t**2 + 0.1, 0.01))
    
    # 6. Bimodal
    initializations.append(np.maximum(1.0 + 2.0*np.exp(-((t-0.6)**2)/0.1) + 2.0*np.exp(-((t+0.6)**2)/0.1), 0.05))
    
    # 7. Centered spike
    h_center = np.zeros(N)
    h_center[N//2] = 1.0
    initializations.append(h_center)
    
    # 8. Edge spikes
    h_edges = np.zeros(N)
    h_edges[0] = 1.0
    h_edges[N-1] = 1.0
    initializations.append(h_edges)
    
    # Process each initialization
    for h0 in initializations:
        if time.time() - start_time > DEADLINE - 10:
            break
        
        h0 = np.maximum(h0, 0)
        s0 = np.sum(h0)
        if s0 < 1e-15:
            continue
        h0 = h0 / s0
        
        # Strategy 1: Iterative reweighting (fast, avoids non-smooth issues)
        try:
            h_rew, score_rew = _iterative_reweighting(
                h0, max_iter=150, tol=1e-8,
                time_limit=DEADLINE - 15, start_time=start_time
            )
            if score_rew > best_score:
                best_score = score_rew
                best_h = h_rew.copy()
        except Exception:
            pass
        
        # Strategy 2: Smooth surrogate optimization with L-BFGS-B
        if time.time() - start_time < DEADLINE - 20:
            try:
                def neg_smooth_ratio(x):
                    x = np.maximum(x, 0)
                    s = np.sum(x)
                    if s < 1e-15:
                        return 0.0
                    return -_smooth_ratio(x / s, beta=30.0)
                
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
                pass
    
    # Final polish: coordinate descent on the best solution
    if time.time() - start_time < DEADLINE - 5:
        try:
            h_polished, score_polished = _coordinate_descent(
                best_h, max_iter=300, tol=1e-10,
                time_limit=DEADLINE - 2, start_time=start_time
            )
            if score_polished > best_score:
                best_score = score_polished
                best_h = h_polished.copy()
        except Exception:
            pass
    
    # One more iterative reweighting pass on the polished result
    if time.time() - start_time < DEADLINE - 3:
        try:
            h_final, score_final = _iterative_reweighting(
                best_h, max_iter=100, tol=1e-9,
                time_limit=DEADLINE - 1, start_time=start_time
            )
            if score_final > best_score:
                best_score = score_final
                best_h = h_final.copy()
        except Exception:
            pass
    
    return best_h.tolist()
