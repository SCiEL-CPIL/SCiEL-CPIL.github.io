import numpy as np
import time
from scipy.optimize import minimize

N = 50
DELTA = 1.0 / N

def _autoconv_all(h):
    """Compute all autoconvolution values at breakpoints (vectorized)."""
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

def _autoconv_sup(h):
    """Compute sup norm of autoconvolution."""
    return np.max(_autoconv_all(h))

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

def _smooth_autoconv_sup(h, beta=50.0):
    """Compute smooth approximation of sup norm using log-sum-exp."""
    ff_vals = _autoconv_all(h)
    max_val = np.max(ff_vals)
    if max_val == 0:
        return 0.0
    shifted = (ff_vals - max_val) * beta
    shifted = np.clip(shifted, -700, 700)
    return max_val + np.log(np.sum(np.exp(shifted))) / beta

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
    """Iterative reweighting on fixed support to optimize heights."""
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
        
        ff_vals = _autoconv_all(h)
        peak_idx = np.argmax(ff_vals)
        
        sum_h2 = np.sum(h**2)
        sum_h = np.sum(h)
        ff_peak = ff_vals[peak_idx]
        
        if sum_h < 1e-15 or ff_peak < 1e-15:
            break
        
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
        
        grad_log = 2.0 * h / sum_h2 - 1.0 / sum_h - dff_dh / ff_peak
        
        eta = 0.1
        for _ in range(10):
            h_new = h * np.exp(eta * grad_log)
            h_new = np.maximum(h_new, 0)
            
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
        
        if it > 0 and abs(best_score - _ratio(h)) < tol:
            break
    
    return best_h, best_score

def _epigraph_optimize(h0, time_limit=None, start_time=None):
    """Epigraph formulation: minimize t subject to ratio constraints.
    
    We maximize sum_h2 / (sum_h * t) subject to ff_vals[k] <= t for all k.
    Equivalently, we minimize t / (sum_h2 / sum_h) = t * sum_h / sum_h2.
    
    Using log-sum-exp for smooth sup:
    Minimize: log(sum_h) - log(sum_h2) + log(sum exp(beta*ff_k)/sum exp(beta*ff_k) + max_ff)
    """
    h = np.maximum(h0, 0).copy()
    s = np.sum(h)
    if s < 1e-15:
        return h, _ratio(h)
    h = h / s
    
    def objective(x):
        x = np.maximum(x, 0)
        sx = np.sum(x)
        if sx < 1e-15:
            return 1e6
        x = x / sx
        ff = _autoconv_all(x)
        max_ff = np.max(ff)
        if max_ff < 1e-15:
            return 1e6
        # log-sum-exp for smooth max
        shifted = (ff - max_ff) * 30.0
        shifted = np.clip(shifted, -700, 700)
        smooth_max = max_ff + np.log(np.sum(np.exp(shifted))) / 30.0
        sum_h2 = np.sum(x**2)
        if sum_h2 < 1e-15:
            return 1e6
        return -(sum_h2 / (sx * smooth_max))
    
    try:
        result = minimize(objective, h.copy(), method='L-BFGS-B',
                         bounds=[(0, None)] * N,
                         options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10,
                                  'maxls': 50})
        h_candidate = np.maximum(result.x, 0)
        s_cand = np.sum(h_candidate)
        if s_cand > 1e-15:
            h_candidate = h_candidate / s_cand
            return h_candidate, _ratio(h_candidate)
    except Exception:
        pass
    
    return h, _ratio(h)

def _coordinate_descent(h0, max_iter=500, tol=1e-10, time_limit=None, start_time=None):
    """Coordinate descent polish."""
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
            
            for delta in [0.01, 0.001, 0.0001, 0.00001, -0.01, -0.001, -0.0001, -0.00001]:
                h_new = h.copy()
                h_new[i] = max(0, h[i] + delta)
                
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

def _refined_mian_chowla(n, max_size):
    """Generate refined Mian-Chowla set with local search."""
    sidon = generate_mian_chowla(n, max_size)
    # Try to improve by swapping elements
    for _ in range(20):
        best_swap = None
        best_score = -1
        for i in range(len(sidon)):
            for j in range(n):
                if j in sidon:
                    continue
                new_sidon = sidon[:]
                new_sidon[i] = j
                new_sidon.sort()
                # Check if still Sidon
                used = set()
                valid = True
                for a in range(len(new_sidon)):
                    for b in range(a, len(new_sidon)):
                        s = new_sidon[a] + new_sidon[b]
                        if s in used:
                            valid = False
                            break
                        used.add(s)
                    if not valid:
                        break
                if valid:
                    h = np.zeros(n)
                    for idx in new_sidon:
                        h[idx] = 1.0
                    sc = _ratio(h)
                    if sc > best_score:
                        best_score = sc
                        best_swap = new_sidon
        if best_swap is not None and best_score > 0:
            sidon = best_swap
        else:
            break
    return sidon

def construct():
    start_time = time.time()
    DEADLINE = 145.0
    
    best_h = np.ones(N)
    best_score = _ratio(best_h)
    
    np.random.seed(42)
    
    initializations = []
    
    # Mian-Chowla sets of various sizes
    for size in [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20]:
        s = generate_mian_chowla(N, size)
        h_mc = np.zeros(N)
        for idx in s:
            h_mc[idx] = 1.0
        initializations.append(h_mc)
    
    # Refined Mian-Chowla
    for size in [6, 8, 10, 12, 14]:
        s = _refined_mian_chowla(N, size)
        h_mc = np.zeros(N)
        for idx in s:
            h_mc[idx] = 1.0
        initializations.append(h_mc)
    
    # Random sparse vectors
    for _ in range(40):
        h_sp = np.zeros(N)
        num_ones = np.random.randint(3, 20)
        indices = np.random.choice(N, num_ones, replace=False)
        h_sp[indices] = np.random.uniform(0.2, 3.0, num_ones)
        initializations.append(h_sp)
    
    # Random dense vectors
    for _ in range(20):
        h_rd = np.random.uniform(0, 1, N)
        initializations.append(h_rd)
    
    # Structured patterns
    h_alt = np.zeros(N)
    h_alt[::2] = 1.0
    initializations.append(h_alt)
    
    h_sp3 = np.zeros(N)
    h_sp3[::3] = 1.0
    initializations.append(h_sp3)
    
    h_sp4 = np.zeros(N)
    h_sp4[::4] = 1.0
    initializations.append(h_sp4)
    
    h_sp5 = np.zeros(N)
    h_sp5[::5] = 1.0
    initializations.append(h_sp5)
    
    # Gaussian and smooth shapes
    t = np.linspace(-1, 1, N)
    for width in [0.2, 0.3, 0.4, 0.5, 0.7, 1.0]:
        initializations.append(np.exp(-t**2/width))
    
    initializations.append(np.maximum(1 - t**2, 0.01))
    initializations.append(np.maximum(t**2 + 0.1, 0.01))
    initializations.append(np.abs(t) + 0.1)
    
    # Bimodal
    for sep in [0.3, 0.5, 0.7]:
        initializations.append(np.maximum(1.0 + 2.0*np.exp(-((t-sep)**2)/0.1) + 2.0*np.exp(-((t+sep)**2)/0.1), 0.05))
    
    # Centered spike
    h_center = np.zeros(N)
    h_center[N//2] = 1.0
    initializations.append(h_center)
    
    # Edge spikes
    h_edges = np.zeros(N)
    h_edges[0] = 1.0
    h_edges[N-1] = 1.0
    initializations.append(h_edges)
    
    # Quarter positions
    h_q = np.zeros(N)
    h_q[N//4] = 1.0
    h_q[3*N//4] = 1.0
    initializations.append(h_q)
    
    # Process each initialization
    for h0 in initializations:
        if time.time() - start_time > DEADLINE - 10:
            break
        
        h0 = np.maximum(h0, 0)
        s0 = np.sum(h0)
        if s0 < 1e-15:
            continue
        h0 = h0 / s0
        
        # Strategy 1: Iterative reweighting
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
        
        # Strategy 2: Epigraph/smooth optimization
        if time.time() - start_time < DEADLINE - 20:
            try:
                h_epi, score_epi = _epigraph_optimize(
                    h0, time_limit=DEADLINE - 15, start_time=start_time
                )
                if score_epi > best_score:
                    best_score = score_epi
                    best_h = h_epi.copy()
            except Exception:
                pass
    
    # Final polish: coordinate descent
    if time.time() - start_time < DEADLINE - 5:
        try:
            h_polished, score_polished = _coordinate_descent(
                best_h, max_iter=400, tol=1e-10,
                time_limit=DEADLINE - 2, start_time=start_time
            )
            if score_polished > best_score:
                best_score = score_polished
                best_h = h_polished.copy()
        except Exception:
            pass
    
    # One more iterative reweighting pass
    if time.time() - start_time < DEADLINE - 3:
        try:
            h_final, score_final = _iterative_reweighting(
                best_h, max_iter=150, tol=1e-9,
                time_limit=DEADLINE - 1, start_time=start_time
            )
            if score_final > best_score:
                best_score = score_final
                best_h = h_final.copy()
        except Exception:
            pass
    
    # Final coordinate descent polish
    if time.time() - start_time < DEADLINE - 1:
        try:
            h_final2, score_final2 = _coordinate_descent(
                best_h, max_iter=200, tol=1e-11,
                time_limit=DEADLINE - 0.5, start_time=start_time
            )
            if score_final2 > best_score:
                best_score = score_final2
                best_h = h_final2.copy()
        except Exception:
            pass
    
    return best_h.tolist()
