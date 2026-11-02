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

def _find_maximal_independent_sets(n, max_sets=200, max_size=15, time_budget=8.0):
    """Find maximal independent sets in the collision graph using greedy randomized search.
    
    Collision graph: vertices 0..n-1, edge (i,j) iff i==j or |i-j| equals |i'-j'| for some other pair.
    We want large sets with all pairwise differences distinct (Sidon sets).
    """
    start = time.time()
    sets_found = []
    
    # Precompute difference conflicts: for each difference d, which pairs have that difference
    # Two indices i,j conflict if they share a difference with another pair
    # Actually: i and j are in the same Sidon set iff all pairwise differences among the set are distinct
    # So the "collision graph" has edge (i,j) if adding j to a set containing i would create a duplicate difference
    # But this depends on the whole set, not just the pair. Let's use a different approach.
    
    # Greedy randomized: try to build a Sidon set by adding elements in random order
    np.random.seed(12345)
    
    for trial in range(2000):
        if time.time() - start > time_budget:
            break
        
        # Random permutation of indices
        perm = np.random.permutation(n)
        
        sidon = []
        diffs = set()
        
        for idx in perm:
            # Check if adding idx keeps all pairwise differences distinct
            valid = True
            for existing in sidon:
                d = abs(idx - existing)
                if d in diffs:
                    valid = False
                    break
            if valid:
                # Add all new differences
                for existing in sidon:
                    diffs.add(abs(idx - existing))
                sidon.append(idx)
            
            if len(sidon) >= max_size:
                break
        
        if len(sidon) >= 5:
            sidon.sort()
            key = tuple(sidon)
            if key not in sets_found:
                sets_found.append(key)
                if len(sets_found) >= max_sets:
                    break
    
    return sets_found

def _optimize_weights_on_support(support, h0=None, time_limit=5.0, start_time=None):
    """Optimize weights on a fixed Sidon support to maximize ||w||_2^2 / (||w||_1 * peak).
    
    For a Sidon set, the autoconvolution peak is bounded, so we want to maximize
    ||w||_2^2 / ||w||_1 subject to the autoconvolution peak constraint.
    This is equivalent to making weights as "peaked" as possible while keeping
    the autoconvolution peak controlled.
    """
    n = len(support)
    if n == 0:
        return np.zeros(N), 0.0
    
    if start_time is None:
        start_time = time.time()
    
    # Initial weights: uniform on support
    if h0 is None:
        w = np.ones(n) / n
    else:
        w = np.array([h0[i] for i in support], dtype=float)
        s = np.sum(w)
        if s < 1e-15:
            w = np.ones(n) / n
        else:
            w = w / s
    
    best_w = w.copy()
    best_score = 0.0
    
    # Build the full vector and compute initial score
    def _full_vec(w_local):
        hv = np.zeros(N)
        for k, idx in enumerate(support):
            hv[idx] = w_local[k]
        return hv
    
    def _score(w_local):
        hv = _full_vec(w_local)
        return _ratio(hv)
    
    best_score = _score(w)
    
    # Use scipy minimize to maximize ||w||_2^2 / ||w||_1 / peak
    # Since peak is fixed by the Sidon structure (approximately), maximize ||w||_2^2 / ||w||_1
    # which is maximized when all weight is on one element, but that increases peak.
    # Actually for Sidon sets, the peak at k=i+j is h[i]*h[j]*delta (for i!=j) or h[i]^2*delta (for i=j)
    # The maximum peak is max over all pairs of h[i]*h[j]*delta
    # So we want to maximize ||w||_2^2 / (||w||_1 * max_pair_product * delta)
    # = (sum w_i^2) / (sum w_i * max_{i,j} w_i*w_j * delta)
    
    # This is a non-trivial optimization. Let's use gradient-based approach.
    
    def neg_objective(w_local):
        w_local = np.maximum(w_local, 0)
        s = np.sum(w_local)
        if s < 1e-15:
            return 0.0
        w_norm = w_local / s
        hv = _full_vec(w_norm)
        return -_ratio(hv)
    
    try:
        result = minimize(neg_objective, w.copy(), method='L-BFGS-B',
                         bounds=[(0, None)] * n,
                         options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10})
        
        w_candidate = np.maximum(result.x, 0)
        s = np.sum(w_candidate)
        if s > 1e-15:
            w_candidate = w_candidate / s
            sc = _score(w_candidate)
            if sc > best_score:
                best_score = sc
                best_w = w_candidate.copy()
    except Exception:
        pass
    
    # Also try: put all weight on the single best element
    # For a Sidon set, peak = max(w_i * w_j) * delta
    # If we put all weight on one element: peak = w_1^2 * delta, ratio = 1/(w_1 * delta) = 1/delta (normalized)
    # Actually with normalization sum=1: ratio = 1/(1*delta) = 1/delta = N = 50? No...
    # Let me just evaluate a few structured weight patterns
    
    # Pattern 1: uniform (already done)
    # Pattern 2: geometric decay from center
    center = n // 2
    for decay in [0.5, 0.7, 0.8, 0.9]:
        w_pat = np.array([decay ** abs(i - center) for i in range(n)])
        s = np.sum(w_pat)
        w_pat /= s
        sc = _score(w_pat)
        if sc > best_score:
            best_score = sc
            best_w = w_pat.copy()
    
    # Pattern 3: endpoint-heavy
    for alpha in [0.3, 0.5, 0.7]:
        w_pat = np.ones(n)
        w_pat[0] = alpha
        w_pat[-1] = alpha
        s = np.sum(w_pat)
        w_pat /= s
        sc = _score(w_pat)
        if sc > best_score:
            best_score = sc
            best_w = w_pat.copy()
    
    # Pattern 4: two large endpoints
    for alpha in [0.5, 1.0, 2.0, 3.0]:
        w_pat = np.zeros(n)
        w_pat[0] = alpha
        w_pat[-1] = alpha
        if n > 1:
            w_pat[1] = 1.0
            w_pat[-2] = 1.0
        s = np.sum(w_pat)
        if s > 0:
            w_pat /= s
            sc = _score(w_pat)
            if sc > best_score:
                best_score = sc
                best_w = w_pat.copy()
    
    # Build final vector
    hv = _full_vec(best_w)
    return hv, best_score

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
            
            for delta in [0.01, 0.001, 0.0001, -0.01, -0.001, -0.0001]:
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

def construct():
    start_time = time.time()
    DEADLINE = 145.0

    best_h = np.ones(N)
    best_score = _ratio(best_h)

    np.random.seed(42)

    # Phase 1: Find maximal Sidon sets (maximal independent sets in collision graph)
    sidon_sets = _find_maximal_independent_sets(N, max_sets=100, max_size=15, time_budget=10.0)
    
    # Also add Mian-Chowla sets of various sizes
    for size in [6, 8, 10, 12, 14, 16]:
        s = generate_mian_chowla(N, size)
        key = tuple(sorted(s))
        if key not in sidon_sets:
            sidon_sets.append(key)
    
    # Sort by size descending - larger sets are better
    sidon_sets.sort(key=lambda x: -len(x))
    
    # Phase 2: For each Sidon set, optimize weights
    for sidon in sidon_sets:
        if time.time() - start_time > DEADLINE - 15:
            break
        
        if len(sidon) < 5:
            continue
        
        # Skip if too many sets already processed
        elapsed = time.time() - start_time
        remaining = DEADLINE - 15 - elapsed
        if remaining < 0.5:
            break
        
        # Time budget per set
        t_budget = min(3.0, remaining / max(1, len(sidon_sets) - sidon_sets.index(sidon)))
        
        try:
            h_candidate, score_candidate = _optimize_weights_on_support(
                list(sidon), h0=None, time_limit=t_budget, start_time=start_time
            )
            if score_candidate > best_score:
                best_score = score_candidate
                best_h = h_candidate.copy()
        except Exception:
            pass
    
    # Phase 3: Also try some random sparse initializations with iterative reweighting
    for _ in range(20):
        if time.time() - start_time > DEADLINE - 10:
            break
        
        h_sp = np.zeros(N)
        num_ones = np.random.randint(5, 15)
        indices = np.random.choice(N, num_ones, replace=False)
        h_sp[indices] = np.random.uniform(0.3, 2.0, num_ones)
        s = np.sum(h_sp)
        if s > 0:
            h_sp /= s
        
        try:
            h_rew, score_rew = _iterative_reweighting(
                h_sp, max_iter=100, tol=1e-8,
                time_limit=DEADLINE - 10, start_time=start_time
            )
            if score_rew > best_score:
                best_score = score_rew
                best_h = h_rew.copy()
        except Exception:
            pass
    
    # Phase 4: Final polish
    if time.time() - start_time < DEADLINE - 5:
        try:
            h_polished, score_polished = _coordinate_descent(
                best_h, max_iter=200, tol=1e-10,
                time_limit=DEADLINE - 3, start_time=start_time
            )
            if score_polished > best_score:
                best_score = score_polished
                best_h = h_polished.copy()
        except Exception:
            pass
    
    # Phase 5: One more iterative reweighting pass
    if time.time() - start_time < DEADLINE - 3:
        try:
            h_final, score_final = _iterative_reweighting(
                best_h, max_iter=80, tol=1e-9,
                time_limit=DEADLINE - 1, start_time=start_time
            )
            if score_final > best_score:
                best_score = score_final
                best_h = h_final.copy()
        except Exception:
            pass
    
    return best_h.tolist()
