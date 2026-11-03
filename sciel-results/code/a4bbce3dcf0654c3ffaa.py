import numpy as np
import time

N = 50
DELTA = 1.0 / N

def _autoconv_all(h):
    """Compute all autoconvolution values at breakpoints for step function."""
    n = len(h)
    num_points = 2 * n - 1
    ff_vals = np.zeros(num_points)
    
    for k in range(num_points):
        total = 0.0
        i_min = max(0, k - (n - 1))
        i_max = min(n - 1, k)
        for i in range(i_min, i_max + 1):
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
    ff_vals = _autoconv_all(h)
    sup_ff = np.max(ff_vals)
    if sup_ff < 1e-15:
        return 0.0
    return sum_h2 / (sum_h * sup_ff)

def _iterative_reweighting(h0, max_iter=300, time_limit=None, start_time=None):
    """Iterative reweighting on fixed support to optimize heights."""
    h = np.maximum(h0, 0).copy()
    support = np.where(h > 1e-10)[0]
    
    if len(support) == 0:
        return h, 0.0
    
    # Normalize to sum=1
    s = np.sum(h)
    if s < 1e-15:
        return h, 0.0
    h = h / s
    
    best_h = h.copy()
    best_score = _ratio(h)
    
    for it in range(max_iter):
        if time_limit is not None and start_time is not None:
            if time.time() - start_time > time_limit:
                break
        
        # Find the current peak of autoconvolution
        ff_vals = _autoconv_all(h)
        peak_idx = np.argmax(ff_vals)
        ff_peak = ff_vals[peak_idx]
        
        if ff_peak < 1e-15:
            break
        
        # Compute gradient of log ratio w.r.t. log h
        # log R = log(||h||²₂) - log(||h||₁) - log(||h*h||∞)
        # d/dlog(h_i) log R = 2*h_i/||h||²₂ - 1/||h||₁ - (d||h*h||∞/dh_i)/||h*h||∞
        
        sum_h2 = np.sum(h**2)
        sum_h = np.sum(h)
        
        # Compute d(ff_peak)/dh_i for the peak
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
        
        # Multiplicative update with line search
        eta = 0.1
        for _ in range(15):
            h_new = h * np.exp(eta * grad_log)
            h_new = np.maximum(h_new, 0)
            
            s_new = np.sum(h_new)
            if s_new < 1e-15:
                break
            h_new = h_new / s_new
            
            score_new = _ratio(h_new)
            if score_new > best_score + 1e-10:
                h = h_new
                best_h = h.copy()
                best_score = score_new
                eta = min(eta * 1.5, 1.0)
                break
            else:
                eta /= 2.0
        else:
            # No improvement with this eta, try smaller
            if eta < 1e-6:
                break
    
    return best_h, best_score

def _coordinate_descent(h0, max_iter=200, time_limit=None, start_time=None):
    """Coordinate descent polish: optimize one coordinate at a time."""
    h = np.maximum(h0, 0).copy()
    n = len(h)
    support = np.where(h > 1e-10)[0]
    
    if len(support) == 0:
        return h, 0.0
    
    best_h = h.copy()
    best_score = _ratio(h)
    
    for it in range(max_iter):
        if time_limit is not None and start_time is not None:
            if time.time() - start_time > time_limit:
                break
        
        improved = False
        for i in support:
            if time_limit is not None and start_time is not None:
                if time.time() - start_time > time_limit:
                    return best_h, best_score
            
            h_old = h[i]
            # Try different values for h[i]
            for mult in [0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.5, 2.0]:
                h[i] = h_old * mult
                if h[i] < 0:
                    h[i] = 0
                s = np.sum(h)
                if s < 1e-15:
                    h[i] = h_old
                    continue
                h_norm = h / s
                sc = _ratio(h_norm)
                if sc > best_score + 1e-10:
                    best_score = sc
                    best_h = h_norm.copy()
                    improved = True
                    break
            h[i] = h_old
        
        if not improved:
            break
    
    return best_h, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    best_h = np.zeros(N)
    best_score = 0.0
    
    def try_support(S, max_iter_rw=200, max_iter_cd=100):
        nonlocal best_h, best_score
        if time.time() > deadline - 5.0:
            return
        
        k = len(S)
        if k == 0:
            return
        
        # Initialize with equal weights
        h0 = np.zeros(N)
        h0[S] = 1.0 / k
        
        # Iterative reweighting
        h_rw, s_rw = _iterative_reweighting(
            h0, max_iter=max_iter_rw, 
            time_limit=deadline - 5.0 - start_time,
            start_time=start_time
        )
        
        if s_rw > best_score:
            best_score = s_rw
            best_h = h_rw.copy()
        
        # Coordinate descent polish
        if time.time() < deadline - 3.0:
            h_cd, s_cd = _coordinate_descent(
                h_rw, max_iter=max_iter_cd,
                time_limit=deadline - 3.0 - start_time,
                start_time=start_time
            )
            if s_cd > best_score:
                best_score = s_cd
                best_h = h_cd.copy()
    
    # Generate candidate supports
    # Strategy: try various sizes and patterns
    
    # 1. Small supports: try all pairs and triples
    for i in range(N):
        for j in range(i+1, N):
            if time.time() > deadline - 30.0:
                break
            S = [i, j]
            try_support(S, max_iter_rw=100, max_iter_cd=50)
        if time.time() > deadline - 30.0:
            break
    
    if time.time() > deadline - 30.0:
        return list(best_h)
    
    # 2. Greedy Sidon-like sets of various sizes
    def find_sidon_set(k, max_attempts=5):
        """Find a k-element set with low autoconvolution peak."""
        best_S = None
        best_mc = np.inf
        
        for attempt in range(max_attempts):
            S = [0]  # Start with 0
            for _ in range(k - 1):
                best_cand = None
                best_mc_cand = np.inf
                for j in range(1, N):
                    if j in S:
                        continue
                    S_test = sorted(S + [j])
                    h_tmp = np.zeros(N)
                    h_tmp[S_test] = 1.0
                    ff_vals = _autoconv_all(h_tmp)
                    mc = np.max(ff_vals)
                    if mc < best_mc_cand:
                        best_mc_cand = mc
                        best_cand = j
                if best_cand is not None:
                    S.append(best_cand)
            
            S = sorted(S)
            h_tmp = np.zeros(N)
            h_tmp[S] = 1.0
            ff_vals = _autoconv_all(h_tmp)
            mc = np.max(ff_vals)
            
            # Local search: try swapping one element
            improved = True
            while improved and time.time() < deadline - 10.0:
                improved = False
                for idx in range(len(S)):
                    for j in range(N):
                        if j in S:
                            continue
                        S_new = sorted(S[:idx] + [j] + S[idx+1:])
                        h_tmp = np.zeros(N)
                        h_tmp[S_new] = 1.0
                        ff_vals = _autoconv_all(h_tmp)
                        mc_new = np.max(ff_vals)
                        if mc_new < mc - 1e-10:
                            mc = mc_new
                            S = S_new
                            improved = True
                            break
                    if improved:
                        break
            
            if mc < best_mc:
                best_mc = mc
                best_S = S
        
        return best_S
    
    # Try various support sizes
    for k in range(3, 16):
        if time.time() > deadline - 20.0:
            break
        
        S = find_sidon_set(k, max_attempts=3)
        if S is not None:
            try_support(S, max_iter_rw=150, max_iter_cd=80)
    
    # 3. Random supports
    rng = np.random.RandomState(42)
    for _ in range(30):
        if time.time() > deadline - 10.0:
            break
        k = rng.randint(3, 12)
        S = sorted(rng.choice(N, k, replace=False).tolist())
        try_support(S, max_iter_rw=100, max_iter_cd=50)
    
    # 4. Structured patterns: arithmetic progressions, etc.
    for start in range(0, N, 2):
        for step in range(1, 5):
            if time.time() > deadline - 5.0:
                break
            S = list(range(start, N, step))
            if len(S) >= 2:
                try_support(S, max_iter_rw=100, max_iter_cd=50)
    
    return list(best_h)
