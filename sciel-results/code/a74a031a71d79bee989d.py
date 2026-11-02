"""Optimized construction for the signed autocorrelation C3-prime task.
Uses max-pooling guided sign-flip search: directly target the current peak
autoconvolution coefficient and find the best single flip to reduce it.
"""
import time
import numpy as np

N = 400


def _autoconv(x):
    """Full linear autoconvolution of x via FFT (zero-padded)."""
    n = len(x)
    size = 1
    while size < 2 * n - 1:
        size <<= 1
    F = np.fft.rfft(x, size)
    c = np.fft.irfft(F * np.conj(F), size)
    return c[:2 * n - 1]


def _score(x):
    """C3' = max|autoconv| / (integral)^2. Integral = sum(x)."""
    s = x.sum()
    if abs(s) < 1e-12:
        return np.inf
    c = _autoconv(x)
    return float(np.max(np.abs(c)) / (s * s))


def _score_and_c(x):
    """Return (score, autoconv) in one pass."""
    s = x.sum()
    c = _autoconv(x)
    if abs(s) < 1e-12:
        return np.inf, c
    return float(np.max(np.abs(c)) / (s * s)), c


def _max_pooling_flip_search(x, deadline, max_iter=200000, patience=5000):
    """
    Max-pooling guided sign-flip local search.
    At each step, identify the lag with maximum |autoconv|, then evaluate
    candidate flips that most directly reduce that peak.
    """
    x = x.copy().astype(np.float64)
    n = len(x)
    best_score, c = _score_and_c(x)
    best_x = x.copy()
    
    # Track the peak lag
    peak_lag = int(np.argmax(np.abs(c)))
    
    no_improve = 0
    rng = np.random.default_rng(777)
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Get current peak
        ac_abs = np.abs(c)
        peak_val = ac_abs[peak_lag]
        
        # Candidate flips: positions that contribute to the peak lag
        # c[peak_lag] = sum_k x[k] * x[k + peak_lag]
        # Flipping x[j] changes c[peak_lag] by -2*x[j]*x[j+peak_lag] (if in range)
        # We want flips that reduce |c[peak_lag]|
        
        candidates = []
        pl = peak_lag
        
        # Positions j where both j and j+pl are in [0, n)
        # c[pl] = sum_{j=0}^{n-1-pl} x[j]*x[j+pl]
        # Flipping x[j]: delta_c = -2*x[j]*x[j+pl]
        # We want delta_c to reduce |c[pl]|, i.e., sign(delta_c) == -sign(c[pl])
        
        if c[pl] > 0:
            # Want delta_c < 0, i.e., x[j]*x[j+pl] > 0
            for j in range(n - pl):
                if x[j] * x[j + pl] > 0:
                    candidates.append(j)
        else:
            # Want delta_c > 0, i.e., x[j]*x[j+pl] < 0
            for j in range(n - pl):
                if x[j] * x[j + pl] < 0:
                    candidates.append(j)
        
        # Also consider flips that affect other high lags (secondary peaks)
        # Add some random candidates for exploration
        if len(candidates) < 10:
            for _ in range(10):
                candidates.append(int(rng.integers(n)))
        
        # Sample up to 30 candidates to keep it fast
        if len(candidates) > 30:
            idx = rng.choice(len(candidates), size=30, replace=False)
            candidates = [candidates[i] for i in idx]
        
        # Evaluate candidates: flip each, compute new peak, keep best
        best_candidate = -1
        best_candidate_score = best_score
        
        for j in candidates:
            x[j] = -x[j]
            # Quick check: recompute autoconv
            c_new = _autoconv(x)
            s_new = x.sum()
            if abs(s_new) < 1e-12:
                x[j] = -x[j]
                continue
            sc_new = float(np.max(np.abs(c_new)) / (s_new * s_new))
            if sc_new < best_candidate_score:
                best_candidate_score = sc_new
                best_candidate = j
                best_c = c_new
            x[j] = -x[j]  # revert
        
        if best_candidate >= 0 and best_candidate_score < best_score:
            x[best_candidate] = -x[best_candidate]
            best_score = best_candidate_score
            c = best_c
            best_x = x.copy()
            no_improve = 0
            # Update peak lag
            peak_lag = int(np.argmax(np.abs(c)))
        else:
            no_improve += 1
            # If stuck for a while, try a random flip to escape
            if no_improve % 100 == 0:
                j = int(rng.integers(n))
                x[j] = -x[j]
                c = _autoconv(x)
                s = x.sum()
                if abs(s) > 1e-12:
                    sc = float(np.max(np.abs(c)) / (s * s))
                    if sc < best_score:
                        best_score = sc
                        best_x = x.copy()
                        no_improve = 0
                        peak_lag = int(np.abs(c).argmax())
                    else:
                        x[j] = -x[j]  # revert
                else:
                    x[j] = -x[j]
            
            if no_improve > patience:
                break
    
    return best_x, best_score


def _multi_start_max_pooling(deadline):
    """Multiple random starts with max-pooling guided search."""
    rng = np.random.default_rng(42)
    best_x = np.ones(N, dtype=np.float64)
    best_score = _score(best_x)
    
    # Generate diverse starts
    starts = []
    
    # All ones
    starts.append(np.ones(N, dtype=np.float64))
    
    # Alternating
    starts.append(np.tile([1.0, -1.0], N // 2).astype(np.float64))
    
    # Random ±1
    for _ in range(15):
        starts.append(rng.choice([-1.0, 1.0], size=N).astype(np.float64))
    
    # Block patterns
    for bs in [4, 8, 16, 32, 64, 100]:
        if N % bs == 0:
            nb = N // bs
            bs_signs = rng.choice([-1.0, 1.0], size=nb)
            starts.append(np.repeat(bs_signs, bs).astype(np.float64))
    
    # Low-frequency sinusoidal patterns
    for freq in [1, 2, 3, 5, 7, 11]:
        t = np.arange(N)
        starts.append(np.sign(np.sin(2 * np.pi * freq * t / N)).astype(np.float64))
    
    # Hadamard-like (Walsh functions)
    for k in [1, 3, 5, 7, 15, 31]:
        t = np.arange(N)
        walsh = np.sign(np.sin(np.pi * k * t / N)).astype(np.float64)
        starts.append(walsh)
    
    # Time budget per start
    n_starts = len(starts)
    time_per_start = (deadline - time.time()) / (n_starts + 1)
    
    for s in starts:
        remaining = deadline - time.time()
        if remaining < 2.0:
            break
        
        alloc = min(time_per_start, remaining - 1.0)
        if alloc < 0.5:
            break
        
        x_cur, sc_cur = _max_pooling_flip_search(
            s, time.time() + alloc, 
            max_iter=50000, patience=2000
        )
        if sc_cur < best_score:
            best_score = sc_cur
            best_x = x_cur.copy()
    
    return best_x, best_score


def _refine_with_adaptive_candidates(x, deadline, max_iter=100000):
    """
    Refined search with adaptive candidate set:
    - Focus on positions that appear in multiple top autoconv lags
    - Use frequency-domain insight: high autoconv at certain lags suggests
      periodic structure; target breaks in that periodicity
    """
    x = x.copy().astype(np.float64)
    n = len(x)
    best_score, c = _score_and_c(x)
    best_x = x.copy()
    rng = np.random.default_rng(999)
    
    # Identify top 5 lags
    ac_abs = np.abs(c)
    top_lags = np.argsort(ac_abs)[::-1][:5]
    
    # Build a "hotspot" score for each position: sum of contributions to top lags
    hotspot = np.zeros(n)
    for pl in top_lags:
        pl = int(pl)
        if pl < n:
            for j in range(n - pl):
                hotspot[j] += abs(c[pl]) * abs(x[j] * x[j + pl]) / (n - pl)
    
    # Sort positions by hotspot score
    hot_order = np.argsort(hotspot)[::-1]
    
    no_improve = 0
    patience = 5000
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Get current top lag
        ac_abs = np.abs(c)
        peak_lag = int(np.argmax(ac_abs))
        
        # Build candidate set: top hotspot positions + positions affecting peak
        candidates = set()
        
        # Positions affecting the peak lag
        pl = peak_lag
        if c[pl] > 0:
            for j in range(n - pl):
                if x[j] * x[j + pl] > 0:
                    candidates.add(j)
        else:
            for j in range(n - pl):
                if x[j] * x[j + pl] < 0:
                    candidates.add(j)
        
        # Add top hotspot positions
        for j in hot_order[:20]:
            candidates.add(int(j))
        
        # Add some random
        for _ in range(10):
            candidates.add(int(rng.integers(n)))
        
        candidates = list(candidates)
        
        if len(candidates) > 50:
            idx = rng.choice(len(candidates), size=50, replace=False)
            candidates = [candidates[i] for i in idx]
        
        best_candidate = -1
        best_candidate_score = best_score
        best_c = None
        
        for j in candidates:
            x[j] = -x[j]
            c_new = _autoconv(x)
            s_new = x.sum()
            if abs(s_new) < 1e-12:
                x[j] = -x[j]
                continue
            sc_new = float(np.max(np.abs(c_new)) / (s_new * s_new))
            if sc_new < best_candidate_score:
                best_candidate_score = sc_new
                best_candidate = j
                best_c = c_new
            x[j] = -x[j]
        
        if best_candidate >= 0 and best_candidate_score < best_score:
            x[best_candidate] = -x[best_candidate]
            best_score = best_candidate_score
            c = best_c
            best_x = x.copy()
            no_improve = 0
            # Update hotspot occasionally
            if no_improve % 500 == 0 or it % 1000 == 0:
                ac_abs = np.abs(c)
                top_lags = np.argsort(ac_abs)[::-1][:5]
                hotspot = np.zeros(n)
                for pl in top_lags:
                    pl = int(pl)
                    if pl < n:
                        for j in range(n - pl):
                            hotspot[j] += abs(c[pl]) * abs(x[j] * x[j + pl]) / (n - pl)
                hot_order = np.argsort(hotspot)[::-1]
        else:
            no_improve += 1
            if no_improve % 200 == 0:
                j = int(rng.integers(n))
                x[j] = -x[j]
                c = _autoconv(x)
                s = x.sum()
                if abs(s) > 1e-12:
                    sc = float(np.max(np.abs(c)) / (s * s))
                    if sc < best_score:
                        best_score = sc
                        best_x = x.copy()
                        no_improve = 0
                    else:
                        x[j] = -x[j]
                else:
                    x[j] = -x[j]
            
            if no_improve > patience:
                break
    
    return best_x, best_score


def construct():
    """Return the sequence of signed step heights minimizing C3'."""
    deadline = time.time() + 140.0
    
    # Phase 1: Multi-start max-pooling guided search
    best_x, best_score = _multi_start_max_pooling(deadline - 30)
    
    # Phase 2: Refined search with adaptive candidates on the best result
    if time.time() < deadline - 10:
        x_ref, sc_ref = _refine_with_adaptive_candidates(
            best_x, deadline - 5, max_iter=150000
        )
        if sc_ref < best_score:
            best_x = x_ref
            best_score = sc_ref
    
    # Phase 3: Final quick polish
    if time.time() < deadline - 3:
        x_fin, sc_fin = _max_pooling_flip_search(
            best_x, deadline - 1, max_iter=30000, patience=1000
        )
        if sc_fin < best_score:
            best_x = x_fin
            best_score = sc_fin
    
    return best_x
